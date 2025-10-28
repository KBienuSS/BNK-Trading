# trading_bot_ml.py
import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLEnhancedTradingBot(OptimizedStrategyBot):
    def __init__(self, initial_capital=10000, leverage=10):
        super().__init__(initial_capital, leverage)
        
        # ML Components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_ml_trained = False
        self.training_data = []
        self.ml_confidence_threshold = 0.70
        
        # ML Features storage
        self.feature_columns = [
            'rsi', 'volume_ratio', 'price_vs_sma20', 'price_vs_sma50',
            'macd_histogram', 'ema_cross', 'atr', 'momentum',
            'support_distance', 'resistance_distance'
        ]
        
        self.logger.info("🧠 ML-ENHANCED TRADING BOT INITIALIZED")
        self.logger.info("📊 ML will enhance signals without changing strategy")
        
        # Try to load existing model
        self.load_ml_model()

    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced features for ML model"""
        try:
            # Basic indicators (already in parent class)
            df = self.calculate_advanced_indicators(df)
            
            # Additional ML features
            df['atr'] = self.calculate_atr(df, 14)  # Average True Range
            df['momentum'] = df['close'].pct_change(5)  # 5-period momentum
            df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
            
            # Support/Resistance distances
            df['support_distance'] = (df['close'] - df['support']) / df['close'] * 100
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close'] * 100
            
            # Volatility
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean() * 100
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ML features: {e}")
            return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr

    def prepare_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model prediction"""
        try:
            if len(df) < 50:
                return None
                
            current = df.iloc[-1]
            features = []
            
            # RSI
            features.append(current['rsi_14'] if not pd.isna(current['rsi_14']) else 50)
            
            # Volume ratio
            features.append(current['volume_ratio'] if not pd.isna(current['volume_ratio']) else 1)
            
            # Price vs SMA
            features.append(current['price_vs_sma20'] if not pd.isna(current['price_vs_sma20']) else 0)
            features.append(current['price_vs_sma50'] if not pd.isna(current['price_vs_sma50']) else 0)
            
            # MACD
            features.append(current['macd_histogram'] if not pd.isna(current['macd_histogram']) else 0)
            
            # EMA Cross
            features.append(1 if current['ema_12'] > current['ema_26'] else 0)
            
            # ATR (volatility)
            features.append(current['atr'] if 'atr' in df.columns and not pd.isna(current['atr']) else 0)
            
            # Momentum
            features.append(current['momentum'] * 100 if 'momentum' in df.columns and not pd.isna(current['momentum']) else 0)
            
            # Support/Resistance distances
            features.append(current['support_distance'] if 'support_distance' in df.columns and not pd.isna(current['support_distance']) else 0)
            features.append(current['resistance_distance'] if 'resistance_distance' in df.columns and not pd.isna(current['resistance_distance']) else 0)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing ML features: {e}")
            return None

    def create_training_label(self, df: pd.DataFrame, future_bars: int = 10) -> int:
        """Create label for training: 1 if price goes up, 0 if down"""
        try:
            current_idx = len(df) - future_bars - 1
            if current_idx < 0:
                return 0
                
            current_price = df.iloc[current_idx]['close']
            future_max = df.iloc[current_idx + 1:current_idx + future_bars + 1]['close'].max()
            future_min = df.iloc[current_idx + 1:current_idx + future_bars + 1]['close'].min()
            
            # Label 1 if price increases by more than it decreases
            price_increase = (future_max - current_price) / current_price
            price_decrease = (current_price - future_min) / current_price
            
            return 1 if price_increase > price_decrease else 0
            
        except Exception as e:
            self.logger.error(f"❌ Error creating training label: {e}")
            return 0

    def collect_training_data(self, symbol: str):
        """Collect training data from historical trades"""
        try:
            df = self.get_binance_klines(symbol, '3m', 200)
            if df is None or len(df) < 100:
                return
                
            df = self.calculate_advanced_features(df)
            
            # Create multiple training samples from historical data
            for i in range(50, len(df) - 20):
                features = self.prepare_ml_features(df.iloc[:i+1])
                if features is not None:
                    label = self.create_training_label(df.iloc[:i+20])
                    self.training_data.append((features.flatten(), label))
                    
            self.logger.info(f"📚 Collected {len(self.training_data)} training samples for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting training data for {symbol}: {e}")

    def train_ml_model(self):
        """Train the ML model on collected data"""
        try:
            if len(self.training_data) < 100:
                self.logger.warning("🤖 Not enough training data yet")
                return False
                
            # Prepare data
            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.ml_model.score(X_train_scaled, y_train)
            test_score = self.ml_model.score(X_test_scaled, y_test)
            
            self.is_ml_trained = True
            
            self.logger.info(f"🎯 ML Model Trained - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
            
            # Save model
            self.save_ml_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error training ML model: {e}")
            return False

    def ml_predict(self, symbol: str) -> Tuple[float, float]:
        """Get ML prediction and confidence"""
        try:
            if not self.is_ml_trained or self.ml_model is None:
                return 0.5, 0.0  # Neutral prediction if no model
                
            df = self.get_binance_klines(symbol, '3m', 100)
            if df is None or len(df) < 50:
                return 0.5, 0.0
                
            df = self.calculate_advanced_features(df)
            features = self.prepare_ml_features(df)
            
            if features is None:
                return 0.5, 0.0
                
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
            prediction = self.ml_model.predict(features_scaled)[0]
            
            # Confidence is the probability of the predicted class
            confidence = float(np.max(prediction_proba))
            
            return float(prediction), confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error in ML prediction for {symbol}: {e}")
            return 0.5, 0.0

    def generate_ml_enhanced_signal(self, symbol: str):
        """Generate signal enhanced with ML predictions"""
        # Get original signal from parent class
        original_signal, original_confidence = self.generate_optimized_signal(symbol)
        
        # If original strategy says HOLD, don't use ML
        if original_signal == "HOLD":
            return "HOLD", original_confidence
        
        # Get ML prediction
        ml_prediction, ml_confidence = self.ml_predict(symbol)
        
        # ML enhancement logic
        ml_boost = 0.0
        
        if ml_confidence > self.ml_confidence_threshold:
            if ml_prediction == 1:  # ML predicts UP
                ml_boost = ml_confidence * 0.3  # Boost up to 30%
                self.logger.info(f"🧠 ML BULLISH - Confidence: {ml_confidence:.1%} - Boosting signal")
            else:  # ML predicts DOWN
                ml_boost = -ml_confidence * 0.5  # Reduce confidence significantly
                self.logger.info(f"🧠 ML BEARISH - Confidence: {ml_confidence:.1%} - Reducing signal")
        
        # Apply ML boost to original confidence
        enhanced_confidence = max(0.0, min(1.0, original_confidence + ml_boost))
        
        # Final decision
        if enhanced_confidence >= 0.65:  # Same threshold as original
            final_signal = "LONG"
            final_confidence = enhanced_confidence
        else:
            final_signal = "HOLD"
            final_confidence = enhanced_confidence
            
        if ml_confidence > 0.7:
            self.logger.info(f"🎯 ML-ENHANCED: {symbol} - Original: {original_confidence:.1%} → Enhanced: {final_confidence:.1%}")
        
        return final_signal, final_confidence

    def save_ml_model(self):
        """Save ML model and scaler to disk"""
        try:
            if self.ml_model is not None:
                joblib.dump(self.ml_model, 'ml_trading_model.pkl')
                joblib.dump(self.scaler, 'ml_scaler.pkl')
                joblib.dump(self.training_data, 'training_data.pkl')
                self.logger.info("💾 ML model saved successfully")
        except Exception as e:
            self.logger.error(f"❌ Error saving ML model: {e}")

    def load_ml_model(self):
        """Load ML model and scaler from disk"""
        try:
            if os.path.exists('ml_trading_model.pkl'):
                self.ml_model = joblib.load('ml_trading_model.pkl')
                self.scaler = joblib.load('ml_scaler.pkl')
                if os.path.exists('training_data.pkl'):
                    self.training_data = joblib.load('training_data.pkl')
                self.is_ml_trained = True
                self.logger.info("📂 ML model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Error loading ML model: {e}")

    def run_ml_training_cycle(self):
        """Run periodic ML model training"""
        while self.is_running:
            try:
                # Collect training data from all symbols
                for symbol in self.priority_symbols:
                    self.collect_training_data(symbol)
                    time.sleep(2)  # Small delay between symbols
                
                # Train model if enough data
                if len(self.training_data) >= 100:
                    self.train_ml_model()
                
                # Wait 6 hours before next training cycle
                for _ in range(6 * 3600):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in ML training cycle: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def run_ml_enhanced_strategy(self):
        """Main trading loop with ML enhancement"""
        self.logger.info("🚀 STARTING ML-ENHANCED TRADING STRATEGY...")
        self.logger.info("🎯 Original strategy preserved + ML confidence boosting")
        
        # Start ML training in separate thread
        ml_training_thread = threading.Thread(target=self.run_ml_training_cycle)
        ml_training_thread.daemon = True
        ml_training_thread.start()
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                self.logger.info(f"\n🔄 Iteration #{iteration} | {current_time}")
                if self.is_ml_trained:
                    self.logger.info("🧠 ML Model: ACTIVE")
                
                # 1. Update P&L
                self.update_positions_pnl()
                
                # 2. Check exit conditions
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Check entry signals WITH ML ENHANCEMENT
                active_symbols = [p['symbol'] for p in self.positions.values() if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.priority_symbols:
                        if symbol not in active_symbols:
                            # USE ML-ENHANCED SIGNAL INSTEAD OF ORIGINAL
                            signal, confidence = self.generate_ml_enhanced_signal(symbol)
                            
                            if signal == "LONG" and confidence >= 0.65:
                                self.logger.info(f"🎯 ML-ENHANCED SIGNAL: {symbol} - Confidence: {confidence:.1%}")
                                position_id = self.open_optimized_position(symbol, "LONG")
                                if position_id:
                                    time.sleep(2)
                
                # 4. Log status
                performance = self.get_performance_metrics()
                ml_status = "ACTIVE" if self.is_ml_trained else "TRAINING"
                self.logger.info(f"📊 Status: {active_count} active positions | ML: {ml_status} | Account: ${self.dashboard_data['account_value']:.2f}")
                
                # 5. Wait 60 seconds
                for i in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in ML-enhanced trading loop: {e}")
                time.sleep(30)

# Global ML-enhanced bot instance
ml_trading_bot = MLEnhancedTradingBot(initial_capital=10000, leverage=10)
