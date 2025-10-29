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
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_ml.log', encoding='utf-8')
    ]
)

class MLTradingBot:
    def __init__(self, initial_capital=50000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # ML Model Components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.feature_columns = []
        
        # Trading Parameters (5% max position size)
        self.max_simultaneous_positions = 6
        self.position_size_percentage = 0.05
        
        # Position sizes from your history
        self.position_sizes = {
            'BTCUSDT': 0.12,
            'ETHUSDT': 3.2968,
            'BNBUSDT': 7.036,
            'XRPUSDT': 1737.0,
            'DOGEUSDT': 27858.0,
            'SOLUSDT': 20.76,
            'ADAUSDT': 5000.0,
            'DOTUSDT': 200.0
        }
        
        self.priority_symbols = ['ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'BTCUSDT', 'DOGEUSDT', 'ADAUSDT', 'DOTUSDT']
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'biggest_win': 0,
            'biggest_loss': 0
        }
        
        # Dashboard
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': 10,
            'average_confidence': 0,
            'ml_accuracy': 0,
            'last_update': datetime.now()
        }
        
        # Initialize ML model
        self.initialize_ml_model()
        
        self.logger.info("🧠 ML TRADING BOT INITIALIZED")
        self.logger.info(f"💰 Initial capital: ${initial_capital}")
        self.logger.info("⚡ 3min Candle Stop Loss Analysis: ACTIVE")

    def initialize_ml_model(self):
        """Initialize ML model with Random Forest"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            self.logger.info("✅ ML Model initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing ML model: {e}")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for ML features"""
        try:
            # Price-based indicators
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            macd = df['ema_12'] - df['ema_26']
            df['macd'] = macd
            df['macd_signal'] = macd.ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['price_change_1h'] = df['close'].pct_change(20)  # 1 hour in 3min intervals
            df['price_change_6h'] = df['close'].pct_change(120)  # 6 hours
            
            # Volatility
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # Support/Resistance
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            df['distance_to_support'] = (df['close'] - df['support']) / df['close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating technical indicators: {e}")
            return df

    def prepare_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        try:
            feature_columns = [
                'rsi', 'macd', 'macd_histogram', 'bb_position', 
                'volume_ratio', 'price_change_1h', 'price_change_6h',
                'volatility', 'distance_to_resistance', 'distance_to_support'
            ]
            
            # Use only the last row for prediction
            current_features = df[feature_columns].iloc[-1:].fillna(0)
            
            return current_features.values
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing ML features: {e}")
            return np.zeros((1, 10))

    def generate_ml_signal(self, symbol: str) -> Tuple[str, float]:
        """Generate trading signal using ML model"""
        try:
            df = self.get_binance_klines(symbol, '3m', 100)
            if df is None or len(df) < 50:
                return "HOLD", 0.5
            
            df = self.calculate_technical_indicators(df)
            
            if not self.is_trained:
                # Fallback to traditional strategy if ML not trained
                return self.fallback_signal(df)
            
            features = self.prepare_ml_features(df)
            
            if features.shape[1] != 10:
                return self.fallback_signal(df)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]
            
            confidence = max(prediction_proba)
            
            if prediction == 1 and confidence > 0.65:
                return "LONG", confidence
            else:
                return "HOLD", confidence
                
        except Exception as e:
            self.logger.error(f"❌ Error in ML signal generation: {e}")
            return "HOLD", 0.5

    def fallback_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Fallback traditional strategy when ML is not available"""
        try:
            current_rsi = df['rsi'].iloc[-1]
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1] if not pd.isna(df['volume_ratio'].iloc[-1]) else 1
            macd_histogram = df['macd_histogram'].iloc[-1]

            confidence = 0.0
            signal = "HOLD"
            
            conditions = 0
            if 30 <= current_rsi <= 65:
                conditions += 1
                confidence += 0.2
            if volume_ratio > 1.2:
                conditions += 1
                confidence += 0.2
            if current_price > sma_20:
                conditions += 1
                confidence += 0.2
            if macd_histogram > 0:
                conditions += 1
                confidence += 0.2
            
            if conditions >= 3:
                signal = "LONG"
                confidence = min(confidence, 0.85)
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error in fallback signal: {e}")
            return "HOLD", 0.5

    def get_binance_klines(self, symbol: str, interval: str = '3m', limit: int = 100):
        """Fetch data from Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str):
        """Get current price"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            self.logger.error(f"❌ Error fetching price for {symbol}: {e}")
            return None

    def should_close_based_on_3min_candle(self, symbol: str, position: dict) -> bool:
        """Sprawdza czy zamknięcie 3-minutowej świecy wymaga zamknięcia pozycji"""
        try:
            # Pobierz 5 ostatnich 3-minutowych świec
            df_3min = self.get_binance_klines(symbol, '3m', 5)
            if df_3min is None or len(df_3min) < 3:
                return False
            
            # Analiza ostatniej zamkniętej świecy
            last_candle = df_3min.iloc[-2]  # -2 bo -1 może być jeszcze otwarta
            current_price = df_3min['close'].iloc[-1]
            
            stop_loss_price = position['exit_plan']['stop_loss']
            entry_price = position['entry_price']
            
            # WARUNK 1: Zamknięcie poniżej Stop Loss
            if last_candle['close'] <= stop_loss_price:
                self.logger.info(f"🔴 3min Candle CLOSE below SL: {last_candle['close']:.4f} <= {stop_loss_price:.4f}")
                return True
            
            # WARUNK 2: Silny bearish pattern
            candle_size = abs(last_candle['close'] - last_candle['open'])
            avg_candle_size = abs(df_3min['close'] - df_3min['open']).tail(10).mean()
            
            # Sprawdź czy świeca jest bearish (czerwona)
            is_bearish = last_candle['close'] < last_candle['open']
            
            # Sprawdź czy świeca jest duża (więcej niż 1.5x średniej)
            is_large_candle = candle_size > (avg_candle_size * 1.5) if avg_candle_size > 0 else False
            
            # Sprawdź czy cena jest poniżej entry
            below_entry = last_candle['close'] < entry_price
            
            # WARUNK 3: Duża bearish świeca poniżej entry price
            if is_bearish and is_large_candle and below_entry:
                self.logger.info(f"🔴 Large Bearish 3min Candle: size={candle_size:.4f}, close={last_candle['close']:.4f}")
                return True
            
            # WARUNK 4: Konsekwentny spadek (2-3 bearish świece z rzędu)
            recent_candles = df_3min.tail(3)
            bearish_count = sum(1 for _, candle in recent_candles.iterrows() 
                              if candle['close'] < candle['open'])
            
            if bearish_count >= 2 and last_candle['close'] < entry_price:
                self.logger.info(f"🔴 {bearish_count} consecutive Bearish 3min Candles")
                return True
            
            # WARUNK 5: Spadek poniżej kluczowego support (SMA20 na 3min)
            sma_20_3min = df_3min['close'].rolling(20).mean().iloc[-2]
            if last_candle['close'] < sma_20_3min and current_price < sma_20_3min:
                self.logger.info(f"🔴 Price below 3min SMA20: {last_candle['close']:.4f} < {sma_20_3min:.4f}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in 3min candle analysis for {symbol}: {e}")
            return False

    def calculate_position_size(self, symbol: str, price: float, confidence: float):
        """Calculate position size (max 5% of capital)"""
        base_quantity = self.position_sizes.get(symbol, 1000.0)
        
        confidence_multiplier = 0.3 + (confidence * 0.7)
        position_value = (self.virtual_capital * self.position_size_percentage) * confidence_multiplier
        adjusted_quantity = position_value / price
        
        final_quantity = min(base_quantity, adjusted_quantity)
        position_value = final_quantity * price
        margin_required = position_value / self.leverage
        
        return final_quantity, position_value, margin_required

    def calculate_exit_levels(self, entry_price: float):
        """Calculate exit levels based on your strategy"""
        take_profit = entry_price * 1.15
        stop_loss = entry_price * 0.95
        invalidation = entry_price * 0.9375
        
        return take_profit, stop_loss, invalidation

    def open_position(self, symbol: str, side: str):
        """Open a new position with enhanced logging"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        signal, confidence = self.generate_ml_signal(symbol)
        if signal != "LONG" or confidence < 0.65:
            return None
        
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
        
        take_profit, stop_loss, invalidation = self.calculate_exit_levels(current_price)
        liquidation_price = current_price * (1 - 0.9 / self.leverage)
        
        position_id = f"ml_pos_{self.position_id}"
        self.position_id += 1
        
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': current_price,
            'quantity': quantity,
            'leverage': self.leverage,
            'margin': margin_required,
            'liquidation_price': liquidation_price,
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'unrealized_pnl': 0,
            'ml_confidence': confidence,
            'exit_plan': {
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'invalidation': invalidation
            }
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        # Enhanced logging with 3min SL info
        self.logger.info(f"🧠 ML OPEN: {side} {quantity:.4f} {symbol} @ ${current_price:.2f}")
        self.logger.info(f"   📊 TP: ${take_profit:.2f} | SL: ${stop_loss:.2f} | Margin: ${margin_required:.2f}")
        self.logger.info(f"   🤖 ML Confidence: {confidence:.1%} | Leverage: {self.leverage}X")
        self.logger.info(f"   ⚡ Stop Loss: 3min candle analysis ACTIVE")
        
        return position_id

    def update_positions_pnl(self):
        """Update P&L for all positions"""
        total_unrealized = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            position['current_price'] = current_price
            
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            
            position['unrealized_pnl'] = unrealized_pnl
            total_unrealized += unrealized_pnl
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Check exit conditions with 3-minute candle analysis"""
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            exit_reason = None
            
            # Take Profit (15%)
            if current_price >= position['exit_plan']['take_profit']:
                exit_reason = "TAKE_PROFIT"
            
            # Stop Loss with 3-minute candle analysis
            elif self.should_close_based_on_3min_candle(position['symbol'], position):
                exit_reason = "STOP_LOSS_3MIN"
            
            # Classic Stop Loss (5%) - backup
            elif current_price <= position['exit_plan']['stop_loss']:
                exit_reason = "STOP_LOSS_CLASSIC"
            
            # Invalidation (6.25%)
            elif current_price <= position['exit_plan']['invalidation']:
                exit_reason = "INVALIDATION"
            
            # Liquidation
            elif current_price <= position['liquidation_price']:
                exit_reason = "LIQUIDATION"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Close a position"""
        position = self.positions[position_id]
        
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        realized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
        self.virtual_balance += position['margin'] + realized_pnl_after_fee
        self.virtual_capital += realized_pnl_after_fee
        
        trade_record = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'realized_pnl': realized_pnl_after_fee,
            'exit_reason': exit_reason,
            'ml_confidence': position.get('ml_confidence', 0),
            'entry_time': position['entry_time'],
            'exit_time': datetime.now()
        }
        
        self.trade_history.append(trade_record)
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        position['status'] = 'CLOSED'
        
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        self.logger.info(f"{pnl_color} ML CLOSE: {position['symbol']} - P&L: ${realized_pnl_after_fee:+.2f} - Reason: {exit_reason}")

    def get_dashboard_data(self):
        """Prepare dashboard data"""
        active_positions = []
        
        for position in self.positions.values():
            if position['status'] == 'ACTIVE':
                active_positions.append({
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'entry_price': position['entry_price'],
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'liquidation_price': position['liquidation_price'],
                    'margin': position['margin'],
                    'unrealized_pnl': position['unrealized_pnl'],
                    'symbol': position['symbol'],
                    'confidence': position.get('ml_confidence', 0)
                })
        
        recent_trades = []
        for trade in self.trade_history[-10:]:
            recent_trades.append({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'quantity': trade['quantity'],
                'realized_pnl': trade['realized_pnl'],
                'exit_reason': trade['exit_reason'],
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'confidence': trade.get('ml_confidence', 0)
            })
        
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'total_fees': round(self.stats['total_fees'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2)
            },
            'performance_metrics': {
                'avg_leverage': self.dashboard_data['average_leverage'],
                'avg_confidence': self.dashboard_data['average_confidence'],
                'ml_accuracy': self.dashboard_data['ml_accuracy'],
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades
            },
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_ml_strategy(self):
        """Main ML trading loop"""
        self.logger.info("🚀 STARTING ML TRADING STRATEGY...")
        self.logger.info("⚡ 3min Candle Stop Loss Analysis: ACTIVE")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                self.logger.info(f"\n🔄 ML Iteration #{iteration} | {current_time}")
                
                self.update_positions_pnl()
                
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                active_symbols = [p['symbol'] for p in self.positions.values() if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.priority_symbols:
                        if symbol not in active_symbols:
                            signal, confidence = self.generate_ml_signal(symbol)
                            
                            if signal == "LONG" and confidence >= 0.65:
                                self.logger.info(f"🧠 ML SIGNAL: {symbol} - Confidence: {confidence:.1%}")
                                position_id = self.open_position(symbol, "LONG")
                                if position_id:
                                    time.sleep(2)
                
                total_trades = self.stats['total_trades']
                win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
                self.logger.info(f"📊 ML Status: {active_count} positions | Win Rate: {win_rate:.1f}%")
                
                for i in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in ML trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Start ML trading"""
        self.is_running = True
        self.run_ml_strategy()

    def stop_trading(self):
        """Stop ML trading"""
        self.is_running = False
        self.logger.info("🛑 ML Trading stopped")

# Global ML bot instance
ml_trading_bot = MLTradingBot(initial_capital=50000, leverage=10)
