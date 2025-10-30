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
        logging.FileHandler('enhanced_trading_bot.log', encoding='utf-8')
    ]
)

class EnhancedMLTradingBot:
    def __init__(self, initial_capital=10000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Enhanced ML Model Components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.feature_columns = []
        
        # ENHANCED STRATEGY - Multi-Timeframe Momentum + Mean Reversion
        self.max_simultaneous_positions = 4  # Reduced for better focus
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Dynamic asset allocation based on volatility and momentum
        self.asset_config = {
            'BTCUSDT': {'base_allocation': 0.25, 'volatility_adjust': 1.0},
            'ETHUSDT': {'base_allocation': 0.22, 'volatility_adjust': 1.1},
            'SOLUSDT': {'base_allocation': 0.18, 'volatility_adjust': 1.3},
            'BNBUSDT': {'base_allocation': 0.15, 'volatility_adjust': 1.2},
            'AVAXUSDT': {'base_allocation': 0.10, 'volatility_adjust': 1.5},
            'MATICUSDT': {'base_allocation': 0.10, 'volatility_adjust': 1.4},
        }
        
        self.priority_symbols = list(self.asset_config.keys())
        
        # Enhanced trading parameters
        self.momentum_threshold = 0.015  # 1.5% momentum
        self.volume_spike_threshold = 2.0  # 200% volume spike
        self.rsi_oversold = 35
        self.rsi_overbought = 65
        self.atr_multiplier = 1.5  # For stop loss
        
        # Advanced strategy parameters
        self.use_multi_timeframe = True
        self.use_mean_reversion = True
        self.use_momentum_confirmation = True
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'biggest_win': 0,
            'biggest_loss': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'average_hold_time': 0
        }
        
        # Enhanced Dashboard
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': leverage,
            'portfolio_diversity': 0,
            'market_regime': 'NEUTRAL',
            'risk_level': 'MEDIUM',
            'strategy_performance': {},
            'last_update': datetime.now()
        }
        
        # Initialize enhanced ML model
        self.initialize_enhanced_ml_model()
        
        self.logger.info("🚀 ENHANCED ML TRADING BOT INITIALIZED")
        self.logger.info("💰 Multi-Strategy: Momentum + Mean Reversion + Breakout")
        self.logger.info("📊 Advanced Risk Management & Position Sizing")

    def initialize_enhanced_ml_model(self):
        """Initialize enhanced ML model with ensemble approach"""
        try:
            # Ensemble of models for better predictions
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=10,
                random_state=42
            )
            self.logger.info("✅ Enhanced ML Model initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing enhanced ML model: {e}")

    def get_multi_timeframe_data(self, symbol: str) -> Dict:
        """Get data across multiple timeframes for better analysis"""
        timeframes = {
            '1m': 100,
            '5m': 100,
            '15m': 100,
            '1h': 200
        }
        
        mtf_data = {}
        for tf, limit in timeframes.items():
            try:
                # Convert to minutes for simulation
                minutes = int(tf.replace('m', '').replace('h', '00'))
                df = self.get_binance_klines(symbol, '3m', limit)
                if df is not None and len(df) > 50:
                    # Resample for different timeframes (simplified)
                    if tf == '5m':
                        df = df.iloc[::2].reset_index(drop=True)
                    elif tf == '15m':
                        df = df.iloc[::5].reset_index(drop=True)
                    elif tf == '1h':
                        df = df.iloc[::20].reset_index(drop=True)
                    
                    mtf_data[tf] = self.calculate_advanced_indicators(df)
            except Exception as e:
                self.logger.warning(f"Could not get {tf} data for {symbol}: {e}")
        
        return mtf_data

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        try:
            # Price-based indicators
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
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR for volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum indicators
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['rate_of_change'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
            
            # Support/Resistance
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            df['distance_to_support'] = (df['close'] - df['support']) / df['close']
            
            # Volatility
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # Price action features
            df['body_size'] = abs(df['close'] - df['open']) / df['open']
            df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
            df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating advanced indicators: {e}")
            return df

    def analyze_market_regime(self, symbol: str) -> str:
        """Analyze current market regime for the symbol"""
        try:
            df = self.get_binance_klines(symbol, '3m', 100)
            if df is None or len(df) < 50:
                return "NEUTRAL"
            
            df = self.calculate_advanced_indicators(df)
            
            # Calculate regime indicators
            volatility = df['volatility'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            bb_position = df['bb_position'].iloc[-1]
            momentum = df['momentum_10'].iloc[-1]
            
            # Determine regime
            if volatility > 0.02 and abs(momentum) > 0.03:
                return "TRENDING"
            elif volatility < 0.01 and abs(momentum) < 0.01:
                return "RANGING"
            elif rsi < 30 or rsi > 70:
                return "EXTREME"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"Error analyzing market regime for {symbol}: {e}")
            return "NEUTRAL"

    def generate_enhanced_signal(self, symbol: str) -> Tuple[str, float, Dict]:
        """Generate enhanced trading signal using multiple strategies"""
        try:
            # Get multi-timeframe data
            mtf_data = self.get_multi_timeframe_data(symbol)
            if not mtf_data:
                return "HOLD", 0.5, {}
            
            current_data = mtf_data.get('5m', mtf_data.get('1m'))
            if current_data is None or len(current_data) < 20:
                return "HOLD", 0.5, {}
            
            current_price = current_data['close'].iloc[-1]
            current_rsi = current_data['rsi'].iloc[-1]
            current_volume_ratio = current_data['volume_ratio'].iloc[-1]
            bb_position = current_data['bb_position'].iloc[-1]
            macd_histogram = current_data['macd_histogram'].iloc[-1]
            atr_pct = current_data['atr_pct'].iloc[-1]
            
            # Strategy 1: Momentum Breakout
            momentum_signal, momentum_confidence = self._momentum_strategy(current_data)
            
            # Strategy 2: Mean Reversion
            mean_reversion_signal, mean_reversion_confidence = self._mean_reversion_strategy(current_data)
            
            # Strategy 3: Volume-based Breakout
            volume_signal, volume_confidence = self._volume_strategy(current_data)
            
            # Combine signals with weights
            signals = []
            confidences = []
            
            if momentum_signal != "HOLD":
                signals.append(momentum_signal)
                confidences.append(momentum_confidence * 0.4)  # 40% weight
            
            if mean_reversion_signal != "HOLD":
                signals.append(mean_reversion_signal)
                confidences.append(mean_reversion_confidence * 0.35)  # 35% weight
            
            if volume_signal != "HOLD":
                signals.append(volume_signal)
                confidences.append(volume_confidence * 0.25)  # 25% weight
            
            if not signals:
                return "HOLD", 0.5, {}
            
            # Weighted consensus
            long_votes = sum(1 for s in signals if s == "LONG")
            short_votes = sum(1 for s in signals if s == "SHORT")
            total_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if long_votes >= 2 and total_confidence > 0.6:
                final_signal = "LONG"
                final_confidence = min(total_confidence * 1.2, 0.95)
            elif short_votes >= 2 and total_confidence > 0.6:
                final_signal = "SHORT"
                final_confidence = min(total_confidence * 1.2, 0.95)
            else:
                final_signal = "HOLD"
                final_confidence = 0.5
            
            strategy_info = {
                'momentum_signal': momentum_signal,
                'mean_reversion_signal': mean_reversion_signal,
                'volume_signal': volume_signal,
                'market_regime': self.analyze_market_regime(symbol),
                'volatility': atr_pct
            }
            
            if final_signal != "HOLD":
                self.logger.info(f"🎯 ENHANCED SIGNAL: {symbol} - {final_signal} (Conf: {final_confidence:.1%})")
                self.logger.info(f"   📊 Strategies: Momentum:{momentum_signal} MeanRev:{mean_reversion_signal} Volume:{volume_signal}")
            
            return final_signal, final_confidence, strategy_info
            
        except Exception as e:
            self.logger.error(f"❌ Error generating enhanced signal for {symbol}: {e}")
            return "HOLD", 0.5, {}

    def _momentum_strategy(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Momentum-based trading strategy"""
        try:
            current_rsi = df['rsi'].iloc[-1]
            macd_histogram = df['macd_histogram'].iloc[-1]
            momentum_5 = df['momentum_5'].iloc[-1]
            price_above_sma20 = df['close'].iloc[-1] > df['sma_20'].iloc[-1]
            price_above_sma50 = df['close'].iloc[-1] > df['sma_50'].iloc[-1]
            
            conditions = 0
            confidence = 0.0
            
            # Bullish momentum conditions
            if macd_histogram > 0:
                conditions += 1
                confidence += 0.2
            
            if momentum_5 > 0.01:  # 1% momentum
                conditions += 1
                confidence += 0.2
            
            if price_above_sma20 and price_above_sma50:
                conditions += 1
                confidence += 0.2
            
            if 40 <= current_rsi <= 70:  # Optimal RSI for momentum
                conditions += 1
                confidence += 0.2
            
            if conditions >= 3:
                return "LONG", min(confidence, 0.8)
            
            # Bearish momentum conditions
            conditions_bearish = 0
            confidence_bearish = 0.0
            
            if macd_histogram < 0:
                conditions_bearish += 1
                confidence_bearish += 0.2
            
            if momentum_5 < -0.01:
                conditions_bearish += 1
                confidence_bearish += 0.2
            
            if not price_above_sma20 and not price_above_sma50:
                conditions_bearish += 1
                confidence_bearish += 0.2
            
            if 30 <= current_rsi <= 60:  # RSI for bearish momentum
                conditions_bearish += 1
                confidence_bearish += 0.2
            
            if conditions_bearish >= 3:
                return "SHORT", min(confidence_bearish, 0.8)
            
            return "HOLD", 0.5
            
        except Exception as e:
            self.logger.error(f"Error in momentum strategy: {e}")
            return "HOLD", 0.5

    def _mean_reversion_strategy(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Mean reversion trading strategy"""
        try:
            current_rsi = df['rsi'].iloc[-1]
            bb_position = df['bb_position'].iloc[-1]
            distance_to_support = df['distance_to_support'].iloc[-1]
            distance_to_resistance = df['distance_to_resistance'].iloc[-1]
            
            conditions = 0
            confidence = 0.0
            
            # Oversold bounce (LONG)
            if current_rsi < self.rsi_oversold:
                conditions += 1
                confidence += 0.3
            
            if bb_position < 0.2:  # Near lower Bollinger Band
                conditions += 1
                confidence += 0.3
            
            if distance_to_support < 0.02:  # Close to support
                conditions += 1
                confidence += 0.2
            
            if conditions >= 2:
                return "LONG", min(confidence, 0.8)
            
            # Overbought rejection (SHORT)
            conditions_bearish = 0
            confidence_bearish = 0.0
            
            if current_rsi > self.rsi_overbought:
                conditions_bearish += 1
                confidence_bearish += 0.3
            
            if bb_position > 0.8:  # Near upper Bollinger Band
                conditions_bearish += 1
                confidence_bearish += 0.3
            
            if distance_to_resistance < 0.02:  # Close to resistance
                conditions_bearish += 1
                confidence_bearish += 0.2
            
            if conditions_bearish >= 2:
                return "SHORT", min(confidence_bearish, 0.8)
            
            return "HOLD", 0.5
            
        except Exception as e:
            self.logger.error(f"Error in mean reversion strategy: {e}")
            return "HOLD", 0.5

    def _volume_strategy(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Volume-based breakout strategy"""
        try:
            volume_ratio = df['volume_ratio'].iloc[-1]
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            body_size = df['body_size'].iloc[-1]
            
            # Volume spike with price movement
            if volume_ratio > self.volume_spike_threshold and abs(price_change) > 0.005:
                if price_change > 0 and body_size > 0.01:  # Bullish volume breakout
                    return "LONG", 0.7
                elif price_change < 0 and body_size > 0.01:  # Bearish volume breakout
                    return "SHORT", 0.7
            
            return "HOLD", 0.5
            
        except Exception as e:
            self.logger.error(f"Error in volume strategy: {e}")
            return "HOLD", 0.5

    def calculate_enhanced_position_size(self, symbol: str, price: float, confidence: float, 
                                       strategy_info: Dict, signal: str) -> Tuple[float, float, float]:
        """Calculate position size with advanced risk management"""
        try:
            # Base allocation from config
            base_config = self.asset_config.get(symbol, {'base_allocation': 0.15, 'volatility_adjust': 1.0})
            base_allocation = base_config['base_allocation']
            volatility_adjust = base_config['volatility_adjust']
            
            # Market regime adjustment
            regime = strategy_info.get('market_regime', 'NEUTRAL')
            regime_multiplier = {
                'TRENDING': 1.2,
                'RANGING': 0.8,
                'EXTREME': 0.5,
                'NEUTRAL': 1.0
            }.get(regime, 1.0)
            
            # Confidence adjustment
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5-1.0
            
            # Volatility adjustment (inverse relationship)
            volatility = strategy_info.get('volatility', 0.02)
            volatility_multiplier = max(0.5, min(1.5, 0.02 / volatility))  # Lower vol = larger position
            
            # Calculate final allocation
            final_allocation = (base_allocation * regime_multiplier * 
                              confidence_multiplier * volatility_multiplier * volatility_adjust)
            
            # Risk-adjusted position value
            position_value = self.virtual_capital * final_allocation
            
            # Maximum position limits
            max_position_value = self.virtual_capital * 0.25  # 25% max per position
            position_value = min(position_value, max_position_value)
            
            # Calculate quantity
            quantity = position_value / price
            
            # Calculate margin required
            margin_required = position_value / self.leverage
            
            return quantity, position_value, margin_required
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating enhanced position size: {e}")
            return 0, 0, 0

    def calculate_enhanced_exit_levels(self, symbol: str, entry_price: float, signal: str, 
                                     strategy_info: Dict) -> Dict:
        """Calculate dynamic exit levels based on market conditions"""
        try:
            df = self.get_binance_klines(symbol, '3m', 50)
            if df is None:
                return self._get_default_exit_levels(entry_price, signal)
            
            df = self.calculate_advanced_indicators(df)
            atr = df['atr'].iloc[-1]
            volatility = df['volatility'].iloc[-1]
            regime = strategy_info.get('market_regime', 'NEUTRAL')
            
            # Base parameters by regime
            regime_params = {
                'TRENDING': {'tp_multiplier': 2.5, 'sl_multiplier': 1.0, 'rr_ratio': 2.5},
                'RANGING': {'tp_multiplier': 1.5, 'sl_multiplier': 1.2, 'rr_ratio': 1.25},
                'EXTREME': {'tp_multiplier': 1.0, 'sl_multiplier': 1.5, 'rr_ratio': 0.67},
                'NEUTRAL': {'tp_multiplier': 2.0, 'sl_multiplier': 1.0, 'rr_ratio': 2.0}
            }
            
            params = regime_params.get(regime, regime_params['NEUTRAL'])
            
            if signal == "LONG":
                take_profit = entry_price + (atr * params['tp_multiplier'])
                stop_loss = entry_price - (atr * params['sl_multiplier'])
            else:  # SHORT
                take_profit = entry_price - (atr * params['tp_multiplier'])
                stop_loss = entry_price + (atr * params['sl_multiplier'])
            
            # Ensure proper risk-reward ratio
            if signal == "LONG":
                actual_rr = (take_profit - entry_price) / (entry_price - stop_loss)
                if actual_rr < params['rr_ratio']:
                    # Adjust take profit to meet minimum RR
                    take_profit = entry_price + (params['rr_ratio'] * (entry_price - stop_loss))
            else:
                actual_rr = (entry_price - take_profit) / (stop_loss - entry_price)
                if actual_rr < params['rr_ratio']:
                    take_profit = entry_price - (params['rr_ratio'] * (stop_loss - entry_price))
            
            return {
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'atr': atr,
                'risk_reward_ratio': params['rr_ratio'],
                'regime': regime
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced exit levels: {e}")
            return self._get_default_exit_levels(entry_price, signal)

    def _get_default_exit_levels(self, entry_price: float, signal: str) -> Dict:
        """Get default exit levels as fallback"""
        if signal == "LONG":
            return {
                'take_profit': entry_price * 1.08,
                'stop_loss': entry_price * 0.94,
                'atr': 0,
                'risk_reward_ratio': 2.0,
                'regime': 'NEUTRAL'
            }
        else:
            return {
                'take_profit': entry_price * 0.92,
                'stop_loss': entry_price * 1.06,
                'atr': 0,
                'risk_reward_ratio': 2.0,
                'regime': 'NEUTRAL'
            }

    def open_enhanced_position(self, symbol: str):
        """Open position using enhanced strategy"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        signal, confidence, strategy_info = self.generate_enhanced_signal(symbol)
        if signal == "HOLD" or confidence < 0.65:
            return None
        
        # Check position limits
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        # Calculate position size
        quantity, position_value, margin_required = self.calculate_enhanced_position_size(
            symbol, current_price, confidence, strategy_info, signal
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
        
        # Calculate exit levels
        exit_levels = self.calculate_enhanced_exit_levels(symbol, current_price, signal, strategy_info)
        
        # Calculate liquidation price
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        position_id = f"enhanced_{self.position_id}"
        self.position_id += 1
        
        position = {
            'symbol': symbol,
            'side': signal,
            'entry_price': current_price,
            'quantity': quantity,
            'leverage': self.leverage,
            'margin': margin_required,
            'liquidation_price': liquidation_price,
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'unrealized_pnl': 0,
            'confidence': confidence,
            'strategy': 'ENHANCED_MULTI',
            'strategy_info': strategy_info,
            'exit_plan': exit_levels
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        self.logger.info(f"🎯 ENHANCED OPEN: {signal} {quantity:.4f} {symbol} @ ${current_price:.2f}")
        self.logger.info(f"   📊 TP: ${exit_levels['take_profit']:.2f} | SL: ${exit_levels['stop_loss']:.2f}")
        self.logger.info(f"   💰 Position: ${position_value:.2f} | Margin: ${margin_required:.2f}")
        self.logger.info(f"   🤖 Confidence: {confidence:.1%} | Regime: {strategy_info.get('market_regime', 'UNKNOWN')}")
        self.logger.info(f"   ⚡ RR Ratio: {exit_levels['risk_reward_ratio']:.1f} | Leverage: {self.leverage}X")
        
        return position_id

    def enhanced_stop_loss_check(self, position: dict) -> bool:
        """Enhanced stop loss with volatility adjustment"""
        try:
            symbol = position['symbol']
            current_price = self.get_current_price(symbol)
            if not current_price:
                return False
            
            stop_loss_price = position['exit_plan']['stop_loss']
            
            # Basic stop loss check
            if (position['side'] == 'LONG' and current_price <= stop_loss_price) or \
               (position['side'] == 'SHORT' and current_price >= stop_loss_price):
                return True
            
            # Additional: Trailing stop logic for profitable positions
            entry_price = position['entry_price']
            current_pnl_pct = 0
            
            if position['side'] == 'LONG':
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price
            
            # If position is significantly profitable, use tighter trailing stop
            if current_pnl_pct > 0.05:  # 5% profit
                # Dynamic trailing stop at 70% of profits
                trailing_stop_pct = current_pnl_pct * 0.3  # Keep 30% of profits
                
                if position['side'] == 'LONG':
                    trailing_stop_price = entry_price * (1 + trailing_stop_pct)
                    if current_price < trailing_stop_price:
                        self.logger.info(f"🔴 TRAILING STOP: {symbol} - Locked in {current_pnl_pct:.1%} profit")
                        return True
                else:
                    trailing_stop_price = entry_price * (1 - trailing_stop_pct)
                    if current_price > trailing_stop_price:
                        self.logger.info(f"🔴 TRAILING STOP: {symbol} - Locked in {current_pnl_pct:.1%} profit")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in enhanced stop loss check: {e}")
            return False

    def run_enhanced_strategy(self):
        """Main enhanced trading strategy loop"""
        self.logger.info("🚀 STARTING ENHANCED TRADING STRATEGY...")
        self.logger.info("🎯 Multi-Strategy: Momentum + Mean Reversion + Volume Breakout")
        self.logger.info("⚡ Advanced Risk Management & Dynamic Position Sizing")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                self.logger.info(f"\n🔄 Enhanced Iteration #{iteration} | {current_time}")
                
                # 1. Update P&L
                self.update_positions_pnl()
                
                # 2. Check exit conditions with enhanced stop loss
                positions_to_close = []
                for position_id, position in self.positions.items():
                    if position['status'] != 'ACTIVE':
                        continue
                    
                    current_price = self.get_current_price(position['symbol'])
                    if not current_price:
                        continue
                    
                    exit_reason = None
                    
                    # Take Profit
                    if (position['side'] == 'LONG' and current_price >= position['exit_plan']['take_profit']) or \
                       (position['side'] == 'SHORT' and current_price <= position['exit_plan']['take_profit']):
                        exit_reason = "TAKE_PROFIT"
                    
                    # Enhanced Stop Loss
                    elif self.enhanced_stop_loss_check(position):
                        exit_reason = "ENHANCED_STOP_LOSS"
                    
                    # Liquidation
                    elif (position['side'] == 'LONG' and current_price <= position['liquidation_price']) or \
                         (position['side'] == 'SHORT' and current_price >= position['liquidation_price']):
                        exit_reason = "LIQUIDATION"
                    
                    if exit_reason:
                        positions_to_close.append((position_id, exit_reason, current_price))
                
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Look for new entry opportunities
                active_symbols = [p['symbol'] for p in self.positions.values() if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    # Analyze all symbols and prioritize by confidence
                    symbol_opportunities = []
                    
                    for symbol in self.priority_symbols:
                        if symbol not in active_symbols:
                            signal, confidence, strategy_info = self.generate_enhanced_signal(symbol)
                            if signal != "HOLD" and confidence >= 0.65:
                                symbol_opportunities.append({
                                    'symbol': symbol,
                                    'signal': signal,
                                    'confidence': confidence,
                                    'strategy_info': strategy_info
                                })
                    
                    # Sort by confidence and open positions
                    symbol_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    for opportunity in symbol_opportunities[:2]:  # Max 2 new positions per iteration
                        if active_count >= self.max_simultaneous_positions:
                            break
                            
                        position_id = self.open_enhanced_position(opportunity['symbol'])
                        if position_id:
                            active_count += 1
                            time.sleep(1)  # Small delay between positions
                
                # 4. Portfolio status update
                portfolio_value = self.dashboard_data['account_value']
                diversity = self.dashboard_data['portfolio_diversity'] * 100
                
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Positions: {active_count}/{self.max_simultaneous_positions}")
                self.logger.info(f"🌐 Diversity: {diversity:.1f}% | Strategy: Multi-Timeframe Enhanced")
                
                # 5. Wait 45 seconds for next iteration
                for i in range(45):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in enhanced trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Start enhanced trading"""
        self.is_running = True
        self.run_enhanced_strategy()

    def stop_trading(self):
        """Stop enhanced trading"""
        self.is_running = False
        self.logger.info("🛑 Enhanced Trading stopped")

    # Keep existing utility methods (get_binance_klines, get_current_price, etc.)
    def get_binance_klines(self, symbol: str, interval: str = '3m', limit: int = 100):
        """Get LIVE price data from working APIs"""
        # Implementation same as original...
        pass

    def get_current_price(self, symbol: str):
        """Get LIVE current price from working APIs"""
        # Implementation same as original...
        pass

    def update_positions_pnl(self):
        """Update P&L for all positions"""
        # Implementation similar to original but enhanced...
        pass

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Close a position"""
        # Implementation similar to original but enhanced...
        pass

    def get_dashboard_data(self):
        """Enhanced dashboard data"""
        # Implementation similar to original but with more metrics...
        pass

# Global enhanced bot instance
enhanced_ml_bot = EnhancedMLTradingBot(initial_capital=10000, leverage=10)
