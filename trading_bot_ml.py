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
    def __init__(self, initial_capital=10000, leverage=10):
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
        
        # NOWA STRATEGIA ALOKACJI - zgodnie z Twoimi danymi
        self.max_simultaneous_positions = 6
        self.asset_allocation = {
            'ETHUSDT': 0.22,  # 22% - główna pozycja
            'BTCUSDT': 0.20,  # 20% - główna pozycja  
            'SOLUSDT': 0.19,  # 19% - główna pozycja
            'BNBUSDT': 0.18,  # 18% - średnia pozycja
            'XRPUSDT': 0.17,  # 17% - średnia pozycja
            'DOGEUSDT': 0.04, # 4% - mniejsza pozycja
        }
        
        self.priority_symbols = list(self.asset_allocation.keys())
        
        # Breakout trading parameters
        self.breakout_threshold = 0.02  # 2% powyżej oporu
        self.min_volume_ratio = 1.5     # 150% średniego volume
        self.max_position_value = 0.30  # MAX 30% na jedną pozycję
        
        # Position sizes from breakout strategy
        self.position_sizes = {
            'ETHUSDT': 2.5,     # ~$2,000 przy $4,000 ETH
            'BTCUSDT': 0.08,    # ~$2,400 przy $30,000 BTC
            'SOLUSDT': 12.0,    # ~$2,400 przy $200 SOL
            'BNBUSDT': 15.0,    # ~$2,250 przy $150 BNB
            'XRPUSDT': 4500.0,  # ~$2,250 przy $0.50 XRP
            'DOGEUSDT': 25000.0, # ~$1,000 przy $0.04 DOGE
        }
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'biggest_win': 0,
            'biggest_loss': 0,
            'breakout_trades': 0,
            'portfolio_utilization': 0
        }
        
        # Dashboard
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': leverage,
            'average_confidence': 0,
            'ml_accuracy': 0,
            'portfolio_diversity': 0,
            'last_update': datetime.now()
        }
        
        # Initialize ML model
        self.initialize_ml_model()
        
        self.logger.info("🧠 ML TRADING BOT - BREAKOUT STRATEGY")
        self.logger.info(f"💰 Initial capital: ${initial_capital}")
        self.logger.info("📊 Asset Allocation: ETH(22%) BTC(20%) SOL(19%) BNB(18%) XRP(17%) DOGE(4%)")
        self.logger.info("⚡ Breakout Trading + 3min Candle Stop Loss")

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

    def detect_breakout_signal(self, symbol: str) -> Tuple[bool, float, float]:
        """Wykrywa sygnały breakout na podstawie oporu i volume"""
        try:
            df = self.get_binance_klines(symbol, '3m', 100)
            if df is None or len(df) < 50:
                return False, 0, 0
            
            # Oblicz poziomy oporu
            resistance_level = df['high'].rolling(20).max().iloc[-1]
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # Sprawdź czy cena przebiła opór
            price_above_resistance = current_price > resistance_level
            breakout_strength = (current_price - resistance_level) / resistance_level
            
            # Sprawdź volume - musi być powyżej średniej
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Warunki breakout
            is_breakout = (price_above_resistance and 
                          breakout_strength >= self.breakout_threshold and
                          volume_ratio >= self.min_volume_ratio)
            
            confidence = min(breakout_strength * 10 + volume_ratio * 0.2, 0.95)
            
            if is_breakout:
                self.logger.info(f"🎯 BREAKOUT DETECTED: {symbol} - Strength: {breakout_strength:.2%}, Volume: {volume_ratio:.1f}x")
            
            return is_breakout, confidence, resistance_level
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting breakout for {symbol}: {e}")
            return False, 0, 0

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for ML features"""
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
            
            # Support/Resistance
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            df['distance_to_support'] = (df['close'] - df['support']) / df['close']
            
            # Momentum
            df['momentum_1h'] = df['close'].pct_change(20)  # 1 hour momentum
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating technical indicators: {e}")
            return df

    def generate_breakout_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał oparty na strategii breakout"""
        try:
            # Najpierw sprawdź breakout
            is_breakout, breakout_confidence, resistance_level = self.detect_breakout_signal(symbol)
            
            if is_breakout and breakout_confidence >= 0.65:
                return "BREAKOUT_LONG", breakout_confidence
            
            # Fallback do tradycyjnej strategii ML
            df = self.get_binance_klines(symbol, '3m', 100)
            if df is None or len(df) < 50:
                return "HOLD", 0.5
            
            df = self.calculate_technical_indicators(df)
            
            current_rsi = df['rsi'].iloc[-1]
            current_price = df['close'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1] if not pd.isna(df['volume_ratio'].iloc[-1]) else 1
            macd_histogram = df['macd_histogram'].iloc[-1]
            momentum_1h = df['momentum_1h'].iloc[-1]

            confidence = 0.0
            signal = "HOLD"
            
            # Warunki dla momentum trading
            conditions = 0
            if 40 <= current_rsi <= 70:  # Optymalny zakres RSI dla breakout
                conditions += 1
                confidence += 0.15
            
            if volume_ratio > 1.3:  # Wysoki volume
                conditions += 1
                confidence += 0.20
            
            if macd_histogram > 0:  # Pozytywny momentum MACD
                conditions += 1
                confidence += 0.20
            
            if momentum_1h > 0.01:  # Pozytywny momentum 1h
                conditions += 1
                confidence += 0.15
            
            if current_price > df['sma_20'].iloc[-1]:  # Cena powyżej SMA20
                conditions += 1
                confidence += 0.15
            
            if conditions >= 3:
                signal = "LONG"
                confidence = min(confidence + (conditions - 3) * 0.1, 0.85)
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error generating breakout signal for {symbol}: {e}")
            return "HOLD", 0.5

    def calculate_breakout_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji zgodnie z alokacją assetów"""
        try:
            # Bazowa alokacja z portfolio
            allocation_percentage = self.asset_allocation.get(symbol, 0.15)
            
            # Dostosowanie na podstawie confidence
            confidence_multiplier = 0.7 + (confidence * 0.3)  # 0.7-1.0
            
            # Oblicz wartość pozycji
            position_value = (self.virtual_capital * allocation_percentage) * confidence_multiplier
            
            # Limit maksymalnej pozycji (30% depozytu)
            max_position_value = self.virtual_capital * self.max_position_value
            position_value = min(position_value, max_position_value)
            
            # Oblicz quantity
            quantity = position_value / price
            
            # Użyj historycznej wielkości jeśli mniejsza
            historical_quantity = self.position_sizes.get(symbol, quantity)
            final_quantity = min(quantity, historical_quantity)
            
            # Przelicz finalną wartość
            final_position_value = final_quantity * price
            margin_required = final_position_value / self.leverage
            
            return final_quantity, final_position_value, margin_required
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size for {symbol}: {e}")
            return 0, 0, 0

    def get_portfolio_diversity(self) -> float:
        """Oblicza dywersyfikację portfela"""
        try:
            active_positions = [p for p in self.positions.values() if p['status'] == 'ACTIVE']
            if not active_positions:
                return 0
            
            total_margin = sum(p['margin'] for p in active_positions)
            if total_margin == 0:
                return 0
            
            # Oblicz wskaźnik Herfindahla (miernik koncentracji)
            concentration_index = sum((p['margin'] / total_margin) ** 2 for p in active_positions)
            diversity = 1 - concentration_index
            
            return diversity
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating portfolio diversity: {e}")
            return 0

    def should_close_based_on_3min_candle(self, symbol: str, position: dict) -> bool:
        """Sprawdza czy zamknięcie 3-minutowej świecy wymaga zamknięcia pozycji"""
        try:
            df_3min = self.get_binance_klines(symbol, '3m', 5)
            if df_3min is None or len(df_3min) < 3:
                return False
            
            last_candle = df_3min.iloc[-2]
            current_price = df_3min['close'].iloc[-1]
            
            stop_loss_price = position['exit_plan']['stop_loss']
            entry_price = position['entry_price']
            take_profit_price = position['exit_plan']['take_profit']
            
            # WARUNK 1: Zamknięcie poniżej Stop Loss
            if last_candle['close'] <= stop_loss_price:
                self.logger.info(f"🔴 3min Candle CLOSE below SL: {last_candle['close']:.4f} <= {stop_loss_price:.4f}")
                return True
            
            # WARUNK 2: Zamknięcie powyżej Take Profit (częściowe zabezpieczenie zysku)
            if last_candle['close'] >= take_profit_price * 0.95:  # 95% TP
                self.logger.info(f"🟢 3min Candle near TP: {last_candle['close']:.4f} >= {take_profit_price * 0.95:.4f}")
                return True
            
            # WARUNK 3: Duża bearish świeca po breakout
            candle_size = abs(last_candle['close'] - last_candle['open'])
            avg_candle_size = abs(df_3min['close'] - df_3min['open']).tail(10).mean()
            
            is_bearish = last_candle['close'] < last_candle['open']
            is_large_candle = candle_size > (avg_candle_size * 2.0) if avg_candle_size > 0 else False
            
            if is_bearish and is_large_candle and last_candle['close'] < entry_price:
                self.logger.info(f"🔴 Large Bearish Reversal Candle after breakout")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in 3min candle analysis for {symbol}: {e}")
            return False

    def calculate_breakout_exit_levels(self, entry_price: float, resistance_level: float) -> Dict:
        """Oblicza poziomy wyjścia dla strategii breakout"""
        # Dla breakout: agresywniejszy TP, tighter SL
        take_profit = entry_price * 1.08   # 8% TP dla breakout
        stop_loss = resistance_level * 0.98  # SL tuż poniżej breakout level
        invalidation = entry_price * 0.96   # 4% invalidation
        
        return {
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'invalidation': invalidation
        }

    def open_breakout_position(self, symbol: str):
        """Otwiera pozycję breakout"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        signal, confidence = self.generate_breakout_signal(symbol)
        if signal not in ["BREAKOUT_LONG", "LONG"] or confidence < 0.65:
            return None
        
        # Sprawdź limit aktywnych pozycji
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        # Oblicz wielkość pozycji zgodnie z alokacją
        quantity, position_value, margin_required = self.calculate_breakout_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
        
        # Oblicz poziomy wyjścia
        is_breakout = signal == "BREAKOUT_LONG"
        if is_breakout:
            _, _, resistance_level = self.detect_breakout_signal(symbol)
            exit_levels = self.calculate_breakout_exit_levels(current_price, resistance_level)
        else:
            exit_levels = {
                'take_profit': current_price * 1.10,  # 10% TP
                'stop_loss': current_price * 0.95,    # 5% SL
                'invalidation': current_price * 0.93  # 7% invalidation
            }
        
        liquidation_price = current_price * (1 - 0.9 / self.leverage)
        
        position_id = f"breakout_{self.position_id}"
        self.position_id += 1
        
        position = {
            'symbol': symbol,
            'side': 'LONG',
            'entry_price': current_price,
            'quantity': quantity,
            'leverage': self.leverage,
            'margin': margin_required,
            'liquidation_price': liquidation_price,
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'unrealized_pnl': 0,
            'confidence': confidence,
            'strategy': 'BREAKOUT' if is_breakout else 'MOMENTUM',
            'exit_plan': exit_levels
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        if is_breakout:
            self.stats['breakout_trades'] += 1
            self.logger.info(f"🎯 BREAKOUT OPEN: {quantity:.4f} {symbol} @ ${current_price:.2f}")
        else:
            self.logger.info(f"📈 MOMENTUM OPEN: {quantity:.4f} {symbol} @ ${current_price:.2f}")
        
        self.logger.info(f"   📊 TP: ${exit_levels['take_profit']:.2f} | SL: ${exit_levels['stop_loss']:.2f}")
        self.logger.info(f"   💰 Position: ${position_value:.2f} ({self.asset_allocation[symbol]*100:.0f}% allocation)")
        self.logger.info(f"   🤖 Confidence: {confidence:.1%} | Leverage: {self.leverage}X")
        
        return position_id

    # Pozostałe metody pozostają bez zmian (update_positions_pnl, check_exit_conditions, close_position, etc.)
    # ... [reszta kodu taka sama jak poprzednio]

    def update_positions_pnl(self):
        """Update P&L for all positions"""
        total_unrealized = 0
        total_margin = 0
        
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
            total_margin += position['margin']
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        self.dashboard_data['portfolio_diversity'] = self.get_portfolio_diversity()
        
        # Oblicz wykorzystanie portfela
        if self.virtual_capital > 0:
            portfolio_utilization = (total_margin * self.leverage) / (self.virtual_capital * self.leverage)
            self.stats['portfolio_utilization'] = portfolio_utilization
        
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
            
            # Take Profit
            if current_price >= position['exit_plan']['take_profit']:
                exit_reason = "TAKE_PROFIT"
            
            # Stop Loss with 3-minute candle analysis
            elif self.should_close_based_on_3min_candle(position['symbol'], position):
                exit_reason = "STOP_LOSS_3MIN"
            
            # Classic Stop Loss
            elif current_price <= position['exit_plan']['stop_loss']:
                exit_reason = "STOP_LOSS_CLASSIC"
            
            # Invalidation
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
            'strategy': position.get('strategy', 'MOMENTUM'),
            'confidence': position.get('confidence', 0),
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
        strategy_icon = "🎯" if position.get('strategy') == 'BREAKOUT' else "📈"
        self.logger.info(f"{pnl_color} {strategy_icon} CLOSE: {position['symbol']} - P&L: ${realized_pnl_after_fee:+.2f} - Reason: {exit_reason}")

    def get_dashboard_data(self):
        """Prepare dashboard data"""
        active_positions = []
        total_confidence = 0
        confidence_count = 0
        
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
                    'confidence': position.get('confidence', 0),
                    'strategy': position.get('strategy', 'MOMENTUM')
                })
        
        # Oblicz średnią confidence
        for symbol in self.priority_symbols:
            signal, confidence = self.generate_breakout_signal(symbol)
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = round((total_confidence / confidence_count) * 100, 1)
        
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
                'strategy': trade.get('strategy', 'MOMENTUM'),
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'confidence': trade.get('confidence', 0)
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
                'portfolio_diversity': round(self.dashboard_data['portfolio_diversity'] * 100, 1),
                'portfolio_utilization': round(self.stats['portfolio_utilization'] * 100, 1),
                'breakout_trades': self.stats['breakout_trades'],
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades
            },
            'asset_allocation': self.asset_allocation,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_breakout_strategy(self):
        """Główna pętla strategii breakout"""
        self.logger.info("🚀 STARTING BREAKOUT TRADING STRATEGY...")
        self.logger.info("📊 Portfolio Allocation:")
        for symbol, allocation in self.asset_allocation.items():
            self.logger.info(f"   {symbol}: {allocation*100:.1f}%")
        self.logger.info("⚡ Breakout Detection + 3min Candle Analysis ACTIVE")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                self.logger.info(f"\n🔄 Breakout Iteration #{iteration} | {current_time}")
                
                # 1. Aktualizuj P&L
                self.update_positions_pnl()
                
                # 2. Sprawdź warunki wyjścia
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Sprawdź sygnały breakout dla każdego assetu
                active_symbols = [p['symbol'] for p in self.positions.values() if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.priority_symbols:
                        if symbol not in active_symbols:
                            signal, confidence = self.generate_breakout_signal(symbol)
                            
                            if signal in ["BREAKOUT_LONG", "LONG"] and confidence >= 0.65:
                                if signal == "BREAKOUT_LONG":
                                    self.logger.info(f"🎯 STRONG BREAKOUT: {symbol} - Confidence: {confidence:.1%}")
                                else:
                                    self.logger.info(f"📈 MOMENTUM SIGNAL: {symbol} - Confidence: {confidence:.1%}")
                                
                                position_id = self.open_breakout_position(symbol)
                                if position_id:
                                    time.sleep(1)  # Małe opóźnienie między pozycjami
                
                # 4. Loguj status portfela
                portfolio_value = self.dashboard_data['account_value']
                diversity = self.dashboard_data['portfolio_diversity'] * 100
                utilization = self.stats['portfolio_utilization'] * 100
                
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Positions: {active_count}/{self.max_simultaneous_positions}")
                self.logger.info(f"🌐 Diversity: {diversity:.1f}% | Utilization: {utilization:.1f}%")
                
                # 5. Czekaj 60 sekund
                for i in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in breakout trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Start breakout trading"""
        self.is_running = True
        self.run_breakout_strategy()

    def stop_trading(self):
        """Stop breakout trading"""
        self.is_running = False
        self.logger.info("🛑 Breakout Trading stopped")

# Global ML bot instance
ml_trading_bot = MLTradingBot(initial_capital=10000, leverage=10)
