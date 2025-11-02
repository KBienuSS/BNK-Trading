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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from flask_cors import CORS

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
        
        # ZAKTUALIZOWANA STRATEGIA ALOKACJI
        self.max_simultaneous_positions = 6
        
        # NOWA ALOKACJA OPARTA NA ANALIZIE
        self.asset_allocation = {
            'ETHUSDT': 0.35,
            'XRPUSDT': 0.25,
            'SOLUSDT': 0.20,
            'BTCUSDT': 0.15,
            'BNBUSDT': 0.05,
            'DOGEUSDT': 0.00,
        }
        
        self.priority_symbols = ['ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'BTCUSDT', 'BNBUSDT']
        
        # ZAKTUALIZOWANE PARAMETRY BREAKOUT
        self.breakout_threshold = 0.015
        self.min_volume_ratio = 1.8
        self.max_position_value = 0.40
        
        # ZAKTUALIZOWANE WIELKOŚCI POZYCJI
        self.position_sizes = {
            'ETHUSDT': 3.2,
            'XRPUSDT': 8500.0,
            'SOLUSDT': 28.0,
            'BTCUSDT': 0.045,
            'BNBUSDT': 8.5,
        }
        
        # NOWE PARAMETRY STRATEGII
        self.trading_hours = {
            'start_utc': 0,
            'end_utc': 23,
            'enabled': False
        }
        
        self.long_bias = 0.95
        self.pyramiding_enabled = True
        self.risk_tolerance = "HIGH"
        
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
            'portfolio_utilization': 0,
            'liquidation_events': 0,
            'macro_clusters': 0
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
            'last_update': datetime.now(),
            'trading_window_active': False
        }
        
        # Initialize ML model
        self.initialize_ml_model()
        
        self.logger.info("🧠 ENHANCED ML TRADING BOT - REALISTIC TP LEVELS")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info("🎯 Target: 20-30% margin returns (2-3% price moves)")

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

    def is_trading_hours(self):
        """Sprawdza czy jesteśmy w preferowanych godzinach handlu"""
        if not self.trading_hours['enabled']:
            self.dashboard_data['trading_window_active'] = True
            return True
            
        current_hour = datetime.utcnow().hour
        is_active = self.trading_hours['start_utc'] <= current_hour <= self.trading_hours['end_utc']
        self.dashboard_data['trading_window_active'] = is_active
        return is_active

    def should_enter_long(self):
        """Decyzja o wejściu LONG z biasem 95%"""
        return random.random() < self.long_bias

    def get_binance_klines(self, symbol: str, interval: str = '3m', limit: int = 100):
        """Get LIVE price data from working APIs"""
        try:
            kucoin_symbol = symbol.replace('USDT', '-USDT')
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={kucoin_symbol}&type=3min"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if candles and len(candles) > 0:
                        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                        for col in ['open', 'close', 'high', 'low', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        if len(df) > limit:
                            df = df.tail(limit)
                            
                        self.logger.info(f"✅ KuCoin LIVE Data for {symbol}: {len(df)} rows, Last: ${df['close'].iloc[-1]:.2f}")
                        return df
                        
        except Exception as e:
            self.logger.warning(f"KuCoin data failed: {e}")
        
        try:
            coin_id = self.symbol_to_coingecko(symbol)
            if coin_id:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1"
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                        for col in ['open', 'high', 'low', 'close']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['volume'] = [10000] * len(df)
                        
                        if len(df) > limit:
                            df = df.tail(limit)
                            
                        self.logger.info(f"✅ CoinGecko Data for {symbol}: {len(df)} rows, Last: ${df['close'].iloc[-1]:.2f}")
                        return df
                        
        except Exception as e:
            self.logger.warning(f"CoinGecko data failed: {e}")
        
        return self.get_realistic_simulation(symbol, limit)

    def get_current_price(self, symbol: str):
        """Get LIVE current price from Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_price = float(data['price'])
                self.logger.info(f"✅ Binance LIVE Price for {symbol}: ${current_price}")
                return current_price
                
        except Exception as e:
            self.logger.warning(f"Binance price failed for {symbol}: {e}")
        
        try:
            kucoin_symbol = symbol.replace('USDT', '-USDT')
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={kucoin_symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    price = float(data['data']['price'])
                    self.logger.info(f"✅ KuCoin Price for {symbol}: ${price}")
                    return price
                    
        except Exception as e:
            self.logger.warning(f"KuCoin price failed: {e}")
        
        try:
            coin_id = self.symbol_to_coingecko(symbol)
            if coin_id:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if coin_id in data and 'usd' in data[coin_id]:
                        price = data[coin_id]['usd']
                        self.logger.info(f"✅ CoinGecko Price for {symbol}: ${price}")
                        return float(price)
                        
        except Exception as e:
            self.logger.warning(f"CoinGecko price failed: {e}")
        
        return self.get_realistic_market_price(symbol)

    def get_realistic_simulation(self, symbol: str, limit: int = 100):
        """Realistic simulation based on current LIVE market prices"""
        import pandas as pd
        import random
        
        base_price = self.get_fallback_price(symbol)
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='3min')
        data = []
        current_price = base_price
        
        for i in range(limit):
            volatility = {
                'BTCUSDT': 0.0015, 'ETHUSDT': 0.002, 'BNBUSDT': 0.0025,
                'SOLUSDT': 0.003, 'XRPUSDT': 0.004, 'DOGEUSDT': 0.005
            }.get(symbol, 0.002)
            
            change = random.gauss(0, volatility)
            current_price = current_price * (1 + change)
            
            data.append({
                'timestamp': dates[i],
                'open': current_price,
                'high': current_price * (1 + abs(random.gauss(0, volatility/2))),
                'low': current_price * (1 - abs(random.gauss(0, volatility/2))),
                'close': current_price * (1 + random.gauss(0, volatility/3)),
                'volume': random.uniform(5000, 20000)
            })
            
            current_price = data[-1]['close']
        
        df = pd.DataFrame(data)
        self.logger.info(f"📊 Realistic Simulation for {symbol}: ${df['close'].iloc[-1]:.2f}")
        return df

    def get_realistic_market_price(self, symbol: str):
        """Get realistic price based on current market conditions"""
        import random
        
        base_price = self.get_fallback_price(symbol)
        
        volatility = {
            'BTCUSDT': 0.0005, 'ETHUSDT': 0.0008, 'BNBUSDT': 0.001,
            'SOLUSDT': 0.0015, 'XRPUSDT': 0.002, 'DOGEUSDT': 0.003
        }.get(symbol, 0.001)
        
        change = random.gauss(0, volatility)
        live_price = base_price * (1 + change)
        live_price = round(live_price, 2)
        
        self.logger.info(f"📊 Realistic Market Price for {symbol}: ${live_price:.2f}")
        return live_price

    def get_fallback_price(self, symbol: str):
        """Fallback prices based on current market"""
        current_market = {
            'BTCUSDT': 112614,
            'ETHUSDT': 3485,
            'BNBUSDT': 582,
            'SOLUSDT': 178,
            'XRPUSDT': 0.615,
            'DOGEUSDT': 0.148
        }
        return current_market.get(symbol, 100)

    def symbol_to_coingecko(self, symbol: str):
        """Convert symbol to CoinGecko ID"""
        mapping = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum', 
            'BNBUSDT': 'binancecoin',
            'SOLUSDT': 'solana',
            'XRPUSDT': 'ripple', 
            'DOGEUSDT': 'dogecoin'
        }
        return mapping.get(symbol, None)

    def detect_breakout_signal(self, symbol: str) -> Tuple[bool, float, float]:
        """Ulepszone wykrywanie breakout"""
        try:
            df = self.get_binance_klines(symbol, '3m', 100)
            if df is None or len(df) < 50:
                return False, 0, 0
            
            resistance_level = df['high'].rolling(20).max().iloc[-1]
            support_level = df['low'].rolling(20).min().iloc[-1]
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            price_above_resistance = current_price > resistance_level
            breakout_strength = (current_price - resistance_level) / resistance_level
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            trend_strength = (current_price - df['close'].iloc[-20]) / df['close'].iloc[-20]
            volatility = df['close'].pct_change().std() * 100
            
            is_breakout = (price_above_resistance and 
                          breakout_strength >= self.breakout_threshold and
                          volume_ratio >= self.min_volume_ratio and
                          trend_strength > 0.01 and
                          volatility > 0.5)
            
            confidence = min(breakout_strength * 12 + volume_ratio * 0.3 + 
                           min(trend_strength * 5, 0.2), 0.95)
            
            if is_breakout:
                self.logger.info(f"🎯 ENHANCED BREAKOUT: {symbol} | Strength: {breakout_strength:.2%}")
            
            return is_breakout, confidence, resistance_level
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced breakout detection: {e}")
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
        """Generuje sygnał oparty na ulepszonej strategii breakout"""
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

    def calculate_dynamic_position_size(self, symbol: str, price: float, confidence: float, is_breakout: bool):
        """Dynamiczne obliczanie wielkości pozycji z pyramidingiem"""
        try:
            # Bazowa alokacja
            base_allocation = self.asset_allocation.get(symbol, 0.15)
            
            # Modyfikator confidence
            confidence_multiplier = 0.6 + (confidence * 0.4)
            
            # Bonus dla breakout
            breakout_bonus = 1.2 if is_breakout else 1.0
            
            # Oblicz wartość pozycji z pyramidingiem
            if self.pyramiding_enabled and self.stats['total_pnl'] > 0:
                # Reinwestuj 60% zysków
                reinvestment = min(self.stats['total_pnl'] * 0.6, self.virtual_capital * 0.3)
                effective_capital = self.virtual_capital + reinvestment
            else:
                effective_capital = self.virtual_capital
            
            position_value = (effective_capital * base_allocation * 
                           confidence_multiplier * breakout_bonus)
            
            # Limit maksymalnej pozycji
            max_position_value = effective_capital * self.max_position_value
            position_value = min(position_value, max_position_value)
            
            # Użyj historycznej wielkości jako referencji
            historical_quantity = self.position_sizes.get(symbol, position_value / price)
            final_quantity = min(position_value / price, historical_quantity * 1.5)
            
            final_position_value = final_quantity * price
            margin_required = final_position_value / self.leverage
            
            return final_quantity, final_position_value, margin_required
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating dynamic position size: {e}")
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

    def calculate_breakout_exit_levels(self, entry_price: float, resistance_level: float, symbol: str) -> Dict:
        """POPRAWIONE: REALISTYCZNE POZIOMY TP/SL PRZY DŹWIGNI"""
        # DOCELOWY ZYSK: 20-30% na marży (2-3% ruchu ceny przy 10x)
        tp_multipliers = {
            'BTCUSDT': 1.022,   # 2.2% TP = 22% zysku przy 10x
            'ETHUSDT': 1.025,   # 2.5% TP = 25% zysku
            'SOLUSDT': 1.028,   # 2.8% TP = 28% zysku  
            'XRPUSDT': 1.026,   # 2.6% TP = 26% zysku
            'BNBUSDT': 1.024,   # 2.4% TP = 24% zysku
        }
        tp_multiplier = tp_multipliers.get(symbol, 1.025)
        
        # STOP LOSS: 10-15% straty na marży (1-1.5% ruchu ceny)
        sl_multipliers = {
            'BTCUSDT': 0.988,   # 1.2% SL = 12% straty przy 10x
            'ETHUSDT': 0.987,   # 1.3% SL = 13% straty
            'SOLUSDT': 0.985,   # 1.5% SL = 15% straty
            'XRPUSDT': 0.986,   # 1.4% SL = 14% straty
            'BNBUSDT': 0.987,   # 1.3% SL = 13% straty
        }
        sl_multiplier = sl_multipliers.get(symbol, 0.987)
        
        take_profit = entry_price * tp_multiplier
        stop_loss = entry_price * sl_multiplier
        
        self.logger.info(f"🎯 REALISTIC TP/SL for {symbol}: TP={tp_multiplier-1:.2%} ({(tp_multiplier-1)*self.leverage:.0f}% margin) | SL={1-sl_multiplier:.2%} ({(1-sl_multiplier)*self.leverage:.0f}% margin)")
        
        return {
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'invalidation': entry_price * 0.98
        }

    def calculate_momentum_exit_levels(self, entry_price: float, symbol: str) -> Dict:
        """POPRAWIONE: KRÓTSZE TP DLA MOMENTUM"""
        # DOCELOWY ZYSK: 15-20% na marży (1.5-2% ruchu ceny)
        tp_multipliers = {
            'BTCUSDT': 1.018,   # 1.8% TP = 18% zysku
            'ETHUSDT': 1.020,   # 2.0% TP = 20% zysku
            'SOLUSDT': 1.022,   # 2.2% TP = 22% zysku
            'XRPUSDT': 1.021,   # 2.1% TP = 21% zysku
            'BNBUSDT': 1.019,   # 1.9% TP = 19% zysku
        }
        tp_multiplier = tp_multipliers.get(symbol, 1.020)
        
        return {
            'take_profit': entry_price * tp_multiplier,
            'stop_loss': entry_price * 0.985,    # 1.5% SL
            'invalidation': entry_price * 0.975  # 2.5% invalidation
        }

    def generate_macro_signal(self):
        """Generuje sygnał makro dla klastrów transakcji"""
        try:
            signals = {}
            for symbol in ['ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
                signal, confidence = self.generate_breakout_signal(symbol)
                signals[symbol] = (signal, confidence)
            
            strong_signals = [s for s in signals.items() 
                            if s[1][0] == "BREAKOUT_LONG" and s[1][1] >= 0.7]
            
            if len(strong_signals) >= 2:
                self.logger.info("🎯 MACRO SIGNAL DETECTED - Multiple breakouts")
                return True, strong_signals
            
            return False, []
            
        except Exception as e:
            self.logger.error(f"❌ Error generating macro signal: {e}")
            return False, []

    def open_macro_positions(self, strong_signals):
        """Otwiera zsynchronizowane pozycje w klastrach"""
        opened_positions = []
        
        for symbol, (signal, confidence) in strong_signals:
            if len(opened_positions) >= 2:
                break
                
            position_id = self.open_breakout_position(symbol)
            if position_id:
                opened_positions.append(position_id)
                self.stats['macro_clusters'] += 1
                time.sleep(2)
        
        return opened_positions

    def open_breakout_position(self, symbol: str):
        """Otwiera pozycję breakout z uwzględnieniem nowej strategii"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        signal, confidence = self.generate_breakout_signal(symbol)
        if signal not in ["BREAKOUT_LONG", "LONG"] or confidence < 0.65:
            return None
        
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        is_breakout = signal == "BREAKOUT_LONG"
        quantity, position_value, margin_required = self.calculate_dynamic_position_size(
            symbol, current_price, confidence, is_breakout
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
        
        # UŻYJ POPRAWIONYCH POZIOMÓW WYJŚCIA
        if is_breakout:
            _, _, resistance_level = self.detect_breakout_signal(symbol)
            exit_levels = self.calculate_breakout_exit_levels(current_price, resistance_level, symbol)
        else:
            exit_levels = self.calculate_momentum_exit_levels(current_price, symbol)
        
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
        
        # LOGOWANIE Z REALISTYCZNYMI POZIOMAMI
        tp_percent = (exit_levels['take_profit'] - current_price) / current_price * 100
        sl_percent = (current_price - exit_levels['stop_loss']) / current_price * 100
        tp_margin = tp_percent * self.leverage
        sl_margin = sl_percent * self.leverage
        
        self.logger.info(f"🎯 {'BREAKOUT' if is_breakout else 'MOMENTUM'} OPEN: {symbol} @ ${current_price:.2f}")
        self.logger.info(f"   📊 TP: ${exit_levels['take_profit']:.2f} ({tp_percent:.1f}% = {tp_margin:.0f}% margin)")
        self.logger.info(f"   🛑 SL: ${exit_levels['stop_loss']:.2f} ({sl_percent:.1f}% = {sl_margin:.0f}% margin)")
        self.logger.info(f"   💰 Position: ${position_value:.2f} | Margin: ${margin_required:.2f}")
        
        return position_id

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
            
            if current_price >= position['exit_plan']['take_profit']:
                exit_reason = "TAKE_PROFIT"
                # Oblicz rzeczywisty zysk
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                actual_margin_return = pnl_pct * self.leverage * 100
                self.logger.info(f"🟢 TP HIT: {position['symbol']} - {actual_margin_return:.1f}% margin return")
            
            elif self.should_close_based_on_3min_candle(position['symbol'], position):
                exit_reason = "STOP_LOSS_3MIN"
            
            elif current_price <= position['exit_plan']['stop_loss']:
                exit_reason = "STOP_LOSS_CLASSIC"
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                actual_margin_return = pnl_pct * self.leverage * 100
                self.logger.info(f"🔴 SL HIT: {position['symbol']} - {actual_margin_return:.1f}% margin return")
            
            elif current_price <= position['exit_plan']['invalidation']:
                exit_reason = "INVALIDATION"
            
            elif current_price <= position['liquidation_price']:
                exit_reason = "LIQUIDATION"
                self.stats['liquidation_events'] += 1
            
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
            if realized_pnl_after_fee > self.stats['biggest_win']:
                self.stats['biggest_win'] = realized_pnl_after_fee
        else:
            self.stats['losing_trades'] += 1
            if realized_pnl_after_fee < self.stats['biggest_loss']:
                self.stats['biggest_loss'] = realized_pnl_after_fee
        
        position['status'] = 'CLOSED'
        
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        strategy_icon = "🎯" if position.get('strategy') == 'BREAKOUT' else "📈"
        margin_return = pnl_pct * self.leverage * 100
        self.logger.info(f"{pnl_color} {strategy_icon} CLOSE: {position['symbol']} - P&L: ${realized_pnl_after_fee:+.2f} ({margin_return:+.1f}% margin) - Reason: {exit_reason}")

    def get_dashboard_data(self):
        """Prepare dashboard data - ZAKTUALIZOWANA WERSJA Z TP/SL INFO"""
        active_positions = []
        total_confidence = 0
        confidence_count = 0
        
        # Pobierz aktualne ceny dla wszystkich aktywnych pozycji
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = self.get_current_price(position['symbol'])
                
                # Oblicz unrealized PnL
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
                # Oblicz odległości do TP/SL
                tp_distance_pct = (position['exit_plan']['take_profit'] - current_price) / current_price * 100
                sl_distance_pct = (current_price - position['exit_plan']['stop_loss']) / current_price * 100
                tp_distance_margin = tp_distance_pct * position['leverage']
                sl_distance_margin = sl_distance_pct * position['leverage']
                
                active_positions.append({
                    'position_id': position_id,
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'liquidation_price': position['liquidation_price'],
                    'margin': position['margin'],
                    'unrealized_pnl': unrealized_pnl,
                    'confidence': position.get('confidence', 0),
                    'strategy': position.get('strategy', 'MOMENTUM'),
                    'exit_plan': position['exit_plan'],
                    'tp_distance_pct': round(tp_distance_pct, 2),
                    'sl_distance_pct': round(sl_distance_pct, 2),
                    'tp_distance_margin': round(tp_distance_margin, 1),
                    'sl_distance_margin': round(sl_distance_margin, 1)
                })
        
        # CONFIDENCE LEVELS DLA KAŻDEGO ASSETU
        confidence_levels = {}
        for symbol in self.priority_symbols:
            try:
                signal, confidence = self.generate_breakout_signal(symbol)
                confidence_percent = round(confidence * 100, 1)
                confidence_levels[symbol] = confidence_percent
                
                # Do obliczenia średniej confidence
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1
                    
                self.logger.info(f"🔮 {symbol} Confidence: {confidence_percent}%")
                
            except Exception as e:
                self.logger.error(f"❌ Error calculating confidence for {symbol}: {e}")
                confidence_levels[symbol] = 0
        
        # Oblicz średnią confidence
        avg_confidence = round((total_confidence / confidence_count * 100), 1) if confidence_count > 0 else 0
        
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
        
        # Oblicz całkowity zwrot
        total_return_pct = ((self.dashboard_data['account_value'] - 10000) / 10000) * 100
        
        # Sprawdź czy jesteśmy w godzinach handlu
        trading_window_active = self.is_trading_hours()
        current_hour_utc = datetime.utcnow().hour
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'total_fees': round(self.stats['total_fees'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2)
            },
            'performance_metrics': {
                'avg_leverage': self.leverage,
                'total_return_pct': round(total_return_pct, 2),
                'portfolio_diversity': round(self.dashboard_data['portfolio_diversity'] * 100, 1),
                'portfolio_utilization': round(self.stats['portfolio_utilization'] * 100, 1),
                'breakout_trades': self.stats['breakout_trades'],
                'macro_clusters': self.stats['macro_clusters'],
                'liquidation_events': self.stats['liquidation_events'],
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'biggest_win': round(self.stats['biggest_win'], 2),
                'biggest_loss': round(self.stats['biggest_loss'], 2),
                'avg_confidence': avg_confidence
            },
            'trading_hours': {
                'active': trading_window_active,
                'current_utc_hour': current_hour_utc,
                'window_start': self.trading_hours['start_utc'],
                'window_end': self.trading_hours['end_utc']
            },
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
            'strategy_profile': {
                'long_bias': f"{self.long_bias * 100:.0f}%",
                'pyramiding': self.pyramiding_enabled,
                'risk_tolerance': self.risk_tolerance,
                'max_positions': self.max_simultaneous_positions
            },
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_enhanced_breakout_strategy(self):
        """Ulepszona główna pętla strategii z realistycznymi TP"""
        self.logger.info("🚀 ENHANCED BREAKOUT STRATEGY - REALISTIC TP LEVELS")
        self.logger.info("🎯 Target: 20-30% margin returns (2-3% price moves)")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now()
                current_hour_utc = current_time.hour
                
                self.logger.info(f"\n🔄 Enhanced Iteration #{iteration} | UTC: {current_hour_utc:02d}:00")
                
                # Sprawdź czy jesteśmy w godzinach handlu
                if not self.is_trading_hours():
                    self.logger.info("⏸️ Outside trading hours (16:00-23:59 UTC) - Waiting...")
                    time.sleep(300)  # 5 minut przerwy
                    continue
                
                # 1. Aktualizuj P&L
                self.update_positions_pnl()
                
                # 2. Sprawdź warunki wyjścia
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. SPRAWDŹ SYGNAŁ MAKRO (klastry transakcji)
                macro_signal, strong_signals = self.generate_macro_signal()
                if macro_signal:
                    self.logger.info("🎯 EXECUTING MACRO CLUSTER STRATEGY")
                    self.open_macro_positions(strong_signals)
                else:
                    # 4. Standardowe sprawdzanie sygnałów per asset
                    active_symbols = [p['symbol'] for p in self.positions.values() 
                                    if p['status'] == 'ACTIVE']
                    active_count = len(active_symbols)
                    
                    if active_count < self.max_simultaneous_positions:
                        for symbol in self.priority_symbols:
                            if symbol not in active_symbols:
                                # Sprawdź bias LONG
                                if not self.should_enter_long():
                                    self.logger.info(f"⏹️ Skipping {symbol} due to long bias")
                                    continue
                                
                                signal, confidence = self.generate_breakout_signal(symbol)
                                
                                entry_conditions = (
                                    signal in ["BREAKOUT_LONG", "LONG"] and 
                                    confidence >= 0.65 and
                                    self.virtual_balance > 100  # Minimalny margin
                                )
                                
                                if entry_conditions:
                                    is_breakout = signal == "BREAKOUT_LONG"
                                    self.logger.info(f"🎯 {'BREAKOUT' if is_breakout else 'MOMENTUM'}: {symbol} - Confidence: {confidence:.1%}")
                                    
                                    position_id = self.open_breakout_position(symbol)
                                    if position_id:
                                        time.sleep(1)  # Opóźnienie między pozycjami
                
                # 5. Loguj status
                portfolio_value = self.dashboard_data['account_value']
                active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
                
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active: {active_count}/{self.max_simultaneous_positions}")
                self.logger.info(f"🕒 Trading Window: {self.trading_hours['start_utc']:02d}:00-{self.trading_hours['end_utc']:02d}:59 UTC")
                
                # 6. Krótszy interwał w godzinach handlu
                wait_time = 30 if self.is_trading_hours() else 60
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in enhanced trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Start enhanced breakout trading"""
        self.is_running = True
        self.run_enhanced_breakout_strategy()

    def stop_trading(self):
        """Stop breakout trading"""
        self.is_running = False
        self.logger.info("🛑 Enhanced Breakout Trading stopped")

# Global ML bot instance
ml_trading_bot = MLTradingBot(initial_capital=10000, leverage=10)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Store chart data in bot's memory
chart_data_storage = {
    'labels': [],
    'values': []
}

# Ładuj dane z pliku przy starcie
def load_chart_data_from_file():
    try:
        if os.path.exists('chart_data.json'):
            with open('chart_data.json', 'r') as f:
                data = json.load(f)
                chart_data_storage.update(data)
                print(f"✅ Loaded chart data: {len(data.get('labels', []))} points")
        else:
            print("ℹ️ No existing chart data file found, starting fresh")
    except Exception as e:
        print(f"❌ Error loading chart data: {e}")

# Wywołaj przy starcie
load_chart_data_from_file()

@app.route('/api/save-chart-data', methods=['POST'])
def save_chart_data():
    """Zapisuje dane wykresu do pamięci bota"""
    try:
        data = request.get_json()
        if data and 'labels' in data and 'values' in data:
            chart_data_storage['labels'] = data['labels']
            chart_data_storage['values'] = data['values']
            
            # Zapis do pliku
            with open('chart_data.json', 'w') as f:
                json.dump(chart_data_storage, f, indent=2)
            
            print(f"✅ Chart data saved: {len(data['labels'])} labels, {len(data['values'])} values")
            return jsonify({'status': 'success', 'message': f'Chart data saved: {len(data["labels"])} points'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid data format'})
    except Exception as e:
        print(f"❌ Error saving chart data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/load-chart-data', methods=['GET'])
def load_chart_data():
    """Ładuje dane wykresu z pamięci bota"""
    try:
        # Zawsze ładuj z pliku dla aktualnych danych
        load_chart_data_from_file()
        
        print(f"📊 Returning chart data: {len(chart_data_storage['labels'])} points")
        return jsonify({
            'status': 'success', 
            'chartData': chart_data_storage
        })
    except Exception as e:
        print(f"❌ Error loading chart data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/trading-data', methods=['GET'])
def get_trading_data():
    """Existing endpoint - teraz z AUTOMATYCZNYM zapisem danych wykresu"""
    try:
        dashboard_data = ml_trading_bot.get_dashboard_data()
        
        # AUTO-SAVE CHART DATA - POPRAWIONE: zawsze zapisuj przy każdej aktualizacji
        current_value = dashboard_data['account_summary']['total_value']
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Sprawdź czy warto dodać nowy punkt (różnica > 1$ lub pierwszy punkt)
        should_add_point = False
        if not chart_data_storage['values']:
            should_add_point = True  # Pierwszy punkt
        else:
            last_value = chart_data_storage['values'][-1]
            if abs(current_value - last_value) >= 1.0:  # Różnica co najmniej 1$
                should_add_point = True
        
        if should_add_point:
            chart_data_storage['labels'].append(current_time)
            chart_data_storage['values'].append(current_value)
            
            # Ogranicz do ostatnich 100 punktów
            if len(chart_data_storage['labels']) > 100:
                chart_data_storage['labels'] = chart_data_storage['labels'][-100:]
                chart_data_storage['values'] = chart_data_storage['values'][-100:]
            
            # AUTO-SAVE do pliku
            try:
                with open('chart_data.json', 'w') as f:
                    json.dump(chart_data_storage, f, indent=2)
                print(f"💾 Auto-saved chart data: {len(chart_data_storage['labels'])} points")
            except Exception as e:
                print(f"❌ Error auto-saving chart data: {e}")
        
        return jsonify(dashboard_data)
    except Exception as e:
        print(f"❌ Error in trading-data endpoint: {e}")
        return jsonify({'error': str(e)})

# Pozostałe endpointy pozostają bez zmian
@app.route('/api/bot-status', methods=['GET'])
def get_bot_status():
    return jsonify({
        'status': 'running' if ml_trading_bot.is_running else 'stopped',
        'capital': ml_trading_bot.virtual_capital,
        'active_positions': len([p for p in ml_trading_bot.positions.values() if p['status'] == 'ACTIVE'])
    })

@app.route('/api/start-bot', methods=['POST'])
def start_bot():
    try:
        if not ml_trading_bot.is_running:
            threading.Thread(target=ml_trading_bot.start_trading, daemon=True).start()
            return jsonify({'status': 'Bot started successfully'})
        else:
            return jsonify({'status': 'Bot is already running'})
    except Exception as e:
        return jsonify({'status': f'Error starting bot: {str(e)}'})

@app.route('/api/stop-bot', methods=['POST'])
def stop_bot():
    try:
        ml_trading_bot.stop_trading()
        return jsonify({'status': 'Bot stopped successfully'})
    except Exception as e:
        return jsonify({'status': f'Error stopping bot: {str(e)}'})

@app.route('/api/force-update', methods=['POST'])
def force_update():
    try:
        ml_trading_bot.update_positions_pnl()
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'status': f'Error updating data: {str(e)}'})

if __name__ == '__main__':
    print("🚀 Starting Enhanced ML Trading Bot Server...")
    print("📍 Dashboard available at: http://localhost:5000")
    print("📊 Chart data synchronization: ENABLED")
    print("🎯 REALISTIC TP LEVELS: 20-30% margin returns (2-3% price moves)")
    app.run(host='0.0.0.0', port=5000, debug=True)
