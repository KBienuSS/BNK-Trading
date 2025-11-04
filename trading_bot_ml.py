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
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qwen_trading_bot.log', encoding='utf-8')
    ]
)

class QwenTradingBot:
    def __init__(self, initial_capital=10000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # TYLKO PROFIL QWEN3 - FIXED
        self.llm_profile = 'Qwen'
        
        # PARAMETRY OPERACYJNE
        self.max_simultaneous_positions = 4
        self.assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']
        
        # STATYSTYKI
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'long_trades': 0,
            'short_trades': 0,
            'avg_holding_time': 0,
            'portfolio_utilization': 0
        }
        
        # DASHBOARD
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': leverage,
            'average_confidence': 0,
            'portfolio_diversity': 0,
            'last_update': datetime.now(),
            'active_profile': self.llm_profile  # ZAWSZE Qwen
        }
        
        self.logger.info("🧠 QWEN3 TRADING BOT - AUTO MODE")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info("🎯 Profile: Qwen3 (Fixed - No Switching)")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Pobiera RZECZYWISTE ceny tylko z API"""
        try:
            # 1. Binance API
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                current_price = float(data['price'])
                self.logger.info(f"✅ Binance LIVE Price for {symbol}: ${current_price}")
                return current_price
                
        except Exception as e:
            self.logger.warning(f"Binance failed for {symbol}: {e}")
        
        try:
            # 2. KuCoin API
            kucoin_symbol = symbol.replace('USDT', '-USDT')
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={kucoin_symbol}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    price = float(data['data']['price'])
                    self.logger.info(f"✅ KuCoin Price for {symbol}: ${price}")
                    return price
                    
        except Exception as e:
            self.logger.warning(f"KuCoin failed: {e}")
        
        try:
            # 3. CoinGecko API
            coin_id = self.symbol_to_coingecko(symbol)
            if coin_id:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if coin_id in data and 'usd' in data[coin_id]:
                        price = data[coin_id]['usd']
                        self.logger.info(f"✅ CoinGecko Price for {symbol}: ${price}")
                        return float(price)
                        
        except Exception as e:
            self.logger.warning(f"CoinGecko failed: {e}")
        
        self.logger.error(f"❌ ALL PRICE APIS FAILED for {symbol}")
        return None

    def symbol_to_coingecko(self, symbol: str) -> Optional[str]:
        """Konwertuje symbol na CoinGecko ID"""
        mapping = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum', 
            'BNBUSDT': 'binancecoin',
            'SOLUSDT': 'solana',
            'XRPUSDT': 'ripple'
        }
        return mapping.get(symbol, None)

    def get_historical_data(self, symbol: str, interval: str = '3m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Pobiera RZECZYWISTE dane historyczne"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                self.logger.info(f"✅ Binance Historical Data for {symbol}: {len(df)} candles")
                return df
                
        except Exception as e:
            self.logger.warning(f"Binance historical data failed: {e}")
        
        try:
            kucoin_symbol = symbol.replace('USDT', '-USDT')
            kucoin_interval = self.convert_interval_to_kucoin(interval)
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={kucoin_symbol}&type={kucoin_interval}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if candles and len(candles) > 0:
                        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                        for col in ['open', 'close', 'high', 'low', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='s')
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        if len(df) > limit:
                            df = df.tail(limit)
                            
                        self.logger.info(f"✅ KuCoin Historical Data for {symbol}: {len(df)} candles")
                        return df
                        
        except Exception as e:
            self.logger.warning(f"KuCoin historical data failed: {e}")
        
        try:
            coin_id = self.symbol_to_coingecko(symbol)
            if coin_id:
                days = max(1, limit // 24)
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
                response = requests.get(url, timeout=10)
                
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
                            
                        self.logger.info(f"✅ CoinGecko Historical Data for {symbol}: {len(df)} candles")
                        return df
                        
        except Exception as e:
            self.logger.warning(f"CoinGecko historical data failed: {e}")
        
        self.logger.error(f"❌ ALL HISTORICAL APIS FAILED for {symbol}")
        return None

    def convert_interval_to_kucoin(self, interval: str) -> str:
        """Konwertuje interwał Binance na KuCoin"""
        mapping = {
            '1m': '1min',
            '3m': '3min', 
            '5m': '5min',
            '15m': '15min',
            '1h': '1hour',
            '4h': '4hour',
            '1d': '1day'
        }
        return mapping.get(interval, '3min')

    def analyze_simple_momentum(self, symbol: str) -> float:
        """Analiza momentum na RZECZYWISTYCH danych"""
        try:
            df = self.get_historical_data(symbol, '3m', 20)
            if df is None or len(df) < 5:
                return 0.0
            
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return 0.0
                
            price_changes = df['close'].pct_change().dropna().tail(5)
            if len(price_changes) == 0:
                return 0.0
                
            momentum = price_changes.mean()
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing momentum for {symbol}: {e}")
            return 0.0

    def check_volume_activity(self, symbol: str) -> bool:
        """Sprawdza aktywność wolumenu na RZECZYWISTYCH danych"""
        try:
            df = self.get_historical_data(symbol, '3m', 20)
            if df is None or len(df) < 10:
                return False
            
            current_volume = df['volume'].iloc[-1] if len(df) > 0 else 0
            avg_volume = df['volume'].tail(10).mean()
            
            return current_volume > avg_volume * 1.3
            
        except Exception as e:
            self.logger.error(f"❌ Error checking volume for {symbol}: {e}")
            return False

    def generate_qwen_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał w stylu Qwen3 - AGRESYWNY"""
        # PARAMETRY QWEN3 - WYSOKA AGRESYWNOŚĆ
        base_confidence = 0.85  # Wysoka confidence
        short_frequency = 0.2   # Rzadko shortuje
        
        # RZECZYWISTE obserwacje
        momentum = self.analyze_simple_momentum(symbol)
        volume_active = self.check_volume_activity(symbol)
        
        confidence_modifiers = 0
        
        if momentum > 0.01:
            confidence_modifiers += 0.20  # Większy bonus dla Qwena
        elif momentum < -0.01: 
            confidence_modifiers += 0.15
            
        if volume_active:
            confidence_modifiers += 0.15  # Większy bonus volume
            
        final_confidence = min(base_confidence + confidence_modifiers + random.uniform(-0.1, 0.1), 0.95)
        final_confidence = max(final_confidence, 0.3)  # Minimum 30% dla Qwena
        
        # AGRESYWNE WARUNKI QWENA
        if momentum > 0.01 and volume_active:  # Mniej restrykcyjne warunki
            signal = "LONG"
        elif momentum < -0.01 and volume_active:
            if random.random() < short_frequency:
                signal = "SHORT"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
            
        self.logger.info(f"🎯 QWEN3 SIGNAL: {symbol} -> {signal} (Conf: {final_confidence:.1%})")
        
        return signal, final_confidence

    def calculate_qwen_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji w stylu Qwen3 - AGRESYWNY"""
        # QWEN3 - WYSOKA ALOKACJA
        base_allocation = 0.30  # 30% kapitału na pozycję
        
        confidence_multiplier = 0.6 + (confidence * 0.4)  # Większy wpływ confidence
        
        # QWEN - VERY AGGRESSIVE
        sizing_multiplier = 1.5
        
        position_value = (self.virtual_capital * base_allocation * 
                         confidence_multiplier * sizing_multiplier)
        
        max_position_value = self.virtual_capital * 0.5  # 50% max dla Qwena
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def calculate_qwen_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """Oblicza plan wyjścia w stylu Qwen3 - AGRESYWNY"""
        # QWEN3 - AGRESYWNE TP/SL
        if confidence > 0.7:
            if side == "LONG":
                take_profit = entry_price * 1.025  # 2.5% TP
                stop_loss = entry_price * 0.985    # 1.5% SL
            else:
                take_profit = entry_price * 0.975  # 2.5% TP
                stop_loss = entry_price * 1.015    # 1.5% SL
        elif confidence > 0.5:
            if side == "LONG":
                take_profit = entry_price * 1.018  # 1.8% TP
                stop_loss = entry_price * 0.988    # 1.2% SL
            else:
                take_profit = entry_price * 0.982  # 1.8% TP
                stop_loss = entry_price * 1.012    # 1.2% SL
        else:
            if side == "LONG":
                take_profit = entry_price * 1.012  # 1.2% TP
                stop_loss = entry_price * 0.992    # 0.8% SL
            else:
                take_profit = entry_price * 0.988  # 1.2% TP
                stop_loss = entry_price * 1.008    # 0.8% SL
        
        # QWEN - HIGH RISK MULTIPLIER
        risk_multiplier = 1.2
        
        if side == "LONG":
            take_profit = entry_price + (take_profit - entry_price) * risk_multiplier
            stop_loss = entry_price - (entry_price - stop_loss) * risk_multiplier
        else:
            take_profit = entry_price - (entry_price - take_profit) * risk_multiplier
            stop_loss = entry_price + (stop_loss - entry_price) * risk_multiplier
        
        return {
            'take_profit': round(take_profit, 4),
            'stop_loss': round(stop_loss, 4),
            'invalidation': entry_price * 0.97 if side == "LONG" else entry_price * 1.03,
            'max_holding_hours': random.randint(2, 8)  # Dłuższe holdowanie
        }

    def should_enter_trade(self) -> bool:
        """Qwen3 - WYSOKA CZĘSTOTLIWOŚĆ TRADINGU"""
        frequency_chance = 0.7  # 70% szans na transakcję
        return random.random() < frequency_chance

    def open_qwen_position(self, symbol: str):
        """Otwiera pozycję w stylu Qwen3"""
        if not self.should_enter_trade():
            return None
            
        current_price = self.get_current_price(symbol)
        if current_price is None:
            self.logger.error(f"❌ Cannot open position for {symbol} - no price data")
            return None
            
        signal, confidence = self.generate_qwen_signal(symbol)
        if signal == "HOLD" or confidence < 0.4:  # Wyższy próg dla Qwena
            return None
            
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
            
        quantity, position_value, margin_required = self.calculate_qwen_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
            
        exit_plan = self.calculate_qwen_exit_plan(current_price, confidence, signal)
        
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        position_id = f"qwen_{self.position_id}"
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
            'llm_profile': self.llm_profile,  # ZAWSZE Qwen
            'exit_plan': exit_plan
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        if signal == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        tp_distance = (exit_plan['take_profit'] - current_price) / current_price * 100
        sl_distance = (current_price - exit_plan['stop_loss']) / current_price * 100
        
        self.logger.info(f"🎯 QWEN3 OPEN: {symbol} {signal} @ ${current_price:.4f}")
        self.logger.info(f"   📊 Confidence: {confidence:.1%} | Size: ${position_value:.2f}")
        self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} ({tp_distance:+.2f}%)")
        self.logger.info(f"   🛑 SL: {exit_plan['stop_loss']:.4f} ({sl_distance:+.2f}%)")
        
        return position_id

    def update_positions_pnl(self):
        total_unrealized = 0
        total_margin = 0
        total_confidence = 0
        confidence_count = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE':
                continue
                
            current_price = self.get_current_price(position['symbol'])
            if current_price is None:
                continue
                
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
            total_unrealized += unrealized_pnl
            total_margin += position['margin']
            total_confidence += position['confidence']
            confidence_count += 1
        
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = total_confidence / confidence_count
        
        if self.virtual_capital > 0:
            self.stats['portfolio_utilization'] = total_margin / self.virtual_capital
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
                
            current_price = position.get('current_price', self.get_current_price(position['symbol']))
            if current_price is None:
                continue
                
            exit_reason = None
            exit_plan = position['exit_plan']
            
            if position['side'] == 'LONG':
                if current_price >= exit_plan['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= exit_plan['stop_loss']:
                    exit_reason = "STOP_LOSS"
                elif current_price <= exit_plan['invalidation']:
                    exit_reason = "INVALIDATION"
                elif current_price <= position['liquidation_price']:
                    exit_reason = "LIQUIDATION"
            else:
                if current_price <= exit_plan['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price >= exit_plan['stop_loss']:
                    exit_reason = "STOP_LOSS"
                elif current_price >= exit_plan['invalidation']:
                    exit_reason = "INVALIDATION"
                elif current_price >= position['liquidation_price']:
                    exit_reason = "LIQUIDATION"
            
            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
            if holding_time > exit_plan['max_holding_hours']:
                exit_reason = "TIME_EXPIRED"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
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
            'llm_profile': position['llm_profile'],
            'confidence': position['confidence'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'holding_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600
        }
        
        self.trade_history.append(trade_record)
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        total_holding = sum((t['exit_time'] - t['entry_time']).total_seconds() 
                          for t in self.trade_history) / 3600
        self.stats['avg_holding_time'] = total_holding / len(self.trade_history) if self.trade_history else 0
        
        position['status'] = 'CLOSED'
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        
        margin_return = pnl_pct * self.leverage * 100
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        self.logger.info(f"{pnl_color} CLOSE: {position['symbol']} {position['side']} - P&L: ${realized_pnl_after_fee:+.2f} ({margin_return:+.1f}% margin) - Reason: {exit_reason}")

    def get_portfolio_diversity(self) -> float:
        try:
            active_positions = [p for p in self.positions.values() if p['status'] == 'ACTIVE']
            if not active_positions:
                return 0
            
            total_margin = sum(p['margin'] for p in active_positions)
            if total_margin == 0:
                return 0
            
            concentration_index = sum((p['margin'] / total_margin) ** 2 for p in active_positions)
            diversity = 1 - concentration_index
            
            return diversity
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating portfolio diversity: {e}")
            return 0

    def get_dashboard_data(self):
        active_positions = []
        total_confidence = 0
        confidence_count = 0
        
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price', self.get_current_price(position['symbol']))
                if current_price is None:
                    continue
                    
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
                tp_distance_pct = (position['exit_plan']['take_profit'] - current_price) / current_price * 100
                sl_distance_pct = (current_price - position['exit_plan']['stop_loss']) / current_price * 100
                
                active_positions.append({
                    'position_id': position_id,
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'margin': position['margin'],
                    'unrealized_pnl': unrealized_pnl,
                    'confidence': position['confidence'],
                    'llm_profile': position['llm_profile'],
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'exit_plan': position['exit_plan'],
                    'tp_distance_pct': round(tp_distance_pct, 2),
                    'sl_distance_pct': round(sl_distance_pct, 2)
                })
                
                total_confidence += position['confidence']
                confidence_count += 1
        
        # Confidence levels dla każdego assetu
        confidence_levels = {}
        for symbol in self.assets:
            try:
                signal, confidence = self.generate_qwen_signal(symbol)
                confidence_levels[symbol] = round(confidence * 100, 1)
            except:
                confidence_levels[symbol] = 0
        
        recent_trades = []
        for trade in self.trade_history[-10:]:
            recent_trades.append({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'realized_pnl': trade['realized_pnl'],
                'exit_reason': trade['exit_reason'],
                'llm_profile': trade['llm_profile'],
                'confidence': trade['confidence'],
                'holding_hours': round(trade['holding_hours'], 2),
                'exit_time': trade['exit_time'].strftime('%H:%M:%S')
            })
        
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        total_return_pct = ((self.dashboard_data['account_value'] - 10000) / 10000) * 100
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2),
                'unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2)
            },
            'performance_metrics': {
                'total_return_pct': round(total_return_pct, 2),
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'long_trades': self.stats['long_trades'],
                'short_trades': self.stats['short_trades'],
                'avg_holding_hours': round(self.stats['avg_holding_time'], 2),
                'portfolio_utilization': round(self.stats['portfolio_utilization'] * 100, 1),
                'portfolio_diversity': round(self.get_portfolio_diversity() * 100, 1),
                'avg_confidence': round(self.dashboard_data['average_confidence'] * 100, 1)
            },
            'llm_config': {
                'active_profile': self.llm_profile,  # ZAWSZE Qwen
                'available_profiles': ['Qwen'],  # TYLKO Qwen
                'max_positions': self.max_simultaneous_positions,
                'leverage': self.leverage
            },
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_qwen_trading_strategy(self):
        """Główna pętla strategii Qwen3"""
        self.logger.info("🚀 STARTING QWEN3 TRADING STRATEGY")
        self.logger.info("🎯 Profile: Qwen3 (Auto Mode - No Switching)")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 Qwen3 Trading Iteration #{iteration}")
                
                self.update_positions_pnl()
                
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                active_symbols = [p['symbol'] for p in self.positions.values() 
                                if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.assets:
                        if symbol not in active_symbols:
                            position_id = self.open_qwen_position(symbol)
                            if position_id:
                                time.sleep(1)
                
                portfolio_value = self.dashboard_data['account_value']
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active Positions: {active_count}/{self.max_simultaneous_positions}")
                
                wait_time = random.randint(20, 60)  # Krótsze interwały dla Qwena
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in Qwen3 trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        self.is_running = True
        threading.Thread(target=self.run_qwen_trading_strategy, daemon=True).start()
        self.logger.info("🚀 Qwen3 Trading Bot started")

    def stop_trading(self):
        self.is_running = False
        self.logger.info("🛑 Qwen3 Trading Bot stopped")

# Globalna instancja bota
qwen_trading_bot = QwenTradingBot(initial_capital=10000, leverage=10)

# Flask app
app = Flask(__name__)
CORS(app)

chart_data_storage = {
    'labels': [],
    'values': []
}

def load_chart_data_from_file():
    try:
        if os.path.exists('qwen_chart_data.json'):
            with open('qwen_chart_data.json', 'r') as f:
                data = json.load(f)
                chart_data_storage.update(data)
                print(f"✅ Loaded Qwen chart data: {len(data.get('labels', []))} points")
    except Exception as e:
        print(f"❌ Error loading Qwen chart data: {e}")

load_chart_data_from_file()

@app.route('/api/save-chart-data', methods=['POST'])
def save_chart_data():
    try:
        data = request.get_json()
        if data and 'labels' in data and 'values' in data:
            chart_data_storage['labels'] = data['labels']
            chart_data_storage['values'] = data['values']
            
            with open('qwen_chart_data.json', 'w') as f:
                json.dump(chart_data_storage, f, indent=2)
            
            return jsonify({'status': 'success', 'message': f'Chart data saved: {len(data["labels"])} points'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid data format'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/load-chart-data', methods=['GET'])
def load_chart_data():
    try:
        load_chart_data_from_file()
        return jsonify({
            'status': 'success', 
            'chartData': chart_data_storage
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/trading-data', methods=['GET'])
def get_trading_data():
    try:
        dashboard_data = qwen_trading_bot.get_dashboard_data()
        
        current_value = dashboard_data['account_summary']['total_value']
        current_time = datetime.now().strftime('%H:%M:%S')
        
        should_add_point = False
        if not chart_data_storage['values']:
            should_add_point = True
        else:
            last_value = chart_data_storage['values'][-1]
            if abs(current_value - last_value) >= 1.0:
                should_add_point = True
        
        if should_add_point:
            chart_data_storage['labels'].append(current_time)
            chart_data_storage['values'].append(current_value)
            
            if len(chart_data_storage['labels']) > 100:
                chart_data_storage['labels'] = chart_data_storage['labels'][-100:]
                chart_data_storage['values'] = chart_data_storage['values'][-100:]
            
            try:
                with open('qwen_chart_data.json', 'w') as f:
                    json.dump(chart_data_storage, f, indent=2)
            except Exception as e:
                print(f"❌ Error auto-saving chart data: {e}")
        
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/bot-status', methods=['GET'])
def get_bot_status():
    return jsonify({
        'status': 'running' if qwen_trading_bot.is_running else 'stopped',
        'capital': qwen_trading_bot.virtual_capital,
        'active_positions': len([p for p in qwen_trading_bot.positions.values() if p['status'] == 'ACTIVE']),
        'active_profile': qwen_trading_bot.llm_profile  # ZAWSZE Qwen
    })

@app.route('/api/start-bot', methods=['POST'])
def start_bot():
    try:
        if not qwen_trading_bot.is_running:
            qwen_trading_bot.start_trading()
            return jsonify({'status': 'Qwen3 Bot started successfully'})
        else:
            return jsonify({'status': 'Qwen3 Bot is already running'})
    except Exception as e:
        return jsonify({'status': f'Error starting Qwen3 bot: {str(e)}'})

@app.route('/api/stop-bot', methods=['POST'])
def stop_bot():
    try:
        qwen_trading_bot.stop_trading()
        return jsonify({'status': 'Qwen3 Bot stopped successfully'})
    except Exception as e:
        return jsonify({'status': f'Error stopping Qwen3 bot: {str(e)}'})

# USUNIĘTY ENDPOINT DO ZMIANY PROFILU - NIE MA TAKIEJ MOŻLIWOŚCI

@app.route('/api/force-update', methods=['POST'])
def force_update():
    try:
        qwen_trading_bot.update_positions_pnl()
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'status': f'Error updating data: {str(e)}'})

if __name__ == '__main__':
    print("🚀 Starting Qwen3 Trading Bot - AUTO MODE")
    print("📍 Dashboard available at: http://localhost:5000")
    print("🧠 Profile: Qwen3 (Fixed - No Switching)")
    print("💰 Data Sources: Binance → KuCoin → CoinGecko")
    app.run(host='0.0.0.0', port=5000, debug=True)
