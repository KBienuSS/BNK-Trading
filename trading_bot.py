# trading_bot.py
import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import threading

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_live.log', encoding='utf-8')
    ]
)

class ExactStrategyBot:
    def __init__(self, initial_capital=10000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Parametry tradingowe
        self.max_simultaneous_positions = 6
        self.position_sizes = {
            'BTCUSDT': 0.00711,
            'ETHUSDT': 3.2968, 
            'BNBUSDT': 7.036,
            'XRPUSDT': 1737.0,
            'DOGEUSDT': 18915.0,
            'SOLUSDT': 20.76,
            'ADAUSDT': 5000.0,
            'DOTUSDT': 200.0
        }
        
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'SOLUSDT']
        
        # Statystyki
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'biggest_win': 0,
            'biggest_loss': 0
        }
        
        # Dane dashboardu
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'total_fees': 0,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'average_leverage': 10,
            'average_confidence': 0,
            'account_history': [initial_capital],
            'last_update': datetime.now()
        }
        
        self.logger.info("🎯 LIVE TRADING BOT - Real-time mode")
        self.logger.info(f"💰 Initial capital: ${initial_capital}")

    def calculate_sma(self, data, window):
        """Oblicza Simple Moving Average"""
        return data.rolling(window=window).mean()

    def calculate_rsi(self, prices, window=14):
        """Oblicza RSI - prosty sposób"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_binance_klines(self, symbol: str, interval: str = '3m', limit: int = 100):
        """Pobiera dane z Binance"""
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
            
            # Konwersja do float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str):
        """Pobiera aktualną cenę z Binance"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            price = float(data['price'])
            return price
        except Exception as e:
            self.logger.error(f"❌ Error fetching price for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame):
        """Oblicza wskaźniki techniczne - własna implementacja"""
        try:
            # SMA
            df['sma_20'] = self.calculate_sma(df['close'], 20)
            df['sma_50'] = self.calculate_sma(df['close'], 50)
            
            # RSI
            df['rsi_14'] = self.calculate_rsi(df['close'], 14)
            
            # Volume
            df['volume_ma_20'] = self.calculate_sma(df['volume'], 20)
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error calculating indicators: {e}")
            return df

    def generate_signal(self, symbol: str):
        """Generuje sygnał tradingowy"""
        df = self.get_binance_klines(symbol, '3m', 100)
        if df is None or len(df) < 50:
            return "HOLD", 0
        
        df = self.calculate_technical_indicators(df)
        
        try:
            if df['rsi_14'].iloc[-1] is None or pd.isna(df['rsi_14'].iloc[-1]):
                return "HOLD", 0
                
            current_rsi = df['rsi_14'].iloc[-1]
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1] if not pd.isna(df['volume_ratio'].iloc[-1]) else 1

            # Prosta strategia
            confidence = 0.0
            signal = "HOLD"
            
            # Warunki dla LONG
            if (current_rsi < 65 and 
                volume_ratio > 1.2 and
                current_price > sma_20 and
                current_rsi > 30):
                
                signal = "LONG"
                confidence = min(0.75 + (volume_ratio - 1.2) * 0.1, 0.95)
                self.logger.info(f"🎯 LONG signal for {symbol} - Confidence: {confidence:.1%}")
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return "HOLD", 0

    def calculate_position_size(self, symbol: str, price: float):
        """Oblicza wielkość pozycji"""
        base_quantity = self.position_sizes.get(symbol, 1000.0)
        position_value = base_quantity * price
        margin_required = position_value / self.leverage
        return base_quantity, position_value, margin_required

    def open_position(self, symbol: str, side: str):
        """Otwiera pozycję"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        signal, confidence = self.generate_signal(symbol)
        if signal != "LONG" or confidence < 0.65:
            return None
        
        # Sprawdź limit pozycji
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
        
        quantity, position_value, margin_required = self.calculate_position_size(symbol, current_price)
        
        if margin_required > self.virtual_balance:
            return None
        
        position_id = f"pos_{self.position_id}"
        self.position_id += 1
        
        liquidation_price = current_price * (1 - 0.9 / self.leverage)
        
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
            'unrealized_pnl_pct': 0,
            'current_price': current_price,
            'exit_plan': {
                'take_profit': current_price * 1.05,
                'stop_loss': current_price * 0.97,
                'invalidation': current_price * 0.95
            }
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        self.logger.info(f"🎯 OPEN POSITION: {side} {quantity} {symbol} @ ${current_price:.2f}")
        return position_id

    def update_positions_pnl(self):
        """Aktualizuje P&L pozycji"""
        total_unrealized = 0
        total_margin = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            position['current_price'] = current_price
            
            # Oblicz P&L
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            
            position['unrealized_pnl'] = unrealized_pnl
            position['unrealized_pnl_pct'] = pnl_pct * 100
            total_unrealized += unrealized_pnl
            total_margin += position['margin']
        
        # Aktualizuj dashboard
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        
        # Oblicz średni leverage
        if total_margin > 0:
            total_position_value = sum(p['quantity'] * p['entry_price'] for p in self.positions.values() if p['status'] == 'ACTIVE')
            self.dashboard_data['average_leverage'] = round(total_position_value / (total_margin + 0.001), 1)
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia"""
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            exit_reason = None
            
            if position['side'] == 'LONG':
                if current_price >= position['exit_plan']['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= position['exit_plan']['stop_loss']:
                    exit_reason = "STOP_LOSS"
                elif current_price <= position['liquidation_price']:
                    exit_reason = "LIQUIDATION"
            
            if exit_reason:
                positions_to_close.append((position_id, exit_reason, current_price))
        
        return positions_to_close

    def close_position(self, position_id: str, exit_reason: str, exit_price: float):
        """Zamyka pozycję"""
        position = self.positions[position_id]
        
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        realized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
        
        # Symulacja opłat
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
            'pnl_pct': pnl_pct * 100,
            'exit_reason': exit_reason,
            'fee': fee,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now()
        }
        
        self.trade_history.append(trade_record)
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        self.stats['total_fees'] += fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
            if realized_pnl_after_fee > self.stats['biggest_win']:
                self.stats['biggest_win'] = realized_pnl_after_fee
        else:
            self.stats['losing_trades'] += 1
            if realized_pnl_after_fee < self.stats['biggest_loss']:
                self.stats['biggest_loss'] = realized_pnl_after_fee
        
        position['status'] = 'CLOSED'
        
        # Aktualizuj dashboard
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        self.dashboard_data['total_fees'] = self.stats['total_fees']
        
        pnl_color = "🟢" if realized_pnl_after_fee > 0 else "🔴"
        self.logger.info(f"{pnl_color} CLOSED: {position['symbol']} - P&L: ${realized_pnl_after_fee:+.2f}")

    def get_dashboard_data(self):
        """Przygotowuje dane dla dashboardu"""
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
                    'symbol': position['symbol']
                })
        
        # Oblicz średnią confidence
        for symbol in self.symbols:
            signal, confidence = self.generate_signal(symbol)
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = round((total_confidence / confidence_count) * 100, 1)
        
        # Ostatnie transakcje
        recent_trades = []
        for trade in self.trade_history[-25:]:
            recent_trades.append({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'quantity': trade['quantity'],
                'realized_pnl': trade['realized_pnl'],
                'exit_time': trade['exit_time'].strftime('%H:%M:%S')
            })
        
        # Aktualizuj historię konta
        if len(self.dashboard_data['account_history']) > 50:
            self.dashboard_data['account_history'].pop(0)
        self.dashboard_data['account_history'].append(self.dashboard_data['account_value'])
        
        return {
            'account_summary': {
                'total_value': round(self.dashboard_data['account_value'], 2),
                'available_cash': round(self.dashboard_data['available_cash'], 2),
                'total_fees': round(self.dashboard_data['total_fees'], 2),
                'net_realized': round(self.dashboard_data['net_realized'], 2)
            },
            'performance_metrics': {
                'avg_leverage': self.dashboard_data['average_leverage'],
                'avg_confidence': self.dashboard_data['average_confidence'],
                'biggest_win': round(self.stats['biggest_win'], 2),
                'biggest_loss': round(self.stats['biggest_loss'], 2)
            },
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
            'account_history': [round(x, 2) for x in self.dashboard_data['account_history']],
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_exact_strategy(self):
        """Główna pętla strategii"""
        self.logger.info("🚀 STARTING LIVE TRADING STRATEGY...")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                self.logger.info(f"\n🔄 Iteration #{iteration} | {current_time}")
                
                # 1. Aktualizuj P&L
                self.update_positions_pnl()
                
                # 2. Sprawdź wyjścia
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Sprawdź sygnały wejścia
                active_symbols = [p['symbol'] for p in self.positions.values() if p['status'] == 'ACTIVE']
                
                for symbol in self.symbols:
                    if symbol not in active_symbols:
                        signal, confidence = self.generate_signal(symbol)
                        
                        if signal == "LONG" and confidence >= 0.65:
                            self.logger.info(f"🎯 STRONG SIGNAL: {symbol} - Confidence: {confidence:.1%}")
                            self.open_position(symbol, "LONG")
                
                # 4. Loguj status
                active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
                self.logger.info(f"📊 Status: {active_count} active positions | Account: ${self.dashboard_data['account_value']:.2f}")
                
                # 5. Czekaj 60 sekund
                for i in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(30)

# Globalna instancja bota
trading_bot = ExactStrategyBot(initial_capital=10000, leverage=10)