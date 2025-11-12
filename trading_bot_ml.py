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
import random
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_trading_bot.log', encoding='utf-8')
    ]
)

class LLMTradingBot:
    def __init__(self, initial_capital=10000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # API Binance
        self.binance_base_url = "https://api.binance.com/api/v3"
        
        # Cache cen
        self.price_cache = {}
        self.price_history = {}
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM (wg Alpha Arena) - ZMODYFIKOWANE
        self.llm_profiles = {
            'Claude': {
                'risk_appetite': 'MEDIUM',
                'confidence_bias': 0.6,
                'short_frequency': 0.1,
                'holding_bias': 'LONG',
                'trade_frequency': 'LOW',
                'position_sizing': 'CONSERVATIVE',
                'min_holding_hours': 2,
                'max_holding_hours': 8,
                'tp_multiplier': 1.0,
                'sl_multiplier': 1.0,
                'confidence_threshold': 0.4
            },
            'Gemini': {
                'risk_appetite': 'HIGH', 
                'confidence_bias': 0.7,
                'short_frequency': 0.35,
                'holding_bias': 'SHORT',
                'trade_frequency': 'HIGH',
                'position_sizing': 'AGGRESSIVE',
                'min_holding_hours': 1,
                'max_holding_hours': 6,
                'tp_multiplier': 1.2,
                'sl_multiplier': 0.9,
                'confidence_threshold': 0.3
            },
            'GPT': {
                'risk_appetite': 'LOW',
                'confidence_bias': 0.3,
                'short_frequency': 0.4,
                'holding_bias': 'NEUTRAL',
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'CONSERVATIVE',
                'min_holding_hours': 2,
                'max_holding_hours': 10,
                'tp_multiplier': 0.8,
                'sl_multiplier': 1.1,
                'confidence_threshold': 0.5
            },
            'Qwen': {
                'risk_appetite': 'HIGH',
                'confidence_bias': 0.85,
                'short_frequency': 0.2,
                'holding_bias': 'LONG', 
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'VERY_AGGRESSIVE',
                'min_holding_hours': 4,        # Wydłużone minimum
                'max_holding_hours': 24,       # Wydłużone maksimum
                'tp_multiplier': 1.3,          # Szersze TP
                'sl_multiplier': 1.2,          # Szersze SL
                'confidence_threshold': 0.4,   # Niższy próg wejścia
                'use_tiered_exits': True,      # System warstwowy
                'use_trailing_stop': True,     # Trailing stop
                'use_volatility_based': True   # Bazowanie na ATR
            }
        }
        
        # AKTYWNY PROFIL
        self.active_profile = 'Qwen'
        
        # PARAMETRY OPERACYJNE
        self.max_simultaneous_positions = 4
        self.assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT']
        
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
            'active_profile': self.active_profile
        }
        
        # Dane wykresu
        self.chart_data = {
            'labels': [],
            'values': []
        }
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - Alpha Arena Inspired")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")
        self.logger.info(f"📈 Trading assets: {', '.join(self.assets)}")

    def get_binance_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę z API Binance - JEDYNE ŹRÓDŁO CEN"""
        try:
            url = f"{self.binance_base_url}/ticker/price"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price = float(data['price'])
            
            # Zapisz w cache
            self.price_cache[symbol] = {
                'price': price,
                'timestamp': datetime.now()
            }
            
            # Zapisz w historii dla analizy
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'price': price,
                'timestamp': datetime.now()
            })
            
            # Ogranicz historię do ostatnich 50 punktów
            if len(self.price_history[symbol]) > 50:
                self.price_history[symbol] = self.price_history[symbol][-50:]
            
            return price
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ API Error getting price for {symbol}: {e}")
            if symbol in self.price_cache:
                cache_age = (datetime.now() - self.price_cache[symbol]['timestamp']).total_seconds()
                if cache_age < 300:  # 5 minut
                    self.logger.info(f"🔄 Using cached price for {symbol} (age: {cache_age:.1f}s)")
                    return self.price_cache[symbol]['price']
            
            self.logger.warning(f"⚠️ Could not get price for {symbol} and no recent cache")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error getting price for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Pobiera aktualną cenę - WYŁĄCZNIE Z API BINANCE"""
        return self.get_binance_price(symbol)

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Oblicza Average True Range dla danego symbolu"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < period + 1:
                return 0.02  # fallback 2%
            
            prices = [entry['price'] for entry in self.price_history[symbol]]
            true_ranges = []
            
            for i in range(1, len(prices)):
                high = max(prices[i], prices[i-1])
                low = min(prices[i], prices[i-1])
                true_range = high - low
                true_ranges.append(true_range)
            
            # Weź ostatnie N true ranges
            recent_true_ranges = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
            atr = np.mean(recent_true_ranges) if recent_true_ranges else 0
            
            # Normalizuj do procentów
            current_price = prices[-1] if prices else 1
            atr_percent = atr / current_price if current_price > 0 else 0.02
            
            return max(min(atr_percent, 0.1), 0.005)  # Limit 0.5% - 10%
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ATR for {symbol}: {e}")
            return 0.02

    def analyze_simple_momentum(self, symbol: str) -> float:
        """Analiza momentum na podstawie rzeczywistych danych z API Binance"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                return random.uniform(-0.02, 0.02)
            
            history = self.price_history[symbol]
            current_price = history[-1]['price']
            
            # Oblicz momentum na podstawie ostatnich punktów
            lookback = min(5, len(history) - 1)
            past_price = history[-lookback]['price']
            
            momentum = (current_price - past_price) / past_price
            
            # Normalizuj momentum
            momentum = max(min(momentum, 0.03), -0.03)
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing momentum for {symbol}: {e}")
            return random.uniform(-0.02, 0.02)

    def check_volume_activity(self, symbol: str) -> bool:
        """Sprawdza aktywność wolumenu na podstawie zmienności cen z API Binance"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return random.random() < 0.6
            
            # Oblicz zmienność na podstawie rzeczywistej historii cen
            prices = [entry['price'] for entry in self.price_history[symbol][-10:]]
            volatility = np.std(prices) / np.mean(prices)
            
            # Wyższa zmienność = wyższa aktywność
            return volatility > 0.002
            
        except Exception as e:
            self.logger.error(f"❌ Error checking volume activity for {symbol}: {e}")
            return random.random() < 0.6

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał w stylu LLM na podstawie rzeczywistych danych z API Binance"""
        profile = self.get_current_profile()
        
        # Podstawowe obserwacje na podstawie rzeczywistych cen
        momentum = self.analyze_simple_momentum(symbol)
        volume_active = self.check_volume_activity(symbol)
        
        # Confidence bazowe z profilu
        base_confidence = profile['confidence_bias']
        
        # Modyfikatory confidence na podstawie rzeczywistych danych
        confidence_modifiers = 0
        
        if momentum > 0.008:  # Silny pozytywny momentum
            confidence_modifiers += 0.2
        elif momentum > 0.003:  # Umiarkowany pozytywny momentum
            confidence_modifiers += 0.1
        elif momentum < -0.008:  # Silny negatywny momentum
            confidence_modifiers += 0.15
        elif momentum < -0.003:  # Umiarkowany negatywny momentum
            confidence_modifiers += 0.08
            
        if volume_active:
            confidence_modifiers += 0.1
            
        # Final confidence z losowością
        final_confidence = min(base_confidence + confidence_modifiers + random.uniform(-0.1, 0.1), 0.95)
        final_confidence = max(final_confidence, 0.1)
        
        # Decyzja o kierunku na podstawie rzeczywistego momentum
        if momentum > 0.01 and volume_active:
            signal = "LONG"
        elif momentum < -0.01 and volume_active:
            if random.random() < profile['short_frequency']:
                signal = "SHORT"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
            
        current_price = self.get_current_price(symbol)
        price_display = f"${current_price:.4f}" if current_price else "N/A"
        self.logger.info(f"🎯 {self.active_profile} SIGNAL: {symbol} -> {signal} (Price: {price_display}, Conf: {final_confidence:.1%}, Mom: {momentum:.2%})")
        
        return signal, final_confidence

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji w stylu LLM - ZMODYFIKOWANE DLA QWEN"""
        profile = self.get_current_profile()
        
        base_allocation = {
            'Claude': 0.15,
            'Gemini': 0.25, 
            'GPT': 0.10,
            'Qwen': 0.40  # Zwiększone z 0.30
        }.get(self.active_profile, 0.15)
        
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        sizing_multiplier = {
            'CONSERVATIVE': 0.8,
            'AGGRESSIVE': 1.2,
            'VERY_AGGRESSIVE': 1.5
        }.get(profile['position_sizing'], 1.0)
        
        position_value = (self.virtual_capital * base_allocation * 
                         confidence_multiplier * sizing_multiplier)
        
        max_position_value = self.virtual_capital * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def calculate_volatility_based_exits(self, symbol: str, entry_price: float, side: str, confidence: float) -> Dict:
        """Oblicza TP/SL bazujące na zmienności (ATR)"""
        profile = self.get_current_profile()
        atr_percent = self.calculate_atr(symbol)
        
        # Domyślne multiplikatory
        if self.active_profile == 'Qwen':
            if confidence > 0.8:
                tp_multiplier = 2.5 * profile['tp_multiplier']
                sl_multiplier = 1.0 * profile['sl_multiplier']
            elif confidence > 0.6:
                tp_multiplier = 2.0 * profile['tp_multiplier']
                sl_multiplier = 1.2 * profile['sl_multiplier']
            else:
                tp_multiplier = 1.5 * profile['tp_multiplier']
                sl_multiplier = 1.5 * profile['sl_multiplier']
        else:
            tp_multiplier = profile['tp_multiplier']
            sl_multiplier = profile['sl_multiplier']
        
        if side == "LONG":
            take_profit = entry_price * (1 + atr_percent * tp_multiplier)
            stop_loss = entry_price * (1 - atr_percent * sl_multiplier)
        else:
            take_profit = entry_price * (1 - atr_percent * tp_multiplier)
            stop_loss = entry_price * (1 + atr_percent * sl_multiplier)
        
        return take_profit, stop_loss

    def calculate_tiered_exit_plan(self, entry_price: float, side: str, confidence: float) -> List[Dict]:
        """System warstwowych zysków dla agresywnego Qwen"""
        profile = self.get_current_profile()
        
        if self.active_profile == 'Qwen' and profile.get('use_tiered_exits', False):
            if confidence > 0.8:
                tiers = [
                    {'percent': 0.3, 'tp_pct': 0.010},  # 30% pozycji przy 1%
                    {'percent': 0.4, 'tp_pct': 0.018},  # 40% przy 1.8%  
                    {'percent': 0.3, 'tp_pct': 0.025}   # 30% przy 2.5%
                ]
            elif confidence > 0.6:
                tiers = [
                    {'percent': 0.5, 'tp_pct': 0.008},  # 50% przy 0.8%
                    {'percent': 0.5, 'tp_pct': 0.015}   # 50% przy 1.5%
                ]
            else:
                tiers = [
                    {'percent': 0.7, 'tp_pct': 0.006},  # 70% przy 0.6%
                    {'percent': 0.3, 'tp_pct': 0.012}   # 30% przy 1.2%
                ]
            
            # Konwersja na ceny
            partial_exits = []
            for tier in tiers:
                if side == "LONG":
                    tp_price = entry_price * (1 + tier['tp_pct'])
                else:
                    tp_price = entry_price * (1 - tier['tp_pct'])
                partial_exits.append({
                    'price': round(tp_price, 4),
                    'percent': tier['percent']
                })
            
            return partial_exits
        else:
            # Dla innych profili - brak partial exits
            return []

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """Oblicza plan wyjścia w stylu LLM - ZMODYFIKOWANE DLA QWEN"""
        profile = self.get_current_profile()
        
        # SPECJALNE TRAJTOWANIE QWEN - volatility based exits
        if self.active_profile == 'Qwen' and profile.get('use_volatility_based', True):
            take_profit, stop_loss = self.calculate_volatility_based_exits(
                'BTCUSDT', entry_price, side, confidence  # Używamy BTC jako proxy
            )
        else:
            # Standardowe obliczenia dla innych profili
            if confidence > 0.7:
                if side == "LONG":
                    take_profit = entry_price * 1.018
                    stop_loss = entry_price * 0.992
                else:
                    take_profit = entry_price * 0.982
                    stop_loss = entry_price * 1.008
            elif confidence > 0.5:
                if side == "LONG":
                    take_profit = entry_price * 1.012
                    stop_loss = entry_price * 0.994
                else:
                    take_profit = entry_price * 0.988
                    stop_loss = entry_price * 1.006
            else:
                if side == "LONG":
                    take_profit = entry_price * 1.008
                    stop_loss = entry_price * 0.996
                else:
                    take_profit = entry_price * 0.992
                    stop_loss = entry_price * 1.004
        
        # Zastosuj multiplikatory profilu
        take_profit = entry_price + (take_profit - entry_price) * profile['tp_multiplier']
        stop_loss = entry_price + (stop_loss - entry_price) * profile['sl_multiplier']
        
        # Oblicz partial exits dla Qwen
        partial_exits = self.calculate_tiered_exit_plan(entry_price, side, confidence)
        
        exit_plan = {
            'take_profit': round(take_profit, 4),
            'stop_loss': round(stop_loss, 4),
            'invalidation': entry_price * 0.98 if side == "LONG" else entry_price * 1.02,
            'max_holding_hours': random.randint(profile['min_holding_hours'], profile['max_holding_hours']),
            'partial_exits': partial_exits,
            'use_trailing_stop': profile.get('use_trailing_stop', False),
            'trailing_start': 0.008 if self.active_profile == 'Qwen' else 0.012,
            'trailing_step': 0.003 if self.active_profile == 'Qwen' else 0.005,
            'original_sl': None  # Do trailing stop
        }
        
        return exit_plan

    def should_enter_trade(self) -> bool:
        """Decyduje czy wejść w transakcję wg profilu częstotliwości - ZMODYFIKOWANE"""
        profile = self.get_current_profile()
        
        frequency_chance = {
            'LOW': 0.2,        # Zmniejszone
            'MEDIUM': 0.3,     # Zmniejszone  
            'HIGH': 0.8        # Zmniejszone
        }.get(profile['trade_frequency'], 0.3)
        
        # DODATKOWY FILTR DLA QWEN - mniej, ale większe pozycje
        if self.active_profile == 'Qwen' and len([p for p in self.positions.values() if p['status'] == 'ACTIVE']) >= 2:
            return False  # Qwen powinien trzymać 1-2 pozycje
        
        return random.random() < frequency_chance

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję w stylu LLM używając rzeczywistych cen z API"""
        if not self.should_enter_trade():
            return None
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.warning(f"❌ Could not get price for {symbol} - skipping trade")
            return None
            
        signal, confidence = self.generate_llm_signal(symbol)
        
        # Sprawdź próg confidence dla profilu
        profile = self.get_current_profile()
        if signal == "HOLD" or confidence < profile['confidence_threshold']:
            return None
            
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
            
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
            
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        if signal == "LONG":
            liquidation_price = current_price * (1 - 0.9 / self.leverage)
        else:
            liquidation_price = current_price * (1 + 0.9 / self.leverage)
        
        position_id = f"llm_{self.position_id}"
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
            'llm_profile': self.active_profile,
            'exit_plan': exit_plan,
            'partial_exits_taken': []  # Śledzenie wykonanych partial exits
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        if signal == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        tp_distance = (exit_plan['take_profit'] - current_price) / current_price * 100
        sl_distance = (current_price - exit_plan['stop_loss']) / current_price * 100
        
        self.logger.info(f"🎯 {self.active_profile} OPEN: {symbol} {signal} @ ${current_price:.4f}")
        self.logger.info(f"   📊 Confidence: {confidence:.1%} | Size: ${position_value:.2f}")
        self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} ({tp_distance:+.2f}%)")
        self.logger.info(f"   🛑 SL: {exit_plan['stop_loss']:.4f} ({sl_distance:+.2f}%)")
        
        if exit_plan['partial_exits']:
            self.logger.info(f"   📈 Partial exits: {len(exit_plan['partial_exits'])} tiers")
        
        return position_id

    def update_trailing_stop(self, position_id: str, current_price: float):
        """Aktualizuje trailing stop dla pozycji"""
        position = self.positions[position_id]
        exit_plan = position['exit_plan']
        
        if not exit_plan.get('use_trailing_stop', False):
            return
        
        unrealized_pnl_pct = abs(current_price - position['entry_price']) / position['entry_price']
        
        # Sprawdź czy osiągnięto poziom startu trailing
        if unrealized_pnl_pct >= exit_plan['trailing_start']:
            if exit_plan['original_sl'] is None:
                exit_plan['original_sl'] = exit_plan['stop_loss']
            
            # Oblicz nowy stop loss
            if position['side'] == "LONG":
                new_sl = current_price * (1 - exit_plan['trailing_step'])
                # Podnieś SL tylko jeśli wyższy niż obecny
                if new_sl > exit_plan['stop_loss']:
                    exit_plan['stop_loss'] = new_sl
            else:
                new_sl = current_price * (1 + exit_plan['trailing_step'])
                # Obniż SL tylko jeśli niższy niż obecny
                if new_sl < exit_plan['stop_loss']:
                    exit_plan['stop_loss'] = new_sl

    def check_partial_exits(self, position_id: str, current_price: float) -> bool:
        """Sprawdza warunki partial take profits"""
        position = self.positions[position_id]
        exit_plan = position['exit_plan']
        
        if not exit_plan['partial_exits']:
            return False
        
        for partial_exit in exit_plan['partial_exits']:
            if partial_exit['price'] in position['partial_exits_taken']:
                continue
                
            if position['side'] == "LONG" and current_price >= partial_exit['price']:
                return self.execute_partial_exit(position_id, partial_exit)
            elif position['side'] == "SHORT" and current_price <= partial_exit['price']:
                return self.execute_partial_exit(position_id, partial_exit)
        
        return False

    def execute_partial_exit(self, position_id: str, partial_exit: Dict) -> bool:
        """Wykonuje partial exit z pozycji"""
        position = self.positions[position_id]
        
        # Oblicz ilość do zamknięcia
        close_quantity = position['quantity'] * partial_exit['percent']
        close_value = close_quantity * position['entry_price'] * position['leverage']
        
        # Oblicz P&L dla partial exit
        current_price = self.get_current_price(position['symbol'])
        if position['side'] == "LONG":
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
        
        realized_pnl = pnl_pct * close_quantity * position['entry_price'] * position['leverage']
        fee = abs(realized_pnl) * 0.001
        realized_pnl_after_fee = realized_pnl - fee
        
        # Aktualizuj pozycję
        position['quantity'] -= close_quantity
        position['margin'] *= (1 - partial_exit['percent'])  # Zmniejsz margin proporcjonalnie
        
        # Zwróć margin i P&L
        returned_margin = position['margin'] * partial_exit['percent']
        self.virtual_balance += returned_margin + realized_pnl_after_fee
        self.virtual_capital += realized_pnl_after_fee
        
        # Zapisz partial exit
        position['partial_exits_taken'].append(partial_exit['price'])
        
        self.logger.info(f"🟡 PARTIAL EXIT: {position['symbol']} - {partial_exit['percent']:.0%} @ ${current_price:.4f} | P&L: ${realized_pnl_after_fee:+.2f}")
        
        return True

    def update_positions_pnl(self):
        """Aktualizuje P&L wszystkich pozycji używając rzeczywistych cen z API"""
        total_unrealized = 0
        total_margin = 0
        total_confidence = 0
        confidence_count = 0
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
                
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
                
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
            # Aktualizuj trailing stop
            self.update_trailing_stop(position_id, current_price)
            
            # Sprawdź partial exits
            self.check_partial_exits(position_id, current_price)
            
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
        """Sprawdza warunki wyjścia z pozycji używając rzeczywistych cen z API"""
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
                
            current_price = position.get('current_price', self.get_current_price(position['symbol']))
            if not current_price:
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
        """Zamyka pozycję"""
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
            'holding_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600,
            'partial_exits_taken': len(position['partial_exits_taken'])
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
        """Oblicza dywersyfikację portfela"""
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

    def get_current_profile(self):
        """Zwraca aktywny profil LLM"""
        return self.llm_profiles[self.active_profile]

    def set_active_profile(self, profile_name: str):
        """Zmienia aktywny profil zachowania"""
        if profile_name in self.llm_profiles:
            self.active_profile = profile_name
            self.dashboard_data['active_profile'] = profile_name
            self.logger.info(f"🔄 Changed LLM profile to: {profile_name}")
            return True
        return False

    def get_dashboard_data(self):
        """Przygotowuje dane dla dashboardu używając rzeczywistych cen z API"""
        active_positions = []
        total_unrealized_pnl = 0
        
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price', self.get_current_price(position['symbol']))
                if not current_price:
                    continue
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
                # Oblicz odległości do TP/SL
                if position['side'] == 'LONG':
                    tp_distance_pct = ((position['exit_plan']['take_profit'] - current_price) / current_price) * 100
                    sl_distance_pct = ((current_price - position['exit_plan']['stop_loss']) / current_price) * 100
                else:
                    tp_distance_pct = ((current_price - position['exit_plan']['take_profit']) / current_price) * 100
                    sl_distance_pct = ((position['exit_plan']['stop_loss'] - current_price) / current_price) * 100
                
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
                    'tp_distance_pct': tp_distance_pct,
                    'sl_distance_pct': sl_distance_pct,
                    'partial_exits_taken': len(position['partial_exits_taken'])
                })
                
                total_unrealized_pnl += unrealized_pnl
        
        # Oblicz confidence levels dla każdego assetu
        confidence_levels = {}
        for symbol in self.assets:
            try:
                signal, confidence = self.generate_llm_signal(symbol)
                confidence_levels[symbol] = round(confidence * 100, 1)
            except:
                confidence_levels[symbol] = 0
        
        # Ostatnie transakcje
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
                'exit_time': trade['exit_time'].strftime('%H:%M:%S'),
                'partial_exits': trade.get('partial_exits_taken', 0)
            })
        
        # Metryki wydajności
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
                'active_profile': self.active_profile,
                'available_profiles': list(self.llm_profiles.keys()),
                'max_positions': self.max_simultaneous_positions,
                'leverage': self.leverage
            },
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': total_unrealized_pnl,
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def save_chart_data(self, chart_data: Dict):
        """Zapisuje dane wykresu"""
        try:
            self.chart_data = chart_data
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving chart data: {e}")
            return False

    def load_chart_data(self) -> Dict:
        """Ładuje dane wykresu"""
        return self.chart_data

    def run_llm_trading_strategy(self):
        """Główna pętla strategii LLM używająca rzeczywistych cen z API"""
        self.logger.info("🚀 STARTING LLM-STYLE TRADING STRATEGY")
        self.logger.info(f"🎯 Active Profile: {self.active_profile}")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 LLM Trading Iteration #{iteration}")
                
                # 1. Aktualizuj P&L używając rzeczywistych cen
                self.update_positions_pnl()
                
                # 2. Sprawdź warunki wyjścia
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Sprawdź możliwości wejścia
                active_symbols = [p['symbol'] for p in self.positions.values() 
                                if p['status'] == 'ACTIVE']
                active_count = len(active_symbols)
                
                if active_count < self.max_simultaneous_positions:
                    for symbol in self.assets:
                        if symbol not in active_symbols:
                            position_id = self.open_llm_position(symbol)
                            if position_id:
                                time.sleep(1)
                
                portfolio_value = self.dashboard_data['account_value']
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active Positions: {active_count}/{self.max_simultaneous_positions}")
                
                wait_time = random.randint(30, 90)
                for i in range(wait_time):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in LLM trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Rozpoczyna trading"""
        self.is_running = True
        threading.Thread(target=self.run_llm_trading_strategy, daemon=True).start()
        self.logger.info("🚀 LLM Trading Bot started")

    def stop_trading(self):
        """Zatrzymuje trading"""
        self.is_running = False
        self.logger.info("🛑 LLM Trading Bot stopped")


# FLASK APP
app = Flask(__name__)
CORS(app)

# Inicjalizacja bota
trading_bot = LLMTradingBot(initial_capital=10000, leverage=10)

# Routes do renderowania stron
@app.route('/')
def index():
    """Strona główna - renderuje template index.html"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard - również renderuje index.html"""
    return render_template('index.html')

# API endpoints
@app.route('/api/trading-data')
def get_trading_data():
    """Zwraca dane tradingowe dla dashboardu"""
    try:
        data = trading_bot.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot-status')
def get_bot_status():
    """Zwraca status bota"""
    status = 'running' if trading_bot.is_running else 'stopped'
    return jsonify({'status': status})

@app.route('/api/start-bot', methods=['POST'])
def start_bot():
    """Uruchamia bota"""
    try:
        trading_bot.start_trading()
        return jsonify({'status': 'Bot started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-bot', methods=['POST'])
def stop_bot():
    """Zatrzymuje bota"""
    try:
        trading_bot.stop_trading()
        return jsonify({'status': 'Bot stopped successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/change-profile', methods=['POST'])
def change_profile():
    """Zmienia profil LLM"""
    try:
        data = request.get_json()
        profile_name = data.get('profile')
        
        if trading_bot.set_active_profile(profile_name):
            return jsonify({'status': f'Profile changed to {profile_name}'})
        else:
            return jsonify({'error': 'Invalid profile name'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/force-update', methods=['POST'])
def force_update():
    """Wymusza aktualizację danych"""
    try:
        trading_bot.update_positions_pnl()
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-chart-data', methods=['POST'])
def save_chart_data():
    """Zapisuje dane wykresu"""
    try:
        data = request.get_json()
        if trading_bot.save_chart_data(data):
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Failed to save chart data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-chart-data')
def load_chart_data():
    """Ładuje dane wykresu"""
    try:
        chart_data = trading_bot.load_chart_data()
        return jsonify({
            'status': 'success',
            'chartData': chart_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting LLM Trading Bot Server...")
    print("📍 Dashboard available at: http://localhost:5000")
    print("🧠 LLM Profiles: Claude, Gemini, GPT, Qwen")
    print("📈 Trading assets: BTC, ETH, SOL, XRP, BNB, DOGE")
    print("💹 Using REAL-TIME prices from Binance API only")
    print("🎯 Qwen Profile Features: Extended holding periods, Tiered exits, Volatility-based TP/SL")
    app.run(debug=True, host='0.0.0.0', port=5000)
