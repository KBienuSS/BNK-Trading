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
from flask import Flask, jsonify, request
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
        
        # PROFIL ZACHOWANIA INSPIROWANY LLM (wg Alpha Arena)
        self.llm_profiles = {
            'Claude': {
                'risk_appetite': 'MEDIUM',
                'confidence_bias': 0.6,
                'short_frequency': 0.1,  # Rzadko shortuje
                'holding_bias': 'LONG',
                'trade_frequency': 'LOW',
                'position_sizing': 'CONSERVATIVE'
            },
            'Gemini': {
                'risk_appetite': 'HIGH', 
                'confidence_bias': 0.7,
                'short_frequency': 0.35,
                'holding_bias': 'SHORT',
                'trade_frequency': 'HIGH',
                'position_sizing': 'AGGRESSIVE'
            },
            'GPT': {
                'risk_appetite': 'LOW',
                'confidence_bias': 0.3,
                'short_frequency': 0.4,
                'holding_bias': 'NEUTRAL',
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'CONSERVATIVE'
            },
            'Qwen': {
                'risk_appetite': 'HIGH',
                'confidence_bias': 0.85,
                'short_frequency': 0.2,
                'holding_bias': 'LONG', 
                'trade_frequency': 'MEDIUM',
                'position_sizing': 'VERY_AGGRESSIVE'
            }
        }
        
        # AKTYWNY PROFIL (można zmieniać)
        self.active_profile = 'Claude'
        
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
            'active_profile': self.active_profile
        }
        
        self.logger.info("🧠 LLM-STYLE TRADING BOT - Alpha Arena Inspired")
        self.logger.info(f"💰 Initial capital: ${initial_capital} | Leverage: {leverage}x")
        self.logger.info(f"🎯 Active LLM Profile: {self.active_profile}")

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

    def get_current_price(self, symbol: str):
        """Pobiera aktualną cenę - uproszczona wersja"""
        try:
            # Symulacja ceny z lekkim szumem
            base_prices = {
                'BTCUSDT': 112614,
                'ETHUSDT': 3485, 
                'SOLUSDT': 178,
                'XRPUSDT': 0.615,
                'BNBUSDT': 582
            }
            
            base_price = base_prices.get(symbol, 100)
            volatility = random.uniform(-0.002, 0.002)  # 0.2% szumu
            current_price = base_price * (1 + volatility)
            
            return round(current_price, 4)
            
        except Exception as e:
            self.logger.error(f"❌ Error getting price for {symbol}: {e}")
            return base_prices.get(symbol, 100)

    def analyze_simple_momentum(self, symbol: str) -> float:
        """Prosta analiza momentum - zamiast skomplikowanych wskaźników"""
        try:
            # Symulacja prostych danych historycznych
            current_price = self.get_current_price(symbol)
            
            # Losowy momentum w zakresie -3% do +3%
            momentum = random.uniform(-0.03, 0.03)
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing momentum for {symbol}: {e}")
            return 0.0

    def check_volume_activity(self, symbol: str) -> bool:
        """Prosty check aktywności wolumenu"""
        # 70% szans na "aktywny" volume
        return random.random() < 0.7

    def generate_llm_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał w stylu LLM - proste reguły + element losowości"""
        profile = self.get_current_profile()
        
        # Podstawowe obserwacje
        momentum = self.analyze_simple_momentum(symbol)
        volume_active = self.check_volume_activity(symbol)
        
        # Confidence bazowe z profilu
        base_confidence = profile['confidence_bias']
        
        # Modyfikatory confidence
        confidence_modifiers = 0
        
        if momentum > 0.01:  # Pozytywny momentum
            confidence_modifiers += 0.15
        elif momentum < -0.01:  # Negatywny momentum  
            confidence_modifiers += 0.1
            
        if volume_active:
            confidence_modifiers += 0.1
            
        # Final confidence z losowością
        final_confidence = min(base_confidence + confidence_modifiers + random.uniform(-0.1, 0.1), 0.95)
        final_confidence = max(final_confidence, 0.1)  # Minimum 10%
        
        # Decyzja o kierunku
        if momentum > 0.015 and volume_active:
            signal = "LONG"
        elif momentum < -0.015 and volume_active:
            # Uwzględnij skłonność do shortowania z profilu
            if random.random() < profile['short_frequency']:
                signal = "SHORT"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
            
        self.logger.info(f"🎯 {self.active_profile} SIGNAL: {symbol} -> {signal} (Conf: {final_confidence:.1%})")
        
        return signal, final_confidence

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> Tuple[float, float, float]:
        """Oblicza wielkość pozycji w stylu LLM"""
        profile = self.get_current_profile()
        
        # Bazowa alokacja w zależności od profilu
        base_allocation = {
            'Claude': 0.15,
            'Gemini': 0.25, 
            'GPT': 0.10,
            'Qwen': 0.30
        }.get(self.active_profile, 0.15)
        
        # Modyfikator confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        # Modyfikator agresywności
        sizing_multiplier = {
            'CONSERVATIVE': 0.8,
            'AGGRESSIVE': 1.2,
            'VERY_AGGRESSIVE': 1.5
        }.get(profile['position_sizing'], 1.0)
        
        # Oblicz wartość pozycji
        position_value = (self.virtual_capital * base_allocation * 
                         confidence_multiplier * sizing_multiplier)
        
        # Limit maksymalnej pozycji
        max_position_value = self.virtual_capital * 0.4
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / price
        margin_required = position_value / self.leverage
        
        return quantity, position_value, margin_required

    def calculate_llm_exit_plan(self, entry_price: float, confidence: float, side: str) -> Dict:
        """Oblicza plan wyjścia w stylu LLM"""
        profile = self.get_current_profile()
        
        if confidence > 0.7:  # Wysoka confidence
            if side == "LONG":
                take_profit = entry_price * 1.018  # 1.8% TP
                stop_loss = entry_price * 0.992    # 0.8% SL
            else:  # SHORT
                take_profit = entry_price * 0.982  # 1.8% TP
                stop_loss = entry_price * 1.008    # 0.8% SL
        elif confidence > 0.5:  # Średnia confidence
            if side == "LONG":
                take_profit = entry_price * 1.012  # 1.2% TP
                stop_loss = entry_price * 0.994    # 0.6% SL
            else:  # SHORT
                take_profit = entry_price * 0.988  # 1.2% TP
                stop_loss = entry_price * 1.006    # 0.6% SL
        else:  # Niska confidence
            if side == "LONG":
                take_profit = entry_price * 1.008  # 0.8% TP
                stop_loss = entry_price * 0.996    # 0.4% SL
            else:  # SHORT
                take_profit = entry_price * 0.992  # 0.8% TP
                stop_loss = entry_price * 1.004    # 0.4% SL
        
        # Modyfikuj wg profilu ryzyka
        risk_multiplier = {
            'LOW': 0.8,
            'MEDIUM': 1.0,
            'HIGH': 1.2
        }.get(profile['risk_appetite'], 1.0)
        
        if side == "LONG":
            take_profit = entry_price + (take_profit - entry_price) * risk_multiplier
            stop_loss = entry_price - (entry_price - stop_loss) * risk_multiplier
        else:
            take_profit = entry_price - (entry_price - take_profit) * risk_multiplier
            stop_loss = entry_price + (stop_loss - entry_price) * risk_multiplier
        
        return {
            'take_profit': round(take_profit, 4),
            'stop_loss': round(stop_loss, 4),
            'invalidation': entry_price * 0.98 if side == "LONG" else entry_price * 1.02,
            'max_holding_hours': random.randint(1, 6)  # Losowy czas holdingu
        }

    def should_enter_trade(self) -> bool:
        """Decyduje czy wejść w transakcję wg profilu częstotliwości"""
        profile = self.get_current_profile()
        
        frequency_chance = {
            'LOW': 0.3,      # 30% szans na transakcję
            'MEDIUM': 0.5,   # 50% szans
            'HIGH': 0.7      # 70% szans
        }.get(profile['trade_frequency'], 0.5)
        
        return random.random() < frequency_chance

    def open_llm_position(self, symbol: str):
        """Otwiera pozycję w stylu LLM"""
        if not self.should_enter_trade():
            return None
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
            
        signal, confidence = self.generate_llm_signal(symbol)
        if signal == "HOLD" or confidence < 0.3:
            return None
            
        # Sprawdź limit pozycji
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            return None
            
        # Oblicz wielkość pozycji
        quantity, position_value, margin_required = self.calculate_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
            
        # Oblicz plan wyjścia
        exit_plan = self.calculate_llm_exit_plan(current_price, confidence, signal)
        
        # Cena likwidacji
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
            'exit_plan': exit_plan
        }
        
        self.positions[position_id] = position
        self.virtual_balance -= margin_required
        
        # Statystyki
        if signal == "LONG":
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        # Logowanie
        tp_distance = (exit_plan['take_profit'] - current_price) / current_price * 100
        sl_distance = (current_price - exit_plan['stop_loss']) / current_price * 100
        
        self.logger.info(f"🎯 {self.active_profile} OPEN: {symbol} {signal} @ ${current_price:.4f}")
        self.logger.info(f"   📊 Confidence: {confidence:.1%} | Size: ${position_value:.2f}")
        self.logger.info(f"   🎯 TP: {exit_plan['take_profit']:.4f} ({tp_distance:+.2f}%)")
        self.logger.info(f"   🛑 SL: {exit_plan['stop_loss']:.4f} ({sl_distance:+.2f}%)")
        
        return position_id

    def update_positions_pnl(self):
        """Aktualizuje P&L wszystkich pozycji"""
        total_unrealized = 0
        total_margin = 0
        total_confidence = 0
        confidence_count = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE':
                continue
                
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
                
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            else:  # SHORT
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
            total_unrealized += unrealized_pnl
            total_margin += position['margin']
            total_confidence += position['confidence']
            confidence_count += 1
        
        # Aktualizuj dashboard
        self.dashboard_data['unrealized_pnl'] = total_unrealized
        self.dashboard_data['account_value'] = self.virtual_capital + total_unrealized
        self.dashboard_data['available_cash'] = self.virtual_balance
        
        if confidence_count > 0:
            self.dashboard_data['average_confidence'] = total_confidence / confidence_count
        
        if self.virtual_capital > 0:
            self.stats['portfolio_utilization'] = total_margin / self.virtual_capital
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia z pozycji"""
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
            else:  # SHORT
                if current_price <= exit_plan['take_profit']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price >= exit_plan['stop_loss']:
                    exit_reason = "STOP_LOSS"
                elif current_price >= exit_plan['invalidation']:
                    exit_reason = "INVALIDATION"
                elif current_price >= position['liquidation_price']:
                    exit_reason = "LIQUIDATION"
            
            # Sprawdź maksymalny czas holdingu
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
        
        # Zapisz transakcję
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
        
        # Aktualizuj statystyki
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += realized_pnl_after_fee
        
        if realized_pnl_after_fee > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        # Aktualizuj średni czas holdingu
        total_holding = sum((t['exit_time'] - t['entry_time']).total_seconds() 
                          for t in self.trade_history) / 3600
        self.stats['avg_holding_time'] = total_holding / len(self.trade_history) if self.trade_history else 0
        
        position['status'] = 'CLOSED'
        self.dashboard_data['net_realized'] = self.stats['total_pnl']
        
        # Logowanie
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
            
            # Wskaźnik Herfindahla
            concentration_index = sum((p['margin'] / total_margin) ** 2 for p in active_positions)
            diversity = 1 - concentration_index
            
            return diversity
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating portfolio diversity: {e}")
            return 0

    def get_dashboard_data(self):
        """Przygotowuje dane dla dashboardu"""
        active_positions = []
        total_confidence = 0
        confidence_count = 0
        
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = position.get('current_price', self.get_current_price(position['symbol']))
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
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
                    'exit_plan': position['exit_plan']
                })
                
                total_confidence += position['confidence']
                confidence_count += 1
        
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
                'exit_time': trade['exit_time'].strftime('%H:%M:%S')
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
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_llm_trading_strategy(self):
        """Główna pętla strategii LLM"""
        self.logger.info("🚀 STARTING LLM-STYLE TRADING STRATEGY")
        self.logger.info(f"🎯 Active Profile: {self.active_profile}")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                self.logger.info(f"\n🔄 LLM Trading Iteration #{iteration}")
                
                # 1. Aktualizuj P&L
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
                                time.sleep(1)  # Krótka przerwa między pozycjami
                
                # 4. Loguj status
                portfolio_value = self.dashboard_data['account_value']
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Active Positions: {active_count}/{self.max_simultaneous_positions}")
                
                # 5. Odczekaj przed kolejną iteracją
                wait_time = random.randint(30, 90)  # Losowy interwał 30-90 sekund
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
