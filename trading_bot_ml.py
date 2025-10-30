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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_enhanced.log', encoding='utf-8')
    ]
)

class EnhancedTradingBot:
    def __init__(self, initial_capital=10000, leverage=10):
        self.virtual_capital = initial_capital
        self.virtual_balance = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.position_id = 0
        
        self.logger = logging.getLogger(__name__)
        
        # STRATEGIA OPARTA NA ANALIZIE PDF - ZAKTUALIZOWANA
        self.max_simultaneous_positions = 6  # Zwiększone do 6
        
        # ALOKACJA KAPITAŁU DLA WSZYSTKICH 6 KRYPTOWALUT
        self.asset_allocation = {
            'ETHUSDT': 0.25,  # Główna pozycja
            'XRPUSDT': 0.20,  # Wysoka alokacja 
            'SOLUSDT': 0.18,  # Znacząca pozycja
            'BTCUSDT': 0.15,  # Umiarkowana alokacja
            'BNBUSDT': 0.12,  # Średnia pozycja
            'DOGEUSDT': 0.10, # Dodana zgodnie z wymaganiami
        }
        
        self.priority_symbols = ['ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'BTCUSDT', 'BNBUSDT', 'DOGEUSDT']
        
        # PARAMETRY BREAKOUT Z ANALIZY
        self.breakout_threshold = 0.015
        self.min_volume_ratio = 1.8
        self.max_position_value = 0.40
        
        # WIELKOŚCI POZYCJI DLA WSZYSTKICH 6 KRYPTOWALUT
        self.position_sizes = {
            'ETHUSDT': 3.2,
            'XRPUSDT': 8500.0, 
            'SOLUSDT': 28.0,
            'BTCUSDT': 0.045,
            'BNBUSDT': 8.5,
            'DOGEUSDT': 25000.0,  # Dodana wielkość pozycji
        }
        
        # HANDEL 24/7 - WYŁĄCZONA OGRANICZENIA CZASOWE
        self.trading_hours = {
            'start_utc': 0,      # 24/7
            'end_utc': 23,       # 24/7
            'enabled': False     # Wyłączone ograniczenia czasowe
        }
        
        # LONG BIAS 95% Z ANALIZY
        self.long_bias = 0.95
        self.pyramiding_enabled = True
        self.risk_tolerance = "HIGH"
        
        # STATYSTYKI
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'breakout_trades': 0,
            'portfolio_utilization': 0,
            'liquidation_events': 0,
            'macro_clusters': 0,
            'full_liquidation_losses': 0  # Nowa statystyka
        }
        
        # DASHBOARD
        self.dashboard_data = {
            'account_value': initial_capital,
            'available_cash': initial_capital,
            'net_realized': 0,
            'unrealized_pnl': 0,
            'portfolio_diversity': 0,
            'last_update': datetime.now(),
            'trading_window_active': True  # Zawsze aktywne
        }
        
        self.logger.info("🚀 ENHANCED TRADING BOT - PDF STRATEGY (24/7)")
        self.logger.info(f"💰 Initial capital: ${initial_capital}")
        self.logger.info("📊 6 Assets Allocation: ETH(25%) XRP(20%) SOL(18%) BTC(15%) BNB(12%) DOGE(10%)")
        self.logger.info("🕒 Trading: 24/7 | Long Bias: 95% | Max Positions: 6")
        self.logger.info("⚡ Breakout Strategy + Pyramiding + Full Risk Acceptance")

    def is_trading_hours(self):
        """Zawsze zwraca True - handel 24/7"""
        self.dashboard_data['trading_window_active'] = True
        return True

    def should_enter_long(self):
        """Decyzja o wejściu LONG z biasem 95%"""
        return random.random() < self.long_bias

    def get_market_data(self, symbol: str, interval: str = '3m', limit: int = 100):
        """Pobiera dane rynkowe z API"""
        try:
            # Symulacja danych - w rzeczywistości podłącz pod prawdziwe API
            base_prices = {
                'BTCUSDT': 112614,
                'ETHUSDT': 3485, 
                'BNBUSDT': 582,
                'SOLUSDT': 178,
                'XRPUSDT': 0.615,
                'DOGEUSDT': 0.148
            }
            
            base_price = base_prices.get(symbol, 100)
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
                    'close': current_price,
                    'volume': random.uniform(5000, 20000)
                })
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market data: {e}")
            return None

    def get_current_price(self, symbol: str):
        """Pobiera aktualną cenę"""
        try:
            # Symulacja ceny - w rzeczywistości podłącz pod prawdziwe API
            base_prices = {
                'BTCUSDT': 112614,
                'ETHUSDT': 3485,
                'BNBUSDT': 582,
                'SOLUSDT': 178, 
                'XRPUSDT': 0.615,
                'DOGEUSDT': 0.148
            }
            
            base_price = base_prices.get(symbol, 100)
            volatility = {
                'BTCUSDT': 0.0005, 'ETHUSDT': 0.0008, 'BNBUSDT': 0.001,
                'SOLUSDT': 0.0015, 'XRPUSDT': 0.002, 'DOGEUSDT': 0.003
            }.get(symbol, 0.001)
            
            change = random.gauss(0, volatility)
            live_price = base_price * (1 + change)
            
            return round(live_price, 6 if symbol == 'DOGEUSDT' else 2)
            
        except Exception as e:
            self.logger.error(f"❌ Error getting current price: {e}")
            return None

    def detect_breakout_signal(self, symbol: str) -> Tuple[bool, float, float]:
        """Wykrywa sygnały breakout zgodnie z analizą PDF"""
        try:
            df = self.get_market_data(symbol, '3m', 100)
            if df is None or len(df) < 50:
                return False, 0, 0
            
            # Analiza oporu i wsparcia
            resistance_level = df['high'].rolling(20).max().iloc[-1]
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # Warunki breakout
            price_above_resistance = current_price > resistance_level
            breakout_strength = (current_price - resistance_level) / resistance_level
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Dodatkowe filtry
            trend_strength = (current_price - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            is_breakout = (price_above_resistance and 
                          breakout_strength >= self.breakout_threshold and
                          volume_ratio >= self.min_volume_ratio and
                          trend_strength > 0.01)
            
            confidence = min(breakout_strength * 12 + volume_ratio * 0.3, 0.95)
            
            if is_breakout:
                self.logger.info(f"🎯 BREAKOUT: {symbol} | Strength: {breakout_strength:.2%}")
            
            return is_breakout, confidence, resistance_level
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting breakout: {e}")
            return False, 0, 0

    def generate_breakout_signal(self, symbol: str) -> Tuple[str, float]:
        """Generuje sygnał oparty na strategii breakout"""
        try:
            is_breakout, breakout_confidence, resistance_level = self.detect_breakout_signal(symbol)
            
            if is_breakout and breakout_confidence >= 0.65:
                return "BREAKOUT_LONG", breakout_confidence
            
            # Fallback do momentum
            df = self.get_market_data(symbol, '3m', 100)
            if df is None:
                return "HOLD", 0.5
            
            current_price = df['close'].iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            confidence = 0.0
            signal = "HOLD"
            
            # Proste warunki momentum
            conditions = 0
            if volume_ratio > 1.3:
                conditions += 1
                confidence += 0.3
            
            if current_price > df['close'].rolling(20).mean().iloc[-1]:
                conditions += 1 
                confidence += 0.3
            
            if (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] > 0.01:
                conditions += 1
                confidence += 0.3
            
            if conditions >= 2:
                signal = "LONG"
                confidence = min(confidence, 0.85)
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal: {e}")
            return "HOLD", 0.5

    def calculate_dynamic_position_size(self, symbol: str, price: float, confidence: float, is_breakout: bool):
        """Oblicza wielkość pozycji z pyramidingiem zgodnie z analizą"""
        try:
            base_allocation = self.asset_allocation.get(symbol, 0.15)
            confidence_multiplier = 0.6 + (confidence * 0.4)
            breakout_bonus = 1.2 if is_breakout else 1.0
            
            # Pyramiding - reinwestycja zysków (jak w analizie ETH $114.51K)
            if self.pyramiding_enabled and self.stats['total_pnl'] > 0:
                reinvestment = min(self.stats['total_pnl'] * 0.6, self.virtual_capital * 0.3)
                effective_capital = self.virtual_capital + reinvestment
            else:
                effective_capital = self.virtual_capital
            
            position_value = (effective_capital * base_allocation * 
                           confidence_multiplier * breakout_bonus)
            
            max_position_value = effective_capital * self.max_position_value
            position_value = min(position_value, max_position_value)
            
            historical_quantity = self.position_sizes.get(symbol, position_value / price)
            final_quantity = min(position_value / price, historical_quantity * 1.5)
            
            final_position_value = final_quantity * price
            margin_required = final_position_value / self.leverage
            
            return final_quantity, final_position_value, margin_required
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0, 0, 0

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
                self.logger.info("🎯 MACRO CLUSTER SIGNAL DETECTED")
                return True, strong_signals
            
            return False, []
            
        except Exception as e:
            self.logger.error(f"❌ Error generating macro signal: {e}")
            return False, []

    def open_macro_positions(self, strong_signals):
        """Otwiera zsynchronizowane pozycje w klastrach"""
        opened_positions = []
        
        for symbol, (signal, confidence) in strong_signals:
            if len(opened_positions) >= 3:  # Zwiększone do 3 w klastrze
                break
                
            position_id = self.open_breakout_position(symbol)
            if position_id:
                opened_positions.append(position_id)
                self.stats['macro_clusters'] += 1
                time.sleep(1)  # Krótsze opóźnienie dla 24/7
        
        return opened_positions

    def calculate_pdf_stop_loss_levels(self, entry_price: float, resistance_level: float, is_breakout: bool):
        """Oblicza poziomy Stop Loss zgodnie z analizą PDF - BRAK LUB BARDZO SZEROKIE SL"""
        if is_breakout:
            # Dla breakout: SL tuż poniżej poziomu breakout
            return {
                'take_profit': entry_price * 1.08,   # 8% TP
                'stop_loss': resistance_level * 0.98, # 2% poniżej breakout
                'invalidation': entry_price * 0.90    # Szeroki SL -10%
            }
        else:
            # Dla momentum: szerokie SL zgodnie z wysoką tolerancją ryzyka
            return {
                'take_profit': entry_price * 1.10,   # 10% TP  
                'stop_loss': entry_price * 0.85,     # Szeroki SL -15%
                'invalidation': entry_price * 0.80   # Bardzo szeroki invalidation -20%
            }

    def should_close_due_to_liquidation_risk(self, position: dict) -> bool:
        """Sprawdza ryzyko likwidacji zgodnie z analizą PDF - ŚWIADOMA AKCEPTACJA LIKWIDACJI"""
        current_price = self.get_current_price(position['symbol'])
        if not current_price:
            return False
        
        # Z analizy PDF: bot akceptuje pełną likwidację jako koszt strategii
        # Zwracamy False - pozwalamy na likwidację zamiast wcześniejszego zamknięcia
        liquidation_risk = (current_price - position['liquidation_price']) / position['entry_price']
        
        # Tylko ekstremalne sytuacje powodują wcześniejsze zamknięcie
        if liquidation_risk <= -0.25:  # 25% straty
            self.logger.info(f"⚠️ Extreme liquidation risk: {liquidation_risk:.1%} - closing early")
            return True
            
        return False

    def open_breakout_position(self, symbol: str):
        """Otwiera pozycję zgodnie ze strategią z PDF"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None
        
        signal, confidence = self.generate_breakout_signal(symbol)
        if signal not in ["BREAKOUT_LONG", "LONG"] or confidence < 0.65:
            return None
        
        # Sprawdź limit pozycji - teraz 6
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        # Oblicz wielkość pozycji
        is_breakout = signal == "BREAKOUT_LONG"
        quantity, position_value, margin_required = self.calculate_dynamic_position_size(
            symbol, current_price, confidence, is_breakout
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
        
        # Poziomy wyjścia ZGODNE Z ANALIZĄ PDF - SZEROKIE SL
        if is_breakout:
            _, _, resistance_level = self.detect_breakout_signal(symbol)
            exit_levels = self.calculate_pdf_stop_loss_levels(current_price, resistance_level, True)
        else:
            exit_levels = self.calculate_pdf_stop_loss_levels(current_price, 0, False)
        
        liquidation_price = current_price * (1 - 0.9 / self.leverage)
        
        position_id = f"pos_{self.position_id}"
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
            self.logger.info(f"🎯 BREAKOUT OPEN: {symbol} @ ${current_price:.2f}")
        else:
            self.logger.info(f"📈 MOMENTUM OPEN: {symbol} @ ${current_price:.2f}")
        
        self.logger.info(f"   📊 TP: ${exit_levels['take_profit']:.2f} | SL: ${exit_levels['stop_loss']:.2f}")
        self.logger.info(f"   💰 Position: ${position_value:.2f} | Margin: ${margin_required:.2f}")
        self.logger.info(f"   ⚠️ Liquidation: ${liquidation_price:.2f} (High Risk Accepted)")
        
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
        
        if self.virtual_capital > 0:
            portfolio_utilization = (total_margin * self.leverage) / self.virtual_capital
            self.stats['portfolio_utilization'] = portfolio_utilization
        
        self.dashboard_data['last_update'] = datetime.now()

    def check_exit_conditions(self):
        """Sprawdza warunki wyjścia z pozycji ZGODNIE Z ANALIZĄ PDF"""
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            exit_reason = None
            
            # 1. TAKE PROFIT - klasyczny
            if current_price >= position['exit_plan']['take_profit']:
                exit_reason = "TAKE_PROFIT"
            
            # 2. STOP LOSS - szeroki zgodnie z analizą PDF
            elif current_price <= position['exit_plan']['stop_loss']:
                exit_reason = "STOP_LOSS"
            
            # 3. INVALIDATION - bardzo szeroki
            elif current_price <= position['exit_plan']['invalidation']:
                exit_reason = "INVALIDATION"
            
            # 4. LIKWIDACJA - ŚWIADOMIE AKCEPTOWANA (jak w analizie SOL)
            elif current_price <= position['liquidation_price']:
                exit_reason = "LIQUIDATION"
                self.stats['liquidation_events'] += 1
                self.stats['full_liquidation_losses'] += position['margin']
            
            # 5. EKSTREMALNE RYZYKO LIKWIDACJI (>25%) - zabezpieczenie
            elif self.should_close_due_to_liquidation_risk(position):
                exit_reason = "EXTREME_LIQUIDATION_RISK"
            
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
            'strategy': position.get('strategy', 'MOMENTUM'),
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
        
        if exit_reason == "LIQUIDATION":
            self.logger.info(f"💥 {strategy_icon} LIQUIDATION: {position['symbol']} - Loss: ${realized_pnl_after_fee:+.2f}")
            self.logger.info(f"   ⚠️ Full margin loss accepted: ${position['margin']:.2f}")
        else:
            self.logger.info(f"{pnl_color} {strategy_icon} CLOSE: {position['symbol']} - P&L: ${realized_pnl_after_fee:+.2f} - {exit_reason}")

    def run_pdf_strategy_24_7(self):
        """Główna pętla strategii 24/7 opartej na analizie PDF"""
        self.logger.info("🚀 STARTING 24/7 PDF-BASED TRADING STRATEGY")
        self.logger.info("📊 Trading 6 assets simultaneously with high risk acceptance")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
                
                self.logger.info(f"\n🔄 Iteration #{iteration} | {current_time}")
                
                # 1. Aktualizuj P&L
                self.update_positions_pnl()
                
                # 2. Sprawdź warunki wyjścia (szerokie SL i akceptacja likwidacji)
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Sprawdź sygnały makro (klastry transakcji)
                macro_signal, strong_signals = self.generate_macro_signal()
                if macro_signal:
                    self.logger.info("🎯 EXECUTING MACRO CLUSTER")
                    self.open_macro_positions(strong_signals)
                else:
                    # 4. Standardowe sprawdzanie sygnałów dla wszystkich 6 assetów
                    active_symbols = [p['symbol'] for p in self.positions.values() 
                                    if p['status'] == 'ACTIVE']
                    active_count = len(active_symbols)
                    
                    if active_count < self.max_simultaneous_positions:
                        for symbol in self.priority_symbols:
                            if symbol not in active_symbols:
                                # Long bias 95%
                                if not self.should_enter_long():
                                    continue
                                
                                signal, confidence = self.generate_breakout_signal(symbol)
                                
                                if signal in ["BREAKOUT_LONG", "LONG"] and confidence >= 0.65:
                                    self.open_breakout_position(symbol)
                                    time.sleep(1)  # Krótkie opóźnienie dla 24/7
                
                # 5. Loguj status
                portfolio_value = self.dashboard_data['account_value']
                active_count = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
                
                self.logger.info(f"💰 Portfolio: ${portfolio_value:.2f} | Active Positions: {active_count}/6")
                self.logger.info(f"📈 Total Trades: {self.stats['total_trades']} | Breakouts: {self.stats['breakout_trades']}")
                self.logger.info(f"⚡ Liquidations: {self.stats['liquidation_events']} | Clusters: {self.stats['macro_clusters']}")
                
                # 6. Czekaj 60 sekund (konsystentny interwał 24/7)
                for i in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(30)

    def start_trading(self):
        """Rozpoczyna handel 24/7"""
        self.is_running = True
        self.run_pdf_strategy_24_7()

    def stop_trading(self):
        """Zatrzymuje handel"""
        self.is_running = False
        self.logger.info("🛑 24/7 Trading stopped")

    def get_dashboard_data(self):
        """Przygotowuje dane do dashboardu"""
        active_positions = []
        
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = self.get_current_price(position['symbol'])
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
                active_positions.append({
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'unrealized_pnl': unrealized_pnl,
                    'liquidation_price': position['liquidation_price'],
                    'strategy': position.get('strategy', 'MOMENTUM'),
                    'margin': position['margin']
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
                'breakout_trades': self.stats['breakout_trades'],
                'macro_clusters': self.stats['macro_clusters'],
                'liquidation_events': self.stats['liquidation_events'],
                'full_liquidation_losses': round(self.stats['full_liquidation_losses'], 2),
                'portfolio_utilization': round(self.stats['portfolio_utilization'] * 100, 1)
            },
            'trading_info': {
                'mode': '24/7',
                'max_positions': 6,
                'active_positions': len(active_positions)
            },
            'active_positions': active_positions,
            'strategy_profile': {
                'long_bias': '95%',
                'pyramiding': 'Enabled', 
                'risk_tolerance': 'High',
                'stop_loss_policy': 'Wide/None (PDF Analysis)'
            },
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

# Global bot instance
trading_bot = EnhancedTradingBot(initial_capital=10000, leverage=10)
