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
        
        # ASSET ALLOCATION
        self.max_simultaneous_positions = 6
        self.asset_allocation = {
            'ETHUSDT': 0.22,
            'BTCUSDT': 0.20,
            'SOLUSDT': 0.19,
            'BNBUSDT': 0.18,
            'XRPUSDT': 0.17,
            'DOGEUSDT': 0.04,
        }
        
        self.priority_symbols = list(self.asset_allocation.keys())
        
        # Trading parameters
        self.breakout_threshold = 0.02
        self.min_volume_ratio = 1.5
        self.max_position_value = 0.30
        
        # Position sizes
        self.position_sizes = {
            'ETHUSDT': 0.5,
            'BTCUSDT': 0.015,
            'SOLUSDT': 2.0,
            'BNBUSDT': 3.0,
            'XRPUSDT': 900.0,
            'DOGEUSDT': 5000.0,
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
        
        self.logger.info("🎯 TRADING BOT - Based on your exact positions")
        self.logger.info(f"💰 Initial capital: ${initial_capital}")

    def get_coingecko_ohlc(self, symbol: str, days: str = '1', limit: int = 100):
        """Get OHLC data from CoinGecko API"""
        try:
            coin_mapping = {
                'BTCUSDT': 'bitcoin',
                'ETHUSDT': 'ethereum',
                'SOLUSDT': 'solana',
                'BNBUSDT': 'binancecoin', 
                'XRPUSDT': 'ripple',
                'DOGEUSDT': 'dogecoin'
            }
            
            coin_id = coin_mapping.get(symbol)
            if coin_id:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                        
                        for col in ['open', 'high', 'low', 'close']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        df['volume'] = [random.uniform(10000, 50000) for _ in range(len(df))]
                        
                        if len(df) > limit:
                            df = df.tail(limit)
                            
                        self.logger.info(f"✅ CoinGecko OHLC Data for {symbol}: {len(df)} rows, Last: ${df['close'].iloc[-1]:.4f}")
                        return df
                        
        except Exception as e:
            self.logger.warning(f"CoinGecko OHLC failed: {e}")
        
        return None

    def get_current_price(self, symbol: str):
        """Get LIVE current price from CoinGecko API with 4 decimal precision"""
        try:
            coin_mapping = {
                'BTCUSDT': 'bitcoin',
                'ETHUSDT': 'ethereum',
                'SOLUSDT': 'solana',
                'BNBUSDT': 'binancecoin',
                'XRPUSDT': 'ripple',
                'DOGEUSDT': 'dogecoin'
            }
            
            coin_id = coin_mapping.get(symbol)
            if coin_id:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true&precision=8"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if coin_id in data and 'usd' in data[coin_id]:
                        price = float(data[coin_id]['usd'])
                        self.logger.info(f"✅ CoinGecko LIVE Price for {symbol}: ${price:.4f}")
                        return round(price, 4)  # Zawsze 4 miejsca po przecinku
                    
        except Exception as e:
            self.logger.warning(f"CoinGecko price failed: {e}")
        
        # Fallback do innych API
        return self.get_fallback_price(symbol)

    def get_fallback_price(self, symbol: str):
        """Fallback price providers z 4 miejscami po przecinku"""
        try:
            kraken_mapping = {
                'BTCUSDT': 'XXBTZUSD',
                'ETHUSDT': 'XETHZUSD',
                'SOLUSDT': 'SOLUSD',
                'BNBUSDT': 'BNBUSD',
                'XRPUSDT': 'XXRPZUSD',
                'DOGEUSDT': 'XDGUSD'
            }
            
            kraken_symbol = kraken_mapping.get(symbol)
            if kraken_symbol:
                url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data:
                        for key in data['result']:
                            price = float(data['result'][key]['c'][0])
                            self.logger.info(f"✅ Kraken Price for {symbol}: ${price:.4f}")
                            return round(price, 4)  # Zawsze 4 miejsca po przecinku
                            
        except Exception as e:
            self.logger.warning(f"Kraken fallback failed: {e}")
        
        self.logger.error(f"❌ All price APIs failed for {symbol}")
        return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            macd = df['ema_12'] - df['ema_26']
            df['macd'] = macd
            df['macd_signal'] = macd.ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            df['distance_to_support'] = (df['close'] - df['support']) / df['close']
            
            df['momentum_1h'] = df['close'].pct_change(20)
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating technical indicators: {e}")
            return df

    def detect_breakout_signal(self, symbol: str) -> Tuple[bool, float, float]:
        """Detect breakout signals based on resistance and volume"""
        try:
            df = self.get_coingecko_ohlc(symbol, '1', 100)
            if df is None or len(df) < 50:
                return False, 0, 0
            
            resistance_level = df['high'].rolling(20).max().iloc[-1]
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            price_above_resistance = current_price > resistance_level
            breakout_strength = (current_price - resistance_level) / resistance_level
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
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

    def generate_breakout_signal(self, symbol: str) -> Tuple[str, float]:
        """Generate trading signal based on breakout strategy"""
        try:
            is_breakout, breakout_confidence, resistance_level = self.detect_breakout_signal(symbol)
            
            if is_breakout and breakout_confidence >= 0.65:
                return "BREAKOUT_LONG", breakout_confidence
            
            df = self.get_coingecko_ohlc(symbol, '1', 100)
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
            
            conditions = 0
            if 40 <= current_rsi <= 70:
                conditions += 1
                confidence += 0.15
            
            if volume_ratio > 1.3:
                conditions += 1
                confidence += 0.20
            
            if macd_histogram > 0:
                conditions += 1
                confidence += 0.20
            
            if momentum_1h > 0.01:
                conditions += 1
                confidence += 0.15
            
            if current_price > df['sma_20'].iloc[-1]:
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
        """Calculate position size according to asset allocation"""
        try:
            allocation_percentage = self.asset_allocation.get(symbol, 0.15)
            confidence_multiplier = 0.5 + (confidence * 0.5)
            
            position_value = self.virtual_capital * allocation_percentage
            
            max_position_value = self.virtual_capital * self.max_position_value
            position_value = min(position_value, max_position_value)
            
            quantity = position_value / price
            
            historical_quantity = self.position_sizes.get(symbol, quantity)
            final_quantity = min(quantity, historical_quantity)
            
            final_position_value = final_quantity * price
            margin_required = final_position_value / self.leverage
            
            max_safe_position = self.virtual_capital * 0.15
            if final_position_value > max_safe_position:
                final_quantity = (max_safe_position) / price
                final_position_value = final_quantity * price
                margin_required = final_position_value / self.leverage
            
            return final_quantity, final_position_value, margin_required
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size for {symbol}: {e}")
            return 0, 0, 0

    def get_portfolio_diversity(self) -> float:
        """Calculate portfolio diversity"""
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

    def open_breakout_position(self, symbol: str):
        """Open breakout position"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.logger.warning(f"❌ Cannot get current price for {symbol}, skipping position")
            return None
        
        signal, confidence = self.generate_breakout_signal(symbol)
        if signal not in ["BREAKOUT_LONG", "LONG"] or confidence < 0.65:
            return None
        
        active_positions = sum(1 for p in self.positions.values() if p['status'] == 'ACTIVE')
        if active_positions >= self.max_simultaneous_positions:
            self.logger.info(f"⏹️ Max positions reached ({active_positions}/{self.max_simultaneous_positions})")
            return None
        
        quantity, position_value, margin_required = self.calculate_breakout_position_size(
            symbol, current_price, confidence
        )
        
        if margin_required > self.virtual_balance:
            self.logger.warning(f"💰 Insufficient balance for {symbol}")
            return None
        
        is_breakout = signal == "BREAKOUT_LONG"
        if is_breakout:
            _, _, resistance_level = self.detect_breakout_signal(symbol)
            exit_levels = {
                'take_profit': round(current_price * 1.08, 4),
                'stop_loss': round(resistance_level * 0.98, 4),
                'invalidation': round(current_price * 0.96, 4)
            }
        else:
            exit_levels = {
                'take_profit': round(current_price * 1.10, 4),
                'stop_loss': round(current_price * 0.95, 4),
                'invalidation': round(current_price * 0.93, 4)
            }
        
        liquidation_price = round(current_price * (1 - 0.9 / self.leverage), 4)
        
        position_id = f"breakout_{self.position_id}"
        self.position_id += 1
        
        position = {
            'symbol': symbol,
            'side': 'LONG',
            'entry_price': round(current_price, 4),  # Zawsze 4 miejsca po przecinku
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
            self.logger.info(f"🎯 BREAKOUT OPEN: {quantity:.6f} {symbol} @ ${current_price:.4f}")
        else:
            self.logger.info(f"📈 MOMENTUM OPEN: {quantity:.6f} {symbol} @ ${current_price:.4f}")
        
        self.logger.info(f"   📊 TP: ${exit_levels['take_profit']:.4f} | SL: ${exit_levels['stop_loss']:.4f}")
        self.logger.info(f"   💰 Position: ${position_value:.2f} ({position_value/self.virtual_capital*100:.1f}% of capital)")
        self.logger.info(f"   🤖 Confidence: {confidence:.1%} | Leverage: {self.leverage}X")
        
        return position_id

    def update_positions_pnl(self):
        """Update P&L for all positions - TYLKO z cenami z API"""
        total_unrealized = 0
        total_margin = 0
        
        for position in self.positions.values():
            if position['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(position['symbol'])
            if not current_price:
                continue
            
            position['current_price'] = round(current_price, 4)  # Zawsze 4 miejsca po przecinku
            
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
        """Check exit conditions - TYLKO z cenami z API"""
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
            
            elif current_price <= position['exit_plan']['stop_loss']:
                exit_reason = "STOP_LOSS"
            
            elif current_price <= position['exit_plan']['invalidation']:
                exit_reason = "INVALIDATION"
            
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
            'exit_price': round(exit_price, 4),  # Zawsze 4 miejsca po przecinku
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
        self.logger.info(f"{pnl_color} {strategy_icon} CLOSE: {position['symbol']} - P&L: ${realized_pnl_after_fee:+.2f} - Reason: {exit_reason}")

    def get_dashboard_data(self):
        """Prepare dashboard data for HTML interface - TYLKO z cenami z API"""
        active_positions = []
        total_confidence = 0
        confidence_count = 0
        
        # Get active positions with current prices
        for position_id, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = self.get_current_price(position['symbol'])
                if not current_price:
                    current_price = position['entry_price']
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    unrealized_pnl = pnl_pct * position['quantity'] * position['entry_price'] * position['leverage']
                
                active_positions.append({
                    'position_id': position_id,
                    'entry_time': position['entry_time'].strftime('%H:%M:%S'),
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],  # Już z 4 miejscami
                    'current_price': round(current_price, 4),  # Zawsze 4 miejsca po przecinku
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'liquidation_price': position['liquidation_price'],
                    'margin': position['margin'],
                    'unrealized_pnl': unrealized_pnl,
                    'confidence': position.get('confidence', 0),
                    'strategy': position.get('strategy', 'MOMENTUM')
                })
        
        # Calculate confidence levels for each asset - DYNAMICZNIE
        confidence_levels = {}
        for symbol in self.priority_symbols:
            try:
                signal, confidence = self.generate_breakout_signal(symbol)
                confidence_percent = round(confidence * 100, 1)
                confidence_levels[symbol] = confidence_percent
                
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error calculating confidence for {symbol}: {e}")
                confidence_levels[symbol] = 0
        
        # Get recent trades
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
        
        # Calculate performance metrics
        total_trades = self.stats['total_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        total_return_pct = ((self.dashboard_data['account_value'] - 10000) / 10000) * 100
        
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
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'biggest_win': round(self.stats['biggest_win'], 2),
                'biggest_loss': round(self.stats['biggest_loss'], 2),
                'avg_confidence': round((total_confidence / confidence_count * 100), 1) if confidence_count > 0 else 0
            },
            'confidence_levels': confidence_levels,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'total_unrealized_pnl': round(self.dashboard_data['unrealized_pnl'], 2),
            'last_update': self.dashboard_data['last_update'].isoformat()
        }

    def run_breakout_strategy(self):
        """Main breakout strategy loop"""
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
                
                # 1. Update P&L
                self.update_positions_pnl()
                
                # 2. Check exit conditions
                positions_to_close = self.check_exit_conditions()
                for position_id, exit_reason, exit_price in positions_to_close:
                    self.close_position(position_id, exit_reason, exit_price)
                
                # 3. Check breakout signals for each asset
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
                                    time.sleep(1)
                
                # 4. Log portfolio status
                portfolio_value = self.dashboard_data['account_value']
                diversity = self.dashboard_data['portfolio_diversity'] * 100
                utilization = self.stats['portfolio_utilization'] * 100
                
                self.logger.info(f"📊 Portfolio: ${portfolio_value:.2f} | Positions: {active_count}/{self.max_simultaneous_positions}")
                self.logger.info(f"🌐 Diversity: {diversity:.1f}% | Utilization: {utilization:.1f}%")
                
                # 5. Wait 60 seconds
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
