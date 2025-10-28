# app.py
from flask import Flask, render_template, jsonify
from trading_bot import trading_bot
import threading
import time
from datetime import datetime
import json
import os

app = Flask(__name__)

# Globalna instancja bota
trading_bot_instance = None
bot_thread = None

def create_bot():
    global trading_bot_instance
    trading_bot_instance = trading_bot
    return trading_bot_instance

def run_bot():
    """Uruchamia bota w tle"""
    bot = create_bot()
    print("🤖 Starting LIVE trading bot...")
    bot.run_exact_strategy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trading-data')
def get_trading_data():
    """API z danymi dla dashboardu"""
    if trading_bot_instance is None:
        return jsonify({'error': 'Bot not initialized'})
    
    try:
        data = trading_bot_instance.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/start-bot')
def start_bot():
    """Uruchamia bota"""
    global bot_thread, trading_bot_instance
    
    if trading_bot_instance is None:
        trading_bot_instance = create_bot()
    
    if not trading_bot_instance.is_running:
        trading_bot_instance.is_running = True
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        return jsonify({
            'status': '✅ Bot LIVE started successfully', 
            'timestamp': datetime.now().isoformat(),
            'mode': 'LIVE - Real-time trading'
        })
    
    return jsonify({'status': 'Bot is already running', 'timestamp': datetime.now().isoformat()})

@app.route('/api/stop-bot')
def stop_bot():
    """Zatrzymuje bota"""
    global trading_bot_instance
    
    if trading_bot_instance and trading_bot_instance.is_running:
        trading_bot_instance.is_running = False
        return jsonify({
            'status': '🛑 Bot stopped successfully', 
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({'status': 'Bot is not running', 'timestamp': datetime.now().isoformat()})

@app.route('/api/bot-status')
def bot_status():
    """Status bota"""
    if trading_bot_instance is None:
        return jsonify({'status': 'not_initialized', 'mode': 'OFFLINE'})
    
    return jsonify({
        'status': 'running' if trading_bot_instance.is_running else 'stopped',
        'mode': 'LIVE',
        'timestamp': datetime.now().isoformat(),
        'symbols': trading_bot_instance.symbols
    })

@app.route('/api/force-update')
def force_update():
    """Wymusza aktualizację danych"""
    if trading_bot_instance:
        trading_bot_instance.update_positions_pnl()
        return jsonify({'status': 'Data updated', 'timestamp': datetime.now().isoformat()})
    return jsonify({'error': 'Bot not available'})

# Inicjalizacja przy starcie
if __name__ == '__main__':
    print("🎯 LIVE TRADING BOT - Starting...")
    print("📍 Local URL: http://localhost:5000")
    print("📍 Network URL: http://192.168.1.X:5000")  # Zmień na swój IP
    print("🤖 Bot will use REAL Binance API data")
    print("⚠️  WARNING: Bot will open/close positions based on real market data!")
    
    # Utwórz instancję bota (ale nie uruchamiaj jeszcze)
    create_bot()
    
    # Uruchom serwer Flask
    app.run(host='0.0.0.0', port=5000, debug=True)