from flask import Flask, render_template, jsonify, request
import threading
import time
import json
import logging
from datetime import datetime, timedelta
import random
import requests
from trading_bot import TradingBot
from trading_bot_ml import MLTradingBot

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global bot instances
trading_bot = None
ml_trading_bot = None
bot_status = "stopped"

class TradingData:
    def __init__(self):
        self.account_value = 50000
        self.available_cash = 35000
        self.total_fees = 124.50
        self.net_realized = 1567.89
        
    def get_trading_data(self):
        # This would normally come from the actual bot
        return {
            'account_summary': {
                'total_value': self.account_value,
                'available_cash': self.available_cash,
                'total_fees': self.total_fees,
                'net_realized': self.net_realized
            },
            'performance_metrics': {
                'avg_leverage': 8.5,
                'avg_confidence': 76.2,
                'biggest_win': 1245.67,
                'biggest_loss': -567.89
            },
            'active_positions': [],
            'recent_trades': [],
            'total_unrealized_pnl': 0
        }

trading_data = TradingData()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trading-data')
def get_trading_data():
    if trading_bot and bot_status == "running":
        return jsonify(trading_bot.get_dashboard_data())
    elif ml_trading_bot and bot_status == "running":
        return jsonify(ml_trading_bot.get_dashboard_data())
    else:
        return jsonify(trading_data.get_trading_data())

@app.route('/api/bot-status')
def get_bot_status():
    return jsonify({'status': bot_status})

@app.route('/api/start-bot')
def start_bot():
    global bot_status, trading_bot, ml_trading_bot
    try:
        if bot_status != "running":
            # Start ML bot by default
            ml_trading_bot = MLTradingBot()
            
            bot_thread = threading.Thread(target=run_bot)
            bot_thread.daemon = True
            bot_thread.start()
            
            bot_status = "running"
            return jsonify({'status': 'ML Bot started successfully'})
        else:
            return jsonify({'status': 'Bot is already running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-bot')
def stop_bot():
    global bot_status
    try:
        if bot_status == "running":
            if ml_trading_bot:
                ml_trading_bot.stop_trading()
            elif trading_bot:
                trading_bot.stop_trading()
            
            bot_status = "stopped"
            return jsonify({'status': 'Bot stopped successfully'})
        else:
            return jsonify({'status': 'Bot is not running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/force-update')
def force_update():
    try:
        return jsonify({'status': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_bot():
    """Run the trading bot in a separate thread"""
    global ml_trading_bot
    try:
        if ml_trading_bot:
            ml_trading_bot.start_trading()
    except Exception as e:
        logging.error(f"Error running bot: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
