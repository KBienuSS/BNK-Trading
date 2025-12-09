# export_history.py
import json
import time
from datetime import datetime

# Podłącz się do istniejącego bota
try:
    # Importuj z głównego pliku
    from trading_bot_ml import trading_bot
    
    print("🔗 Połączono z działającym botem")
    print("📊 Pobieranie historii transakcji...")
    
    # Poczekaj chwilę na aktualizację danych
    time.sleep(2)
    
    # Pobierz historię transakcji
    history = trading_bot.trade_history
    
    print(f"✅ Znaleziono {len(history)} transakcji")
    
    if history:
        # Przygotuj dane do eksportu
        export_data = []
        total_pnl = 0
        
        for trade in history:
            serializable_trade = {
                'position_id': trade['position_id'],
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'quantity': trade['quantity'],
                'realized_pnl': trade['realized_pnl'],
                'exit_reason': trade['exit_reason'],
                'llm_profile': trade['llm_profile'],
                'confidence': trade['confidence'],
                'entry_time': trade['entry_time'].isoformat(),
                'exit_time': trade['exit_time'].isoformat(),
                'holding_hours': round(trade['holding_hours'], 2)
            }
            export_data.append(serializable_trade)
            total_pnl += trade['realized_pnl']
        
        # Zapisz do JSON
        filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Historia zapisana do: {filename}")
        print(f"💰 Łączny P&L: ${total_pnl:.2f}")
        
        # Wyświetl podsumowanie
        print("\n📈 PODSUMOWANIE:")
        print("-" * 60)
        for trade in history[-10:]:  # Ostatnie 10 transakcji
            pnl_sign = "+" if trade['realized_pnl'] > 0 else ""
            print(f"{trade['symbol']} {trade['side']} | "
                  f"P&L: ${pnl_sign}{trade['realized_pnl']:.2f} | "
                  f"Czas: {trade['exit_time'].strftime('%H:%M:%S')}")
    
except Exception as e:
    print(f"❌ Błąd: {e}")
    print("Upewnij się, że bot jest uruchomiony")
