# main.py
"""Entry point to run the MGC trading bot backtest"""

from bot import GoldTradingBot

if __name__ == "__main__":
    print("Gold MGC Day Trading Bot - Locked Config (2026)")
    print("TP: 50 ticks | Trail start: 20 ticks | Trail dist: 40 ticks")
    print("Strict trend filter | Risk: $500/trade | Fees & slippage included\n")

    bot = GoldTradingBot()
    bot.fetch_data(years=5)
    bot.backtest()

    print("\nBacktest finished.")
