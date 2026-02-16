# Gold MGC Day Trading Bot

Locked configuration (2026):
- Micro Gold Futures (MGC)
- TP: 50 ticks
- Trailing stop: activate +20 ticks, trail 40 ticks
- Strict trend filter (only buy in uptrend, sell in downtrend)
- Risk: $500/trade
- Fees & slippage included

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and add your Polygon API key
3. `python main.py`

## Features
- Hourly zones + 1-min entries (consolidation & pin bars)
- RL signal filtering
- Realistic backtest with fees/slippage
- Strict directional trend filter

## Limitations
- Backtest uses XAUUSD spot proxy
- 1-min data limited to recent ~60 days (expand locally or use broker feed for full history)
- This is research/paper trading code — not production live trading

For improvements or live integration (NinjaTrader/Tradovate) → open an issue.
