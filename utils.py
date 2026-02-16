# utils.py
from datetime import datetime
from config import *

def is_trading_time(ts: datetime) -> bool:
    h, m = ts.hour, ts.minute
    if h < TRADING_START_HOUR or h > TRADING_END_HOUR: return False
    if h == TRADING_START_HOUR and m < TRADING_START_MIN: return False
    if h == TRADING_END_HOUR   and m > TRADING_END_MIN:   return False
    return True

def prepare_state(row, price, strength):
    trend_diff = (price - row['sma_50']) / row['sma_50'] if row['sma_50'] != 0 else 0
    vol_ratio  = row['volume'] / row['avg_vol'] if row['avg_vol'] != 0 else 1.0
    time_frac  = row.name.hour + row.name.minute / 60.0
    return [trend_diff, vol_ratio, time_frac, strength]
