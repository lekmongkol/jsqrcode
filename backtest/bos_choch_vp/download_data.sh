#!/usr/bin/env bash
# Download the free XAUUSD M15 datasets used by backtest.py into ./data
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
# 2012-05 .. 2022-03, MT5 export (prices x100, tick volume)
curl -sSL -o data/XAUUSDm15.csv \
  https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/XAUUSD/XAUUSDm15.csv
# 2026-03 .. 2026-09 rolling free sample (UTC, volume)
curl -sSL -o data/XAUUSD_15m_2026.csv \
  https://raw.githubusercontent.com/getdata-finance/xauusd-15m-ohlcv-metals-historical-data/main/XAUUSD_15m.csv
wc -l data/*.csv
