#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:23:16 2018

@author: ckardatzke
"""

import ta
import iexfinance
from iexfinance.stocks import get_historical_data
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import pickle

def momentumAo(df):
    return (ta.momentum.ao(df["high"], df["low"], l=34, fillna=False))

def momentumMFI(df):
    return (ta.momentum.money_flow_index(df["high"], df["low"], df["close"], df["volume"], n=14, fillna=False))

def momentumRSI(df):
    return (ta.momentum.rsi(df["close"], n=14, fillna=False))

def momentumStoch(df):
    return (ta.momentum.stoch(df["high"], df["low"], df["close"], n=14, fillna=False))

def momentumStochSignal(df):
    return (ta.momentum.stoch_signal(df["high"], df["low"], df["close"], n=14, d_n=3, fillna=False))

def momentumTSI(df):
    return (ta.momentum.tsi(df["close"], r=25, fillna=False))

def momentumUO(df):
    return(ta.momentum.uo(df["high"], df["low"], df["close"], s=7, m=14, ws=4.0, wm=2.0, wl=1.0, fillna=False))
    
def momentumWR(df):
    return(ta.momentum.wr(df["high"], df["low"], df["close"], lbp=14, fillna=False))
    
def volumeADI(df):
    return(ta.volume.acc_dist_index(df["high"], df["low"], df["close"], df["volume"], fillna=False))
    
def volumeCMF(df):
    return(ta.volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"], n=20, fillna=False))

def volumeEMV(df):
    return(ta.volume.ease_of_movement(df["high"], df["low"], df["close"], df["volume"], n=20, fillna=False))
    
def volumeForce(df):
    return(ta.volume.force_index(df["close"], df["volume"], n=1, fillna=False))

def volumeNVI(df):
    return(ta.volume.negative_volume_index(df["close"], df["volume"], fillna=False))
    
def volumeOBV(df):
    return(ta.volume.on_balance_volume(df["close"], df["volume"], fillna=False)) 
    
def volumeOBVMean(df):
    return(ta.volume.on_balance_volume_mean(df["close"], df["volume"], n=10, fillna=False))
    
def volumeVPT(df):
    return(ta.volume.volume_price_trend(df["close"], df["volume"], fillna=False))
    
def volatilityATR(df):
    return(ta.volatility.average_true_range(df["high"], df["low"], df["close"], n=14, fillna=False))
    
def volatilityBB(df):
    return(ta.volatility.bollinger_hband(df["close"], n=20, ndev=2, fillna=False))
    
def volatilityBBI(df):
    return(ta.volatility.bollinger_hband_indicator(df["close"], n=20, ndev=2, fillna=False))
    
def volatilityBBMA(df):
    return(ta.volatility.bollinger_mavg(df["close"], n=20, fillna=False))

#There are a few more BB indicators which could be added

def volatilityDC(df):
    return(ta.volatility.donchian_channel_hband_indicator(df["close"], n=20, fillna=False))

def volatilityKC(df):
    return(ta.volatility.keltner_channel_central(df["high"], df["low"], df["close"], n=10, fillna=False))
    
def volatilityKCH(df):
    return(ta.volatility.keltner_channel_hband(df["high"], df["low"], df["close"], n=10, fillna=False))
    
def volatilityKCL(df):
    return(ta.volatility.keltner_channel_lband(df["high"], df["low"], df["close"], n=10, fillna=False))
    
def trendADX(df):
    return (ta.trend.adx(df["high"], df["low"], df["close"], n=14, fillna=False))
    
def trendAID(df):
    return(ta.trend.aroon_down(df["close"], n=25, fillna=False))
    
def trendAIU(df):
    return(ta.trend.aroon_up(df["close"], n=25, fillna=False))
    
def trendCCI(df):
    return(ta.trend.cci(df["high"], df["low"], df["close"], n=20, c=0.015, fillna=False))    
    
def trendDPO(df):
    return(ta.trend.dpo(df["close"], n=20, fillna=False))
    
def trendEMA(df):
    return(ta.trend.ema_indicator(df["close"], n=12, fillna=False))

def trendIKH(df):
    return(ta.trend.ichimoku_a(df["high"], df["low"], n1=9, n2=26, fillna=False))    
    
def trendKST(df):
    return(ta.trend.kst(df["close"], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False))
    
def trendMACD(df):
    return(ta.trend.macd(df["close"], n_fast=12, n_slow=26, fillna=False))

def trendMACDSig(df):
    return(ta.trend.macd_signal(df["close"], n_fast=12, n_slow=26, n_sign=9, fillna=False))

def trendMI(df):
    return(ta.trend.mass_index(df["high"], df["low"], n=9, n2=25, fillna=False))
    
def trendTRIX(df):
    return(ta.trend.trix(df["close"], n=15, fillna=False))
    
def trendVIN(df):
    return(ta.trend.vortex_indicator_neg(df["high"], df["low"], df["close"], n=14, fillna=False))

def trendVIP(df):
    return(ta.trend.vortex_indicator_pos(df["high"], df["low"], df["close"], n=14, fillna=False))
    
def dailyReturn(df):
    return(ta.others.daily_return(df["close"], fillna=False))

def dailyLogReturn(df):
    return(ta.others.daily_log_return(df["close"], fillna=False))

def getIndicators(df):
    return([dailyReturn(df).tolist(), dailyLogReturn(df).tolist()])



def main():
    start = datetime(2014, 1, 1)
    end = datetime(2018, 1, 1)

    with open('ticker_sectors.data', 'rb') as f:
    		tickerSectors = pickle.load(f)
    companies = tickerSectors[0]
    sectors = tickerSectors[1]
    
    for i in range(0, len(companies)):
        company = companies[i]
        sector = sectors[i]
        print(company)

        df = get_historical_data(company, start, end, output_format="pandas")
        high = df["high"]
        low = df["low"]
        close = df["close"]
        op = df["open"]
        volume = df["volume"]

        print(getIndicators(df))

        dfSec = get_historical_data(sector, start, end, output_format="pandas")
        highSec = dfSec["high"]
        lowSec = dfSec["low"]
        closeSec = dfSec["close"]
        opSec = dfSec["open"]
        volumeSec = dfSec["volume"]

        dfMarket = get_historical_data("SPY", start, end, output_format="pandas")
        highMarket = dfMarket["high"]
        lowMarket = dfMarket["low"]
        closeMarket = dfMarket["close"]
        opMarket = dfMarket["open"]
        volumeMarket = dfMarket["volume"]




if __name__ == "__main__":
	print("Building Model")
	try:
		main()
	except KeyboardInterrupt:
		print("Ctrl+C pressed. Stopping...")
    