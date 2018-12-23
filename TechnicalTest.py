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
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def momentumAo(df):
    return (ta.momentum.ao(df["high"], df["low"], l=34, fillna=True))

def momentumMFI(df):
    return (ta.momentum.money_flow_index(df["high"], df["low"], df["close"], df["volume"], n=14, fillna=True))

def momentumRSI(df):
    return (ta.momentum.rsi(df["close"], n=14, fillna=True))

def momentumStoch(df):
    return (ta.momentum.stoch(df["high"], df["low"], df["close"], n=14, fillna=True))

def momentumStochSignal(df):
    return (ta.momentum.stoch_signal(df["high"], df["low"], df["close"], n=14, d_n=3, fillna=True))

def momentumTSI(df):
    return (ta.momentum.tsi(df["close"], r=25, fillna=True))

def momentumUO(df):
    return(ta.momentum.uo(df["high"], df["low"], df["close"], s=7, m=14, ws=4.0, wm=2.0, wl=1.0, fillna=True))
    
def momentumWR(df):
    return(ta.momentum.wr(df["high"], df["low"], df["close"], lbp=14, fillna=True))
    
def volumeADI(df):
    return(ta.volume.acc_dist_index(df["high"], df["low"], df["close"], df["volume"], fillna=True))
    
def volumeCMF(df):
    return(ta.volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"], n=20, fillna=True))

def volumeEMV(df):
    return(ta.volume.ease_of_movement(df["high"], df["low"], df["close"], df["volume"], n=20, fillna=True))
    
def volumeForce(df):
    return(ta.volume.force_index(df["close"], df["volume"], n=1, fillna=True))

def volumeNVI(df):
    return(ta.volume.negative_volume_index(df["close"], df["volume"], fillna=True))
    
def volumeOBV(df):
    return(ta.volume.on_balance_volume(df["close"], df["volume"], fillna=True)) 
    
def volumeOBVMean(df):
    return(ta.volume.on_balance_volume_mean(df["close"], df["volume"], n=10, fillna=True))
    
def volumeVPT(df):
    return(ta.volume.volume_price_trend(df["close"], df["volume"], fillna=True))
    
def volatilityATR(df):
    return(ta.volatility.average_true_range(df["high"], df["low"], df["close"], n=14, fillna=True))
    
def volatilityBB(df):
    return(ta.volatility.bollinger_hband(df["close"], n=20, ndev=2, fillna=True))
    
def volatilityBBI(df):
    return(ta.volatility.bollinger_hband_indicator(df["close"], n=20, ndev=2, fillna=True))
    
def volatilityBBMA(df):
    return(ta.volatility.bollinger_mavg(df["close"], n=20, fillna=True))

#There are a few more BB indicators which could be added

def volatilityDC(df):
    return(ta.volatility.donchian_channel_hband_indicator(df["close"], n=20, fillna=True))

def volatilityKC(df):
    return(ta.volatility.keltner_channel_central(df["high"], df["low"], df["close"], n=10, fillna=True))
    
def volatilityKCH(df):
    return(ta.volatility.keltner_channel_hband(df["high"], df["low"], df["close"], n=10, fillna=True))
    
def volatilityKCL(df):
    return(ta.volatility.keltner_channel_lband(df["high"], df["low"], df["close"], n=10, fillna=True))
    
def trendADX(df):
    return (ta.trend.adx(df["high"], df["low"], df["close"], n=14, fillna=True))
    
def trendAID(df):
    return(ta.trend.aroon_down(df["close"], n=25, fillna=True))
    
def trendAIU(df):
    return(ta.trend.aroon_up(df["close"], n=25, fillna=True))
    
def trendCCI(df):
    return(ta.trend.cci(df["high"], df["low"], df["close"], n=20, c=0.015, fillna=True))    
    
def trendDPO(df):
    return(ta.trend.dpo(df["close"], n=20, fillna=True))
    
def trendEMA(df):
    return(ta.trend.ema_indicator(df["close"], n=12, fillna=True))

def trendIKH(df):
    return(ta.trend.ichimoku_a(df["high"], df["low"], n1=9, n2=26, fillna=True))    
    
def trendKST(df):
    return(ta.trend.kst(df["close"], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=True))
    
def trendMACD(df):
    return(ta.trend.macd(df["close"], n_fast=12, n_slow=26, fillna=True))

def trendMACDSig(df):
    return(ta.trend.macd_signal(df["close"], n_fast=12, n_slow=26, n_sign=9, fillna=True))

def trendMI(df):
    return(ta.trend.mass_index(df["high"], df["low"], n=9, n2=25, fillna=True))
    
def trendTRIX(df):
    return(ta.trend.trix(df["close"], n=15, fillna=True))
    
def trendVIN(df):
    return(ta.trend.vortex_indicator_neg(df["high"], df["low"], df["close"], n=14, fillna=True))

def trendVIP(df):
    return(ta.trend.vortex_indicator_pos(df["high"], df["low"], df["close"], n=14, fillna=True))
    
def dailyReturn(df):
    return(ta.others.daily_return(df["close"], fillna=True))

def dailyLogReturn(df):
    return(ta.others.daily_log_return(df["close"], fillna=True))

def getIndicators(df):
    AO = momentumAo(df).tolist()
    RSI = momentumRSI(df).tolist()
    Stoch = momentumStoch(df).tolist()
    StochSignal = momentumStochSignal(df).tolist()
    TSI = momentumTSI(df).tolist()
    ADI = volumeADI(df).tolist()
    CMF = volumeCMF(df).tolist()
    Force = volumeForce(df).tolist()
    EMV = volumeEMV(df).tolist()
    DC = volatilityDC(df).tolist()
    BB = volatilityBB(df).tolist()
    AID = trendAID(df).tolist()
    ADX = trendADX(df).tolist()
    AIU = trendAIU(df).tolist()
    CCI = trendCCI(df).tolist()
    DPO = trendDPO(df).tolist()
    EMA = trendEMA(df).tolist()
    IKH = trendIKH(df).tolist()
    KST = trendKST(df).tolist()
    MACD = trendMACD(df).tolist()
    MACDSig = trendMACDSig(df).tolist()
    MI = trendMI(df).tolist()
    TRIX = trendTRIX(df).tolist()
    VIN = trendVIN(df).tolist()
    VIP = trendVIP(df).tolist()
    dayReturn = dailyReturn(df).tolist()
    return([AO, RSI, Stoch, StochSignal, TSI, ADI, CMF, Force, EMV, DC, BB, AID, ADX, AIU, CCI, DPO, EMA, IKH, KST, MACD, MACDSig, MI, TRIX, VIN, VIP, dayReturn])

def getLabels(close, days, change):
    closes = close.tolist()
    labels = []
    for i in range(0, len(closes)-days):
            gain = (closes[i+days]-closes[i])/closes[i]
            if(gain > 1 + change):
                labels.append(1)
            else:
                labels.append(0)
    return labels

def main():
    start = datetime(2014, 1, 1)
    end = datetime(2018, 1, 1)
    days = 60
    change = .04

    trainFeatures = []
    trainLabels = []
    testFeatures = []
    testLabels = []

    with open('ticker_sectors.data', 'rb') as f:
    		tickerSectors = pickle.load(f)
    companies = tickerSectors[0]
    sectors = tickerSectors[1]
    
    for i in range(0, len(companies)):
        company = companies[i]
        sector = sectors[i]
        print(company)
        print(sector)

        if(sector == "XLC"):
            sector = "XLK"

        df = get_historical_data(company, start, end, output_format="pandas")

        if(df.shape[0] < 800):
            continue;

        compIndicators = getIndicators(df)

        dfSec = get_historical_data(sector, start, end, output_format="pandas")

        secIndicators = getIndicators(dfSec)

        dfMarket = get_historical_data("SPY", start, end, output_format="pandas")

        marketIndicators = getIndicators(dfMarket)

        indicators = compIndicators + secIndicators + marketIndicators

        n = len(indicators[0])

        compFeatures = np.array(indicators).T
        compFeatures = compFeatures[30:n-days]
        compLabels = getLabels(df["close"], days, change)
        compLabels = np.asarray(compLabels)
        compLabels = compLabels[30:]

        for i in range(0, 700):
            trainFeatures.append(compFeatures[i])
            trainLabels.append(compLabels[i])
        for i in range(730, len(compFeatures)):
            testFeatures.append(compFeatures[i])
            testLabels.append(compLabels[i])


    rf = RandomForestRegressor(n_estimators=65, random_state = 42)
    rf.fit(trainFeatures, trainLabels)

    predictions = rf.predict(testFeatures)
    for i in range(len(predictions)):
        if(predictions[i] > .5):
            predictions[i] = 1
        else:
            predictions[i] = 0
            
    target_names = ['0', '1']

    print(classification_report(testLabels, predictions, target_names=target_names))
    confusionMatrix = confusion_matrix(testLabels, predictions)
    print('Confusion Matrix: ', confusionMatrix)







if __name__ == "__main__":
	print("Building Model")
	try:
		main()
	except KeyboardInterrupt:
		print("Ctrl+C pressed. Stopping...")
    
