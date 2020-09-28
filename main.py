"""
時系列解析
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']
import warnings
warnings.filterwarnings('ignore')

import math

def plot_acf(acf):
    """ ACFをプロットする """
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(111)
    sm.graphics.tsa.plot_acf(acf, lags=40, ax=ax1)


def sin_acf_func():
    """ sin波の自己相関 """
    sin_wav = [math.sin(x) for x in range(100)]
    #plt.plot([x for x in range(50)], sin_wav)   
    
    sin_acf = sm.tsa.stattools.acf(sin_wav, nlags=40)
    plot_acf(sin_acf)
    
    
def rand_acf_func():
    """ ランダム波形の自己相関 """
    rand_wav = np.random.random(100)    
    print(rand_wav)    
    rand_acf = sm.tsa.stattools.acf(rand_wav, nlags=40)
    plot_acf(rand_acf)

def visualize_timeseries_components(ts, freq):
    """ 時系列の可視化 """
    # オリジナル ->トレンド成分、季節成分、残差成分に分解してプロット
    #"""
    # 通常スケール
    decom =  sm.tsa.seasonal_decompose(ts, freq=freq) # 7日周期なので
    trend = decom.trend
    seasonal = decom.seasonal
    residual = decom.resid
    plt.figure(figsize=(8, 8))    
    # オリジナルの時系列データプロット
    plt.subplot(411)
    plt.plot(pcr_ts)
    plt.ylabel('Original')    
    # トレンド成分（trend） のプロット
    plt.subplot(412)
    plt.plot(trend)
    plt.ylabel('Trend')    
    # 季節成分（seasonal） のプロット
    plt.subplot(413)
    plt.plot(seasonal)
    plt.ylabel('Seasonality')    
    # 残渣成分（residual） のプロット
    plt.subplot(414)
    plt.plot(residual)
    plt.ylabel('Residuals')  
    #"""
    
    """
    # Logスケール
    ts_log = np.log(ts)
    res_log = sm.tsa.seasonal_decompose(ts_log, freq=freq)
    trend_log = res_log.trend
    seasonal_log = res_log.seasonal
    residual_log = res_log.resid
    plt.figure(figsize=(8, 8))
    # オリジナルlog変換
    plt.subplot(411)
    plt.plot(ts_log, c='b')
    plt.ylabel('Original')
    # trend
    plt.subplot(412)
    plt.plot(trend_log, c='b')
    plt.ylabel('Trend')
    # seasonal
    plt.subplot(413)
    plt.plot(seasonal_log, c='b')
    plt.ylabel('Seasonality')
    # residual
    plt.subplot(414)
    plt.plot(residual_log, c='b')
    plt.ylabel('Residuals')
    plt.tight_layout()
    """

if __name__ == '__main__':
    covid_data = pd.read_excel(r'D:\DataSet\TimeSeriesAnalysis\pcr_positive_fromJuly.xlsx', index=False)
    header = covid_data.columns 
    
    # 可視化    
    #print(covid_data[header[1]])    
    #plt.plot(covid_data[header[1]]) #なぜかめっちゃ重い
    
    #sin_acf_func()
    #rand_acf_func()
    
    # 1日毎の感染者数リスト
    pcr_ts = pd.Series(covid_data[header[1]], dtype='float') 
    
    # まずはローデータの自己相関がどんなもんか
    pcr_acf = sm.tsa.stattools.acf(pcr_ts, nlags=40)
    #print(pcr_acf)
    #plot_acf(pcr_acf)
        
    # 階差系列（日ごとの差分）　トレンド成分の除去でACF計算
    # --> 7日毎に強い相関　が見られた　周期=7日
    pcr_diff = pcr_ts - pcr_ts.shift()
    pcr_diff = pcr_diff.dropna()
    #print(pcr_diff.to_list())    
    #plt.plot(pcr_diff.to_list())    
    pcr_diff_acf = sm.tsa.stattools.acf(pcr_diff, nlags=40)
    #plot_acf(pcr_diff_acf)
    
    # 各成分に分解、可視化する
    #visualize_timeseries_components(pcr_ts, 7)
    
    # SARIMAモデルを生成
    """
    モデル変数名 = sm.tsa.SARIMAX(
            ts_data, order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity = False,
            enforce_invertibility = False).fit()
    p,P: ARモデルの次数
    d,D: 差分の次数
    q,Q: MAモデルの次数
    s: 季節周期
    
    sは自己相関係数などから導出
    p,d,qはうまくフィッティングできるようパラメータ探索
    """
    srimax = sm.tsa.SARIMAX(pcr_ts, order=(2,1,3),
                            seasonal_order=(1,1,3,7),
                            enforce_stationarity = False,
                            enforce_invertibility = False
                           ).fit()
    
    # 未来の予測
    pred = srimax.forecast(100) 
    #print(pred[:])
    
    pred_non_neg = pred.where(pred>0, 0) # 負数を0に置き換えた
    #print(pred_non_neg)
    
    plt.figure(figsize=(10, 5))
    plt.plot(pcr_ts, label='Actual')
    #plt.plot(pred, label='Prediction', linestyle='--')
    plt.plot(pred_non_neg, label='Prediction', linestyle='--')
    plt.legend(loc='best')
    
    
    """
    左の数値は7/1から数えての日数　ピークの日付をプロット
    
    87     495.841601
    94     404.510307
    101    334.913523
    108    242.384172
    115    170.365795
    122     80.193023
    129      6.080464
    136    -82.239395
    143   -157.991100
    150   -244.860927
    157   -321.895368
    164   -407.630452
    171   -485.668718
    178   -570.515793
    185   -649.339615
    
    今回の検証ではピークが負になるのが大体11/6～11/13頃
    
    """
    
    """
    # データ予測（期間指定）
    predict_fromTo = srimax.predict(60, 109)    
    # 実データと予測データのグラフ描画
    plt.figure(figsize=(10, 5))
    plt.plot(pcr_ts, label='Actual')
    plt.plot(predict_fromTo, label='Prediction', alpha=0.5)
    plt.legend(loc='best')
    """
    