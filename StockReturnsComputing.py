# Python code to compute the daily returns in percentage, of Dow Stocks listed in Sec.1.1
# calls function StockReturnsComputing to compute asset returns

# dependencies
import numpy as np
import pandas as pd

from InputFile import yfinance_veri_cek

# input stock prices dataset
semboller = ['THYAO.IS', 'GARAN.IS', 'AAPL', 'AMZN', 'CRM']
baslangic_tarih = '2023-01-01'
bitis_tarih = '2023-08-01'
frekans = '1mo'

veriler_df = yfinance_veri_cek(
    sembol=semboller,
    baslangic_tarih=baslangic_tarih,
    bitis_tarih=bitis_tarih,
    frekans=frekans
)

# read stock prices
df = veriler_df
# extract asset labels
assetLabels = df.columns
print('Asset labels of k-portfolio 1: \n', assetLabels)

# extract asset labels
assetLabels = df.values
print(assetLabels)

# extract asset prices data
stockPrice = df.iloc[0:, 1:]
print(stockPrice.shape)

# print stock price
print(stockPrice)


# function to compute asset returns
def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100

    return StockReturn


stockPriceArray = np.asarray(stockPrice)
[Rows, Cols] = stockPriceArray.shape
stockReturns = StockReturnsComputing(stockPriceArray, Rows, Cols)
print('Daily returns of selective Dow 30 stocks\n', stockReturns)

meanReturns = np.mean(stockReturns, axis=0)
print('Mean returns of Dow Stocks:\n', meanReturns)
covReturns = np.cov(stockReturns, rowvar=False)
print('Variance-covariance matrix of returns of Dow Stocks:\n')
print(covReturns)
