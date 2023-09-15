# dependencies
import numpy as np
import pandas as pd
from StockReturnsComputing import StockReturnsComputing

# input stock prices and market datasets
stockFileName = 'C://Users//Muham//Desktop//NASDAQ//NASDAQ//DJIAkpf1Apr2016to20193YBeta.csv'
marketFileName = 'C://Users//Muham//Desktop//NASDAQ//NASDAQ//DJIAMarketDataApr2016to20193YBeta.csv'
stockRows = 756  # excluding header
stockColumns = 15  # excluding date
marketRows = 756
marketColumns = 7

# read stock prices dataset and market dataset
dfStock = pd.read_csv(stockFileName, nrows=stockRows)
dfMarket = pd.read_csv(marketFileName, nrows=marketRows)

# extract asset labels of stocks in the portfolio
assetLabels = dfStock.columns[1:stockColumns + 1].tolist()
print('Portfolio stocks\n', assetLabels)

# extract asset prices data and market data
stockData = dfStock.iloc[0:, 1:]
marketData = dfMarket.iloc[0:, [4]]  # closing price

# compute asset returns
arrayStockData = np.asarray(stockData)
[sRows, sCols] = arrayStockData.shape
stockReturns = StockReturnsComputing(arrayStockData, sRows, sCols)

# compute market returns
arrayMarketData = np.asarray(marketData)
[mRows, mCols] = arrayMarketData.shape
marketReturns = StockReturnsComputing(arrayMarketData, mRows, mCols)

# compute betas of assets in the portfolio
beta = []
Var = np.var(marketReturns, ddof=1)
for i in range(stockColumns):
    CovarMat = np.cov(marketReturns[:, 0], stockReturns[:, i])
    Covar = CovarMat[1, 0]
    beta.append(Covar / Var)

# output betas of assets in the portfolio
print('Asset Betas:  \n')
for data in beta:
    print('{:9.3f}'.format(data))