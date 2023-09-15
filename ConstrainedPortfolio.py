# Dependencies
import numpy as np
import pandas as pd

from InputFile import yfinance_veri_cek


# function computes asset returns
def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100

    return StockReturn


# input k portfolio 1 dataset  comprising 15 Dow stocks and DJIA market dataset
# over a 3 Year period (April 2016 to April 2019)
stockFileName = 'C://Users//Muham//Desktop//NASDAQ//NASDAQ//DJIAkpf1Apr2016to20193YBeta.csv'
marketFileName = 'C://Users//Muham//Desktop//NASDAQ//NASDAQ//DJIAMarketDataApr2016to20193YBeta.csv'
stockRows = 756  # excluding header of stock dataset
stockColumns = 15  # excluding date of stock dataset
marketRows = 756  # excluding header of market dataset
marketColumns = 7  # excluding date of market dataset

# read stock prices and closing prices of market data (column index 4),  into dataframes
dfStock = pd.read_csv(stockFileName, nrows=stockRows)
dfMarket = pd.read_csv(marketFileName, nrows=marketRows)
stockData = dfStock.iloc[0:, 1:]
marketData = dfMarket.iloc[0:, [4]]

# extract asset labels in the portfolio
assetLabels = dfStock.columns[1:stockColumns + 1].tolist()
print('Asset labels of k-portfolio 1: \n', assetLabels)

# compute asset returns
arStockPrices = np.asarray(stockData)
[sRows, sCols] = arStockPrices.shape
arStockReturns = StockReturnsComputing(arStockPrices, sRows, sCols)

# compute market returns
arMarketPrices = np.asarray(marketData)
[mRows, mCols] = arMarketPrices.shape
arMarketReturns = StockReturnsComputing(arMarketPrices, mRows, mCols)

# compute betas of the assets in k-portfolio 1
beta = []
Var = np.var(arMarketReturns, ddof=1)
for i in range(stockColumns):
    CovarMat = np.cov(arMarketReturns[:, 0], arStockReturns[:, i])
    Covar = CovarMat[1, 0]
    beta.append(Covar / Var)

# display results
print('Asset Betas:\n')
for data in beta:
    print('{:9.3f}'.format(data))

# obtain mean returns and variance-covariance matrix of returns of k-portfolio 1
# historical dataset: DJIA Index April 2014 to April 2019

# Dependencies
import numpy as np
import pandas as pd

# input k portfolio 1 dataset comprising 15 Dow stocks
Rows = 1259  # excluding header
Columns = 15  # excluding date

semboller = ["AAPL", "INTC", "MSFT", "GOOGL", "AMZN", "PYPL", "META", "NFLX", "NVDA", "TSLA"]
baslangic_tarih = '2023-01-01'
bitis_tarih = '2023-08-01'
frekans = '1d'

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

# extract the asset prices data
stockData = df.values

# compute asset returns
arStockPrices = np.asarray(stockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

# set precision for printing data
np.set_printoptions(precision=3, suppress=True)

# compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)
print('\nMean Returns:\n', meanReturns)
print('\nVariance-Covariance Matrix of Returns:\n', covReturns)
