import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from InputFile import yfinance_veri_cek


def StockReturnsComputing(StockPrice, Rows, Columns):
    import numpy as np

    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j])

    return StockReturn



semboller = ["AAPL", "INTC", "MSFT", "GOOGL", "AMZN", "PYPL", "META", "NFLX", "NVDA", "TSLA"]
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
clusters = 4

# extract stock prices excluding header and trading dates
dfStockPrices = df.values

# store stock prices as an array
arStockPrices = np.asarray(dfStockPrices)
[rows, cols] = arStockPrices.shape
print(rows, cols)
print(arStockPrices)

# compute daily returns of all stocks in the mini universe
arReturns = StockReturnsComputing(arStockPrices, rows, cols)
print('Size of the array of daily returns of stocks:\n', arReturns.shape)
print('Array of daily returns of stocks\n', arReturns)

# compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis=0)
print('Mean returns:\n', meanReturns)
covReturns = np.cov(arReturns, rowvar=False)
# set precision for printing results
np.set_printoptions(precision=5, suppress=True)
print('Size of Variance-Covariance matrix of returns:\n', covReturns.shape)
print('Variance-Covariance matrix of returns:\n', covReturns)

# prepare asset parameters for k-means clustering
# reshape for concatenation
meanReturns = meanReturns.reshape(len(meanReturns), 1)
assetParameters = np.concatenate([meanReturns, covReturns], axis=1)
print('Size of the asset parameters for clustering:\n', assetParameters.shape)
print('Asset parameters for clustering:\n', assetParameters)

# kmeans clustering of assets using the characteristic vector of
# mean return and variance-covariance vector of returns

assetsCluster = KMeans(algorithm='lloyd', max_iter=600, n_clusters=clusters, n_init='auto')
print('Clustering of assets completed!')
assetsCluster.fit(assetParameters)
centroids = assetsCluster.cluster_centers_
labels = assetsCluster.labels_

print('Centroids:\n', centroids)
print('Labels:\n', labels)

# fixing asset labels to cluster points
print('Stocks in each of the clusters:\n', )
assets = np.array(assetLabels)
for i in range(clusters):
    print('Cluster', i + 1)
    clt = np.where(labels == i)
    assetsCluster = assets[clt]
    print(assetsCluster)
