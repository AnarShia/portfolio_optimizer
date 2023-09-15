def EmptyRowsElimination(dfAssetPrices):
    # read dataset and extract its dimensions
    [Rows, Columns] = dfAssetPrices.shape
    dFrame = dfAssetPrices.iloc[0:Rows, 0:Columns]

    # call dropna method from Pandas
    dFClean = dFrame.dropna(axis=0, how='all')
    return dFClean




#empty rows elimination from stock prices dataset

#dependencies
import numpy as np
import pandas as pd

#input dataset and dimensions of the dataset
StockFileName = 'C://Users//Muham//Desktop//NASDAQ//NASDAQ//DJIAkpf1Apr2016to20193YBeta.csv'
Rows = 12      #excluding headers
Columns = 18  #excluding date

#read stock prices
df = pd.read_csv(StockFileName,  nrows= Rows)

#extract asset Names
assetNames = df.columns[1:Columns+1].tolist()
print(assetNames)

#clean the stock dataset of empty rows
StockData = df.iloc[0:, 1:]
dfClean = EmptyRowsElimination(StockData)
print('\nData cleaning completed!')
[rows, cols]=dfClean.shape
print('Dimensions of the cleaned dataset', dfClean.shape)
print('Cleaned dataset: \n', dfClean)



