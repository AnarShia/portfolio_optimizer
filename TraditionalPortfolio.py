# read k portfolio 1 dataset comprising 15 stocks

# dependencies
import numpy as np
import pandas as pd

from InputFile import yfinance_veri_cek


# function to compute stock returns
def StockReturnsComputing(StockPrice, Rows, Columns):

    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j])

    return StockReturn

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

# store stock prices as an array
arStockPrices = np.asarray(df.values)
[rows, cols] = arStockPrices.shape
print('k-portfolio 1 dataset size:\n', rows, cols)
print('k-portfolio 1 stock prices:\n', arStockPrices)





# compute asset returns
arReturns = StockReturnsComputing(arStockPrices, rows, cols)
print('k-portfolio 1 returns:\n', arReturns)

# compute mean returns and variance covariance matrix of returns

# set precision for printing results
np.set_printoptions(precision=5, suppress=True)

meanReturns = np.mean(arReturns, axis=0)
print('Mean returns of k-portfolio 1:\n', meanReturns)
covReturns = np.cov(arReturns, rowvar=False)
print('\nVariance-Covariance matrix of returns of k-portfolio 1: \n')
print('Size  ', covReturns.shape, '\n', covReturns)

[Rows,Cols]=arReturns.shape
#equal weighted portfolio construction: Annualized risk and
#expected annualized portfolio return
#trading days = 251
PortfolioSize = Cols
EqualWeightVector = np.ones((1,PortfolioSize))*(1.0/PortfolioSize)
RiskMatrixParam=(np.matmul(EqualWeightVector,covReturns))
RiskMatrix=np.matmul(RiskMatrixParam, \
                     np.transpose(EqualWeightVector))
EqWgtPortfolioRisk = np.sqrt(RiskMatrix)
EqWgtAnnPortfolioRisk = EqWgtPortfolioRisk*np.sqrt(251)*100
EqWgtPortfolioReturn = np.matmul(EqualWeightVector, np.transpose(meanReturns))
EqWgtAnnPortfolioReturn = 251*EqWgtPortfolioReturn * 100

print("Annualized Portfolio Risk :  %4.2f" % EqWgtAnnPortfolioRisk.item(), "%")
print("\nAnnualized Expected Portfolio Return:  %4.2f" %  EqWgtAnnPortfolioReturn.item(),"%")

# Equal weighted portfolio: Diversification Ratio
EqWgtPortfolioAssetStdDev = np.sqrt(np.diagonal(covReturns))
EqWgtPortfolioDivRatio = np.sum(np.multiply(EqWgtPortfolioAssetStdDev, EqualWeightVector)) \
                         / EqWgtPortfolioRisk
print("\n Equal Weighted Portfolio:Diversification Ratio  %4.2f" % EqWgtPortfolioDivRatio.item())



#Inverse volatility weighted portfolio construction: Annualized risk and
#Expected annualized portfolio return
#Trading days = 251
InvVolWeightAssets_Risk = np.sqrt(np.diagonal(covReturns))
InvVolWeightAssets_ReciprocalRisk = 1.0/InvVolWeightAssets_Risk
InvVolWeightAssets_ReciprocalRisk_Sum = np.sum(InvVolWeightAssets_ReciprocalRisk)
InvVolWeightAssets_Weights = InvVolWeightAssets_ReciprocalRisk / \
                             InvVolWeightAssets_ReciprocalRisk_Sum
InvVolWeightPortfolio_Risk = np.sqrt(np.matmul((np.matmul(InvVolWeightAssets_Weights,\
                             covReturns)), np.transpose(InvVolWeightAssets_Weights)))

#annualized risk and return
InvVolWeightPortfolio_AnnRisk = np.sqrt(251)* InvVolWeightPortfolio_Risk *100
InvVolWeightPortfolio_AnnReturn = 251* np.matmul(InvVolWeightAssets_Weights,\
                                  np.transpose(meanReturns)) *100

print("Annualized Portfolio Risk: %4.2f" % InvVolWeightPortfolio_AnnRisk,"%\n")
print("Annualized Expected Portfolio Return: %4.2f" % InvVolWeightPortfolio_AnnReturn,"%")