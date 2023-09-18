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
Rows = 143  # excluding header
Columns = 10  # excluding date

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

# function to handle bi-criterion portfolio optimization with constraints

# dependencies
import numpy as np
from scipy import optimize


def BiCriterionFunctionOptmzn(MeanReturns, CovarReturns, RiskAversParam, PortfolioSize):
    def f(x, MeanReturns, CovarReturns, RiskAversParam, PortfolioSize):
        PortfolioVariance = np.matmul(np.matmul(x, CovarReturns), x.T)
        PortfolioExpReturn = np.matmul(np.array(MeanReturns), x.T)
        func = RiskAversParam * PortfolioVariance - (1 - RiskAversParam) * PortfolioExpReturn
        return func

    def ConstraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T) - b
        return constraintVal

    def ConstraintIneqUpBounds(x):
        A = [[0, 0, 0, 0, 0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 1, 0, 0, 1]]
        bUpBounds = np.array([0.6, 0.4]).T
        constraintValUpBounds = bUpBounds - np.matmul(A, x.T)
        return constraintValUpBounds

    def ConstraintIneqLowBounds(x):
        A = [[0, 0, 0, 0, 0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 1, 0, 0, 1]]
        bLowBounds = np.array([0.01, 0.01]).T
        constraintValLowBounds = np.matmul(A, x.T) - bLowBounds
        return constraintValLowBounds

    xinit = np.repeat(0.01, PortfolioSize)
    cons = ({'type': 'eq', 'fun': ConstraintEq},
            {'type': 'ineq', 'fun': ConstraintIneqUpBounds},
            {'type': 'ineq', 'fun': ConstraintIneqLowBounds})
    bnds = [(0, 0.1), (0, 0.1), (0, 0.1), (0, 0.1), (0, 0.1), (0, 1), (0, 0.1), (0, 1), (0, 1), (0, 0.1)]

    opt = optimize.minimize(f, x0=xinit, args=(MeanReturns, CovarReturns,
                                               RiskAversParam, PortfolioSize),
                            method='SLSQP', bounds=bnds, constraints=cons,
                            tol=10 ** -3)
    print(opt)
    return opt


# obtain optimal portfolios for the constrained portfolio optimization model
# Maximize returns and Minimize risk with fully invested, bound and
# class constraints

# set portfolio size
portfolioSize = Columns

# initialization
xOptimal = []
minRiskPoint = []
expPortfolioReturnPoint = []

for points in range(0, 60):
    riskAversParam = points / 60.0
    result = BiCriterionFunctionOptmzn(meanReturns, covReturns, riskAversParam, \
                                       portfolioSize)
    xOptimal.append(result.x)

# compute annualized risk and return  of the optimal portfolios for trading days = 251
xOptimalArray = np.array(xOptimal)
minRiskPoint = np.diagonal(np.matmul((np.matmul(xOptimalArray, covReturns)), \
                                     np.transpose(xOptimalArray)))
riskPoint = np.sqrt(minRiskPoint * 251)
expPortfolioReturnPoint = np.matmul(xOptimalArray, meanReturns)
retPoint = 251 * np.array(expPortfolioReturnPoint)

# set precision for printing results
np.set_printoptions(precision=3, suppress=True)

# display optimal portfolio results
print("Optimal weights of the efficient set portfolios\n:", xOptimalArray)
print("\nAnnualized Risk and Return of the efficient set portfolios:\n", np.c_[riskPoint, retPoint])

import matplotlib.pyplot as plt

# Graph Efficient Frontier for the constrained portfolio model
NoPoints = riskPoint.size

colours = "blue"
area = np.pi * 3

plt.title('Efficient Frontier for constrained k-portfolio 1 of Dow stocks')
plt.xlabel('Annualized Risk(%)')
plt.ylabel('Annualized Expected Portfolio Return(%)')
plt.scatter(riskPoint, retPoint, s=area, c=colours, alpha=0.5)
plt.show()
