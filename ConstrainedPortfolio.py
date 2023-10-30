# Dependencies
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from InputFile import yfinance_veri_cek


# function computes asset returns
def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100

    return StockReturn


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
        A = [
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
             0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
             0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
             1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
             1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1]]
        # repeat x1 and x2 for each stock (20 times)

        bUpBounds = np.array([0.6, 0.4]).T
        Axt = np.matmul(A, x.T)
        constraintValUpBounds = bUpBounds - Axt
        return constraintValUpBounds

    def ConstraintIneqLowBounds(x):
        A = [
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
             0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
             0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
             1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
             1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1]]
        bLowBounds = np.array([0.01, 0.01]).T
        Axt = np.matmul(A, x.T)
        constraintValLowBounds = Axt - bLowBounds
        return constraintValLowBounds

    xinit = np.repeat(0.01, PortfolioSize)
    cons = ({'type': 'eq', 'fun': ConstraintEq},
            {'type': 'ineq', 'fun': ConstraintIneqUpBounds},
            {'type': 'ineq', 'fun': ConstraintIneqLowBounds})
    bnds = tuple([(0, 1) for x in xinit])

    opt = optimize.minimize(f, x0=xinit, args=(MeanReturns, CovarReturns,
                                               RiskAversParam, PortfolioSize),
                            method='SLSQP', bounds=bnds, constraints=cons,
                            tol=10 ** -3)
    print(opt)
    return opt


# input k portfolio 1 dataset comprising 15 Dow stocks
Rows = 143  # excluding header
Columns = 100  # excluding date

semboller = ['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN',
          'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANSS', 'ASML',
          'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CDNS',
          'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP',
          'CSX', 'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EBAY', 'ENPH', 'EXC', 'FANG', 'FAST', 'FTNT', 'GEHC',
          'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC',
          'LCID', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX',
          'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'ROST',
          'SBUX', 'SGEN', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TTD', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY',
          'XEL', 'ZM']
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

# set portfolio size
portfolioSize = Columns

# initialization
xOptimal = []
minRiskPoint = []
expPortfolioReturnPoint = []

for points in range(0, 60):
    riskAversParam = points / 60.0
    result = BiCriterionFunctionOptmzn(meanReturns, covReturns, riskAversParam, portfolioSize)
    xOptimal.append(result.x)

# compute annualized risk and return  of the optimal portfolios for trading days = 251
xOptimalArray = np.array(xOptimal)
x = (np.matmul(xOptimalArray, covReturns))
arrayTranspose = np.transpose(xOptimalArray)
arrayMultiply = np.matmul(x, arrayTranspose)
minRiskPoint = np.diagonal(arrayMultiply)
riskPoint = np.sqrt(minRiskPoint * 251)
expPortfolioReturnPoint = np.matmul(xOptimalArray, meanReturns)
retPoint = 251 * np.array(expPortfolioReturnPoint)

# set precision for printing results
np.set_printoptions(precision=3, suppress=True)

# display optimal portfolio results
print("Optimal weights of the efficient set portfolios\n:", xOptimalArray)
print("\nAnnualized Risk and Return of the efficient set portfolios:\n", np.c_[riskPoint, retPoint])

# Graph Efficient Frontier for the constrained portfolio model
NoPoints = riskPoint.size

colours = "blue"
area = np.pi * 3

plt.title('Efficient Frontier for constrained k-portfolio 1 of Dow stocks')
plt.xlabel('Annualized Risk(%)')
plt.ylabel('Annualized Expected Portfolio Return(%)')
plt.scatter(riskPoint, retPoint, s=area, c=colours, alpha=0.5)
plt.show()
