# filling missing values of stock prices dataset
def FillMissingValues(StockPrices):
    import numpy as np
    print('Fill missing values...')

    # identify positions of the missing values in StockPrices
    [rows, cols] = np.where(np.asarray(np.isnan(StockPrices)))

    # replace missing value with the previous day's price
    for t in range(rows.size):
        i = rows[t]
        j = cols[t]

        if i == 0:
            firstItem = StockPrices.iloc[i, j]
            if firstItem == np.nan:
                # find first non-missing value
                k = 1
                while np.isnan(firstItem):
                    firstItem = StockPrices.iloc[k, j]
                    k = k + 1
                StockPrices.iloc[i, j] = firstItem

        if (i - 1) >= 0:
            StockPrices.iloc[i, j] = StockPrices.iloc[i - 1, j].copy()
        else:
            print('error')

    return StockPrices
