import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.optimize import minimize
from datetime import datetime
import os
import yfinance as yf
yf.pdr_override()


print("burada başladı")
def get_portfolio_variance(weights):
  return weights.dot(cov).dot(weights)

def target_return_constraint(weights, target):
  return weights.dot(mean_return) - target

def portfolio_constraint(weights):
  return weights.sum() - 1

def neg_sharpe_ratio(weights):
  mean = weights.dot(mean_return)
  sd = np.sqrt(weights.dot(cov).dot(weights))
  return - (mean - risk_free_rate) / sd

def percentOfGain(x1, x2):
  return (x2 - x1) * 100 / x1

def neg_sharpe_ratio(weights):
  mean = weights.dot(mean_return)
  sd = np.sqrt(weights.dot(cov).dot(weights))
  return -(mean - risk_free_rate) / sd

def getSharpeRatioStocks(prices, dates, stocks, startDate, path, st_weight):
  global risk_free_rate
  global cov
  global mean_return
  returns = pd.DataFrame(index=dates[1:])
  for stock in stocks:
    current_returns = prices[stock].pct_change()
    returns[stock] = current_returns.iloc[1:] * 100

  mean_return = returns.mean()
  cov = returns.cov()
  cov_np = cov.to_numpy()
  date = datetime.strptime(str(startDate), '%Y-%m-%d %H:%M:%S').date()
  risk_free_rate = getRiskFreeRate(path, date) / 252

  D = len(mean_return)
  bounds = [(0, 1)]*D

  res = minimize(
    fun=neg_sharpe_ratio,
    x0=np.ones(D) / D, # uniform
    # args=args,
    method='SLSQP',
    constraints={
      'type': 'eq',
      'fun': portfolio_constraint,
    },
    bounds=bounds,
  )
  best_sr, sr_w = -res.fun, res.x
  sr_stocks = prices.iloc[:, sr_w > st_weight].columns
  return sr_stocks


def createDataset(stocks, start, end):
  stockData = pdr.get_data_yahoo(stocks, start=start, end=end)['Close']
  if stockData.isna().sum().sum() > 0:
    stockData.fillna(method='ffill', inplace=True)
  return stockData

def getRiskFreeRate(path, date):
  rfr_df = pd.read_csv(path, index_col='DATE', parse_dates=True)
  month = date.month
  year = date.year
  rfr_date = datetime(year,month,1)
  rfr = rfr_df.loc[rfr_date]["TB3MS"]
  return rfr

def getStockCloses(data, start, end):
  return data.iloc[start : end]

def getOptStocks(close_prices, all_dates, start, end, startDate, path, st_weight):
  prices = getStockCloses(close_prices, start, end)
  dates = all_dates[start : end]

  optStocks = getSharpeRatioStocks(prices, dates, stocks, startDate, path, st_weight)
  return optStocks

def trainPortfolio(prices, dates, opt_type, stocks):
  global cov
  global mean_return
  # dates = all_dates[start : end]
  returns = pd.DataFrame(index=dates[1:])

  print(stocks)

  for stock in stocks:
    current_returns = prices[stock].pct_change() # her bir hisse senedi için günlük getiriler (percentage change)
    returns[stock] = current_returns.iloc[1:] * 100 # yüzdelik olarak değişimi tutar
    # İlk değer (NaN) hariç tüm değerleri almak için iloc[1:] kullanılır.

  mean_return = returns.mean() # Her bir hisse senedi için ortalama getiriyi hesaplar. (hisse bazında beklenen getiri)
  mean_return.replace([np.inf, -np.inf], 0, inplace=True)

  cov = returns.cov() # Hisseler arasındaki getiri kovaryans matrisini hesaplar.
  cov_np = cov.to_numpy() # Kovaryans matrisini numpy dizisine dönüştürür.

  D = len(mean_return) # toplam hisse sayısı
  A_eq = np.ones((1, D))
  b_eq = np.ones(1) # A_eq ve b_eq ağırlıklar toplamının 1 olmasını sağlar.
  bounds = [(0, 1)]*D # Her bir hisse senedinin ağırlığının 0 ile 1 arasında olması gerektiğini belirtir.

  # minimize
  res = linprog(mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
  min_return = res.fun # minimum getirili portföy ağırlıkları hesaplanır.

  # maximize
  res = linprog(-mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
  max_return = -res.fun # elde edilen max getiri

  N = 100 # minimum ve maksimum getiri arasında eşit aralıklı 100 farklı hedef getiri oluşturulur.
  target_returns = np.linspace(min_return, max_return, num=N) # bu eşit aralıklı getiri değerlerinin listesi

  constraints = [
    {
      'type': 'eq',
      'fun': target_return_constraint, # Portföyün beklenen getirisi, belirli bir hedef getiriye eşit olmalıdır.
      'args': [target_returns[0]], # will be updated in loop
    },
    {
      'type': 'eq',
      'fun': portfolio_constraint, # Portföy ağırlıklarının toplamı 1 olmalıdır.
    }
  ]

  optimized_risks = [] # belirli bir hedef getiri aralığı için
  # portföy riskini (standart sapma) en aza indiren ağırlıkları bulmayı amaçlar.
  for target in target_returns:
    # set target return constraint
    constraints[0]['args'] = [target]

    res = minimize( # belirlenen hedef getiriyi sağlayacak minimum risk. Etkin sınırda kullanacağız.
      fun=get_portfolio_variance,
      x0=np.ones(D) / D, # uniform, her bir hisse senedi için başlangıç ağırlığı 1/D olur.
      method='SLSQP',
      constraints=constraints,
      bounds=bounds,
    )
    optimized_risks.append(np.sqrt(res.fun)) # res.fun = porföy varyansı. sqrt ile std hesaplanmış olur.
    # Ve bu standart sapma, belirlenen hedef getiriyi sağlamak için gerekli MİNİMUM RİSKİ ifade eder.
    # "optimized_risks" isimli vektör, bu optimum risk değerlerini tutar.Bu bilgi efficient frontier
    # belirlemede yardımcı olur.
    if res.status != 0:
      print(res)

  if opt_type == 1: #  find the portfolio with the minimum variance.
    # Min variance portfolio
    # Let's limit the magnitude of the weights
    res = minimize(
      fun=get_portfolio_variance,
      # x0=[np.ones(D) / D, cov], # uniform
      x0=np.ones(D) / D, # uniform (başlangıç) => en sonda optimize edilmiş ağırlıklar elde edilecek.
      method='SLSQP',
      constraints={
          'type': 'eq',
          'fun': portfolio_constraint,
      },
      bounds=bounds,
    )
    # res.fun bize portföyün varyansını döndürür. (get_portfolio_variance)
    mv_risk = np.sqrt(res.fun) # portföyün standart sapması - RİSKİ
    mv_weights = res.x # portföyün optimum ağırlıkları
    mv_ret = mv_weights.dot(mean_return) # portföyün beklenen getirisi

  if opt_type == 2: # Sharpe ratio based
    res = minimize(
      fun=neg_sharpe_ratio, # The function to be minimized.
      x0=np.ones(D) / D, # uniform
      method='SLSQP', # Sequential Least Squares Quadratic Programming
      constraints={
        'type': 'eq',
        'fun': portfolio_constraint,
      },
      bounds=bounds,
    )
    mv_risk = np.sqrt(-res.fun) # sharpe ratio'yu maksimize eden portföyün riski
    mv_weights = res.x # optimum portföyün ağırlıkları
    mv_ret = mv_weights.dot(mean_return) # optimum portföyün return değeri


  if opt_type == 3: # monte carlo opt.
    N = 10000
    returns = np.zeros(N)
    risks = np.zeros(N)
    random_weights = []
    for i in range(N):
      # rand_range = 1.0
      # w = np.random.random(D)*rand_range - rand_range / 2 # with short-selling
      w = np.random.random(D)
      w = w / w.sum()
      # np.random.shuffle(w)
      random_weights.append(w)
      ret = mean_return.dot(w)
      risk = np.sqrt(w.dot(cov_np).dot(w))
      returns[i] = ret
      risks[i] = risk

    mc_best_w = None
    mc_best_sr = float('-inf')
    for i, (risk, ret) in enumerate(zip(risks, returns)):
      sr = (ret - risk_free_rate) / risk
      if sr > mc_best_sr:
        mc_best_sr = sr
        mc_best_w = random_weights[i]
    mv_weights = mc_best_w
    mv_risk = np.sqrt(mc_best_sr)
    mv_ret = mv_weights.dot(mean_return)
    print(f"weights = {mv_weights}")

  # plt.scatter(risks, returns, alpha=0.1);
  # plt.plot(optimized_risks, target_returns, c='black');
  # plt.scatter([mv_risk], [mv_ret], c='red');

  print(f"weights = {mv_weights}")
  for stock, weight in zip(stocks, mv_weights):
    print(f"Stock: {stock}, Weight: {100 * weight}%")

  print(f"\nTotal Weights: {mv_weights.sum() * 100}%")
  print(f"Total Negative Weights: {mv_weights[mv_weights < 0 ].sum() * 100}%")
  print(f"Total Positive Weights: {mv_weights[mv_weights > 0].sum() * 100}%\n")

  indexs = mv_weights > 0 # pozitif ağırlıklı olan hisseler nihai olarak "buy_stocks" içerisinde yer alır.
  # negatif ağırlıklı olanlar "buy_stocks" ta yer almaz. Ve ağırlıkları, pozitif ağırlıklılara eşit
  # olarak dağıtılır.
  buy_stocks = np.array(stocks)[indexs]
  sum_negative = mv_weights[mv_weights < 0 ].sum()
  # negatif ağırlıkları diğer hisselere dağıtıyor.
  # negatif ağırlıklara (short-sell yapılması gerektiğini gösteren) sahip hisse senetlerinin ağırlığını,
  # pozitif ağırlıklara sahip olanlara eşit şekilde dağıtarak pozitif ağırlıkları yeniden ayarlıyor.
  buy_weights = mv_weights[indexs] + mv_weights[indexs] * ((sum_negative) / mv_weights[indexs].sum())

  for stock, weight in zip(buy_stocks, buy_weights):
    print(f"{stock} : {np.round(weight * 100,2)}%")
  # Eğitim sürecinin sonunda yatırımcının portföyünde hangi hisse senetlerine yatırım yapması gerektiğini
  # (buy_stocks) ve bu hisse senetlerine ne kadar yatırım yapması gerektiğini (buy_weights) gösterir.

  return [buy_stocks, buy_weights]

def testPortfolio(prices, buyWeights, buyStocks, budget):
  stock_budgets = budget * buyWeights # Her bir hisse senedi için yatırılacak bütçeyi hesaplar.
  amounts = stock_budgets / prices.iloc[0] # Her bir hisse senedi için alınacak miktarı belirler.
  # Bu, belirlenen bütçe ile o anki hisse senedi fiyatına bölünerek bulunur.
  final_budget = np.sum(amounts * prices.iloc[-1]) # Yatırımın sonunda portföydeki toplam değeri hesaplar.
  gain = final_budget - budget # Yatırımın sonunda elde edilen kar veya zararı hesaplar.

  cost = amounts * prices.iloc[0] # Her bir hisse senedi için başlangıçta yatırılan miktarı hesaplar.
  final = amounts * prices.iloc[-1] # Her bir hisse senedi için yatırımın sonunda elde edilen değeri hesaplar.
  kar = final - cost # Her bir hisse senedi için elde edilen kar veya zararı hesaplar.
  print(f"\nstock\t\tadet\t\tcost\t\tfinal\t\tgain") # ÖNEMLİ
  # Daha sonra bu değerler bir tablo şeklinde ekrana yazdırılır. (MAİN kısmında var)
  for m, i, j, k, l in zip(buyStocks, amounts, cost, final, kar):
    if i > 0:
      print(f"{m}\t\t{np.round(i)}\t\t{np.round(j)}\t\t{np.round(k)}\t\t{np.round(l)}")
    else: continue
  print(f"\nPortf. budget = {np.round(budget,2)} {currency} , Portf. final budget = {np.round(final_budget,2)} {currency}, gain = {np.round(gain,2)} {currency}")
  return final_budget

month_period = 1
total_month = 3

# 20.06.2023 - 20.09.2023 (3 aylık)
# ilk 1 ay : train => 2.ay => test
# 2.ay train => 3. Test

opt_types = {
  1: "Mean Variance Optimization",
  2: "Sharpe Ratio Optimization",
  3: "Monte Carlo Simulation"
}

opt_type = 3
risk_free_rate = 25 / 252
stocks = ["AFA","AFS","AFT","AFV","AOY","GBG","TMG","YAY","ZSF"]

base_path = r"C:\Users\irems\OneDrive - ABDULLAH GUL UNIVERSITESI\Masaüstü\veriler\new\csv\\"


start_budget = 100000 # Initial budget
budget = start_budget
stock_market_budget = start_budget

start_budget_list = []
port_budget_list = []
market_budget_list = []

# Custom date parser function
date_parser = lambda x: pd.to_datetime(x, format='%d.%m.%Y')

# Load the first stock data to get the list of all dates
df = pd.read_csv(f"{base_path + stocks[0]}.csv", index_col='Tarih', parse_dates=True, date_parser=date_parser)
all_dates = df.index.sort_values().unique()
close_prices = pd.DataFrame(index=all_dates)

# Loop through each stock and add its closing prices to the DataFrame
for stock in stocks:
    df = pd.read_csv(f"{base_path + stock}.csv", index_col='Tarih', parse_dates=True, date_parser=date_parser)
    df_tmp = df.reindex(all_dates, method='ffill')['Fiyat']
    # Convert string prices to floats for the current stock
    df_tmp = df_tmp.str.replace(',', '.').astype(float)
    close_prices[stock] = df_tmp

# Handle any missing data by forward filling
if close_prices.isna().sum().sum() > 0:
    close_prices.fillna(method='ffill', inplace=True)

# Display the closing prices for all shares
print(close_prices.head())

all_dates # 2023-06-20 ile 2023-09-20 arasında olmalı

def percentOfGain(x1, x2):
  return (x2 - x1) * 100 / x1
cov = 0

currency = "TL"
stock_market = "HISSE_SENEDI_FONLARI"

train_start_date = all_dates.min() # 2023-06-20
train_start_index = all_dates.get_loc(str(train_start_date))
periods = [] # periyot numaralarını tutar
period_dates = [] # her bir periyoda denk gelen tarih aralıklarını tutar
for period in range((total_month // month_period) - 1): # -1 eklenmesinin sebebi, son periyodun (9-12.ay) sadece eğitim için kullanılması. Total 3 periyot
# eğitim ve test olacak.
  print(f"Period: {period + 1}")
  periods.append(str(period + 1)) # periods isimli diziye periyot numarasını (1,2,3) ekler
  train_end_date = train_start_date + pd.DateOffset(months=month_period) + pd.DateOffset(days = 1) # 3 aylık eğitim sürecinin sonu "train_end_date" olur.
  if train_end_date > all_dates[-1]: # Eğer eğitim periyodunun bitiş tarihi veri setinin son tarihinden daha sonraysa, bu periyoda atlanır.
    continue
  while not (train_end_date in all_dates): # Eğer train_end_date belirtilen tarihlere sahip değilse, tarihi bir gün ileri alarak doğru tarihi bulmaya çalışır.
    train_end_date += pd.DateOffset(days = 1)
  train_end_index = all_dates.get_loc(str(train_end_date))

  # print(f"Train start: {train_start_index}, Train End = {train_end_index}")
  test_start_date = train_end_date # Test süreci, eğitim sürecinin tamamen bittiği tarihle başlar.
  test_start_index = train_end_index
  test_end_date = test_start_date + pd.DateOffset(months=month_period) + pd.DateOffset(days = 1) # 3 aylık test süreci (eğitimin bittiği tarihten itibaren)
  if test_end_date > all_dates[-1]: # Eğer hesaplanan test_end_date, veri setindeki son tarihten sonraysa, test bitiş indeksi -1 olarak atanır.
      test_end_index = -1
  else: # test_end_date tarih dizisinde yoksa, bu tarih bulunana kadar her seferinde bir gün eklenir.
    while not (test_end_date in all_dates):
      test_end_date += pd.DateOffset(days = 1)
    test_end_index = all_dates.get_loc(str(test_end_date))

  # print(f"Test start: {test_start_index}, Test End = {test_end_index}")
  # print(f"test start date = {test_start_date}")
  # print(f"test end date = {test_end_date}")
  period_dates.append(str(datetime.strptime(str(test_start_date), '%Y-%m-%d %H:%M:%S').date()) + " / " + str(datetime.strptime(str(test_end_date), '%Y-%m-%d %H:%M:%S').date()))

  prices = close_prices.iloc[train_start_index : train_end_index] # Eğitim setindeki kapanış fiyatlarını al.
  dates = all_dates[train_start_index : train_end_index] # Eğitim setindeki tarihleri al.
  [buyStocks, buyWeights] = trainPortfolio(prices, dates, opt_type, stocks) # Eğitim seti üzerinde trainPortfolio fonksiyonunu çalıştırarak
  # en iyi hisseleri ve bu hisselere ne kadar yatırım yapılması gerektiğini hesapla.
  prices = close_prices.iloc[test_start_index : test_end_index][buyStocks] # Test setindeki, eğitim setinde belirlenen hisselerin kapanış fiyatlarını al.
  final_budget = testPortfolio(prices, buyWeights, buyStocks, budget) # Test setindeki hisselerin performansını testPortfolio fonksiyonuyla kontrol et ve

  print(f"portfolio gain = {np.round(percentOfGain(budget, final_budget),2)}%")

  # budget = final_budget
  budget = 100000
  start_budget_list.append(start_budget)
  port_budget_list.append(final_budget)


  train_start_date = test_start_date # 1 periyot kaymış oldu. 2. periyoda geçildi.
  train_start_index = all_dates.get_loc(str(train_start_date))
  print("\n\n\n")

