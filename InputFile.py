import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from SeedData import FillMissingValues

plt.style.use('fivethirtyeight')

def yfinance_veri_cek(sembol, baslangic_tarih, bitis_tarih, frekans):
    liste = []

    for s in sembol:
        tarihsel = yf.download(
            tickers=s,
            start=baslangic_tarih,
            end=bitis_tarih,
            interval=frekans
        )

        veriler_df_alt = pd.DataFrame(tarihsel).reset_index()
        veriler_df_alt = veriler_df_alt[['Date', 'Adj Close']].rename(
            columns={'Adj Close': f'{s.replace(".IS", "")}'})
        liste.append(veriler_df_alt)

    veriler_df = liste[0]
    for i in range(1, len(liste)):
        veriler_df = pd.merge(veriler_df, liste[i], on='Date', how='outer')

    veriler_df = veriler_df.drop(columns='Date')

    dFClean = veriler_df.dropna(axis=0, how='all')

    dFClean=FillMissingValues(dFClean)

    return dFClean



