"""
Creazione: Tue Jul 21 15:53:47 2026
Autore: daniele.carnevale
"""

import os
import sys
import ast
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF/pioggia')
os.chdir(cartella_lavoro)

from funzioni import f_eda_calibrazione

config = configparser.ConfigParser()
config.read('./../config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = os.path.join(CARTELLA_REPO_ROOT, f"estrazioni_puntuali/output/{modello}/tp")

range_date = pd.date_range('2019-01-01', '2026-06-30', freq='1d')

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index[0:15]:
    sr_obs = pd.read_csv(f'{cartella_lavoro}/../osservati/{stazione}.csv', index_col=0, parse_dates=True)['RAIN03HX']
    
    lista_0_24, lista_24_48, lista_48_72 = [], [], []
    
    df_0_24 = pd.read_csv(os.path.join(CARTELLA_REPO_ROOT, f'estrazioni_puntuali/concatenazioni/{modello}/df_{stazione}_0_24.csv'), index_col=0, parse_dates=True)['tp']
    df_24_48 = pd.read_csv(os.path.join(CARTELLA_REPO_ROOT, f'estrazioni_puntuali/concatenazioni/{modello}/df_{stazione}_24_48.csv'), index_col=0, parse_dates=True)['tp']
    df_48_72 = pd.read_csv(os.path.join(CARTELLA_REPO_ROOT, f'estrazioni_puntuali/concatenazioni/{modello}/df_{stazione}_48_72.csv'), index_col=0, parse_dates=True)['tp']
    
    # ######### Valido per ecita
    # for giorno in range_date:
    #     try:
    #         df = pd.read_csv(f"{cartella_dataset}/{stazione}/{giorno.strftime('%Y-%m-%d')}.csv", index_col=0, parse_dates=True).squeeze()
    #     except FileNotFoundError:
    #         continue
        
    #     if df.shape[0] < 54:
    #         continue
        
    #     df = df * 1000 # m -> mm
    #     df = df.reindex(pd.date_range(giorno + pd.Timedelta(hours=3), giorno + pd.Timedelta(days=3), freq='3h'))
    #     df = f_cumulata_a_3h(df, giorno).round(1)
    #     lista_0_24.append(df.loc[giorno + pd.Timedelta(hours=3):giorno + pd.Timedelta(hours=24)])
    #     lista_24_48.append(df.loc[giorno + pd.Timedelta(hours=24+3):giorno + pd.Timedelta(hours=48)])
    #     lista_48_72.append(df.loc[giorno + pd.Timedelta(hours=48+3):giorno + pd.Timedelta(hours=72)])
    
    # df_0_24 = pd.concat(lista_0_24, axis=0) if lista_0_24 else pd.Series(dtype=float)
    # df_24_48 = pd.concat(lista_24_48, axis=0) if lista_24_48 else pd.Series(dtype=float)
    # df_48_72 = pd.concat(lista_48_72, axis=0) if lista_48_72 else pd.Series(dtype=float)
    
    # #########

    idx_completo_0_24 = pd.date_range(df_0_24.index.min(), df_0_24.index.max(), freq='3h')
    idx_completo_24_48 = pd.date_range(df_24_48.index.min(), df_24_48.index.max(), freq='3h')
    idx_completo_48_72 = pd.date_range(df_48_72.index.min(), df_48_72.index.max(), freq='3h')
    
    df_0_24 = pd.concat([df_0_24, sr_obs], axis=1).dropna().reindex(idx_completo_0_24)
    df_24_48 = pd.concat([df_24_48, sr_obs], axis=1).dropna().reindex(idx_completo_24_48)
    df_48_72 = pd.concat([df_48_72, sr_obs], axis=1).dropna().reindex(idx_completo_48_72)
    
    df_0_24.columns = ['prev', 'obs']
    df_24_48.columns = ['prev', 'obs']
    df_48_72.columns = ['prev', 'obs']

    res_0_24 = f_eda_calibrazione(df_0_24, f"{stazione} 0-24h")
    print()
    res_24_48 = f_eda_calibrazione(df_24_48, f"{stazione} 24-48h")
    print()
    res_48_72 = f_eda_calibrazione(df_48_72, f"{stazione} 48-72h")
    print()
    
    """
    frac_wet_obs vs frac_wet_prev
        Se sono molto diverse (es. obs: 0.166  prev: 0.216) hai un bias sistematico su pioggia/no-pioggia. Questo è il primo bias da correggere.

    Skew obs_wet vs Skew prev_wet
        Se obs_wet >> prev_wet il modello appiattisce gli eventi intenti. Serve una distribuzione a coda pesante per l'osservato.

    Spearman rho
        Misura quanto la feature 'prev' (in questo caso tp), è informativa. Se bassa (es. <0.4) sulle 48-72h rispetto alle 0-24h dice che a flt lunghi la 'prev' perde skill.

    n_oltre_obs
        Se lì ci sono pochi eventi, qualunque fit di coda sarà statisticamente fragile.

    Istogramma log1p
        Se la distribuzione di prev è spostata a sinistra rispetto a obs, il modello sottostima sistematicamente l'intensità quando piove (bias moltiplicativo) — indizio a favore di correzioni tipo quantile mapping o un termine di scala nella distribuzione parametrica.

    Hexbin
        Se la nuvola di punti è sistematicamente sotto la diagonale, il modello sovrastima; sopra, sottostima.

    Boxplot
        Guarda l'ampiezza della "scatola" (IQR) bin per bin: se cresce con l'intensità prevista, è la conferma visiva più diretta che ti serve un modello a varianza condizionale (CSGD/EGPD/QRF), non un semplice fattore di correzione costante.

    QQ-plot
        Punti allineati sulla retta = la gamma descrive bene la parte piovosa osservata → buon segno per usare CSGD (che è basata su gamma) come distribuzione candidata.
        Deviazione sistematica verso l'alto nella coda destra (gli ultimi punti si staccano sopra la retta) = la coda osservata è più pesante di una gamma → serve GPD/EGPD invece di una gamma pura, oppure CSGD con un parametro di forma libero.
    """

    # sss

print('\n\nDone.')
