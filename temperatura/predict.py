
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')

import sys
import ast
import configparser

import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF')
os.chdir(cartella_lavoro)

from funzioni import f_apri_pickle
from danilib import f_log_ciclo_for

config = configparser.ConfigParser()
config.read('./config.ini')

if len(sys.argv) > 1:
    data_arg = ' '.join(sys.argv[1:])
    data = pd.Timestamp(data_arg)
else:
    data = pd.Timestamp('2026-07-14')

print(data)

modello = config.get('COMMON', 'modello')
cartella_modelli_allenati = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_modelli_allenati')}", f'temperatura/modelli_allenati/{modello}')
cartella_dataset = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_dataset')}")
quantili = ast.literal_eval(config.get('COMMON', 'quantili'))

percorso_data = f"{data.strftime('%Y/%m/%d')}"

cartella_previsioni = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_previsioni')}/temperatura/{modello}/{percorso_data}")
os.makedirs(cartella_previsioni, exist_ok=True)

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/temperatura/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index:
    # if os.path.exists(f"{cartella_previsioni}/{stazione}.csv"):
    #     continue
    
    f_log_ciclo_for([['Stazione ', stazione, df_stazioni.index.tolist()]])

    tipo_di_previsioni = set([x.split('_')[-1].split('.')[0] for x in os.listdir(f'{cartella_modelli_allenati}/{stazione}') if x.endswith('.pkl')])
    
    df_previsioni_tot_TEMPN = pd.DataFrame()
    df_previsioni_tot_TEMPM = pd.DataFrame()
    df_previsioni_tot_TEMPX = pd.DataFrame()
    
    for intervallo in ['0_24', '24_48', '48_72']:
        ### La TEMPM c'è sempre
        dict_TEMPM = f_apri_pickle(f'{cartella_modelli_allenati}/{stazione}/QRF_{stazione}_{intervallo}_TEMPM.pkl')
        modello_TEMPM = dict_TEMPM['modello']
        
        X = pd.read_csv(f"{cartella_dataset}/{modello}/df_{stazione}_{intervallo}_{data.strftime('%Y%m%d')}.csv", index_col=0, parse_dates=True)
        X['10gust3max'] = X['10gust3max'].interpolate(method='linear', limit_direction='both')
        X = X[dict_TEMPM['campi']]

        def f_fai_le_previsioni(tipo_T, modello_temperatura, df_previsioni_tot):
            previsioni = modello_temperatura.predict(X)
            
            df_previsioni = pd.DataFrame(previsioni.T, index=X.index, columns=quantili)
            df_previsioni['media'] = df_previsioni.mean(axis=1)
            
            if tipo_T == 'TEMPM':
                df_previsioni.columns = [f'QRF media {x}' for x in df_previsioni.columns]
            if tipo_T == 'TEMPN':
                df_previsioni.columns = [f'QRF minima {x}' for x in df_previsioni.columns]
            if tipo_T == 'TEMPX':
                df_previsioni.columns = [f'QRF massima {x}' for x in df_previsioni.columns]
    
            if tipo_T == 'TEMPM':
                df_raw = pd.DataFrame(X['2t'].values - 273.15, index=X.index, columns=[modello])
                df_previsioni = pd.concat([df_previsioni, df_raw], axis=1)
    
            df_previsioni_tot = pd.concat([df_previsioni_tot, df_previsioni], axis=0)
        
            return df_previsioni_tot
        
        df_previsioni_tot_TEMPM = f_fai_le_previsioni('TEMPM', modello_TEMPM, df_previsioni_tot_TEMPM)
        
        if 'TEMPN' in tipo_di_previsioni:
            dict_TEMPN = f_apri_pickle(f'{cartella_modelli_allenati}/{stazione}/QRF_{stazione}_{intervallo}_TEMPN.pkl')
            modello_TEMPN = dict_TEMPN['modello']

            dict_TEMPX = f_apri_pickle(f'{cartella_modelli_allenati}/{stazione}/QRF_{stazione}_{intervallo}_TEMPX.pkl')
            modello_TEMPX = dict_TEMPX['modello']

            df_previsioni_tot_TEMPN = f_fai_le_previsioni('TEMPN', modello_TEMPN, df_previsioni_tot_TEMPN)
            df_previsioni_tot_TEMPX = f_fai_le_previsioni('TEMPX', modello_TEMPX, df_previsioni_tot_TEMPX)

    df_previsioni_tot_TEMPN = df_previsioni_tot_TEMPN.round(1)
    df_previsioni_tot_TEMPM = df_previsioni_tot_TEMPM.round(1)
    df_previsioni_tot_TEMPX = df_previsioni_tot_TEMPX.round(1)
    
    ####################
    
    df_clima_stazione = pd.read_csv(f"./temperatura/climatologie/{stazione}.csv", index_col=0, parse_dates=True)
    
    anno = df_previsioni_tot_TEMPM.index[0].year
    if df_previsioni_tot_TEMPM.shape[0] == 72:
        freq = '1h'
    else:
        freq = '3h'
        
    if anno % 4 != 0:
        # Anno bisestile
        df_clima_stazione = df_clima_stazione[~((df_clima_stazione.index.month == 2) & (df_clima_stazione.index.day == 29))]
    
    df_clima_stazione.index = pd.date_range(f'{anno}-01-01 00:00:00', f'{anno}-12-31 23:00:00', freq=freq)
    df_clima_stazione = df_clima_stazione.loc[df_previsioni_tot_TEMPM.index].round(1)
    df_clima_stazione.columns = ['Clima']
    
    # !!! Nel caso di cambio d'anno possono succedere casini!

    ####################
    dict_colonne = {'QRF minima media': 'QRF minima', 'QRF media media': 'QRF media', 'QRF massima media': 'QRF massima'}
    
    try:
        df_previsioni_tot = pd.concat([
            df_previsioni_tot_TEMPN[['QRF minima media']],
            df_previsioni_tot_TEMPM[['QRF media media', modello]],
            df_previsioni_tot_TEMPX[['QRF massima media']],
            df_previsioni_tot_TEMPN[['QRF minima 0.25', 'QRF minima 0.75']],
            df_previsioni_tot_TEMPM[['QRF media 0.25', 'QRF media 0.75']],
            df_previsioni_tot_TEMPX[['QRF massima 0.25', 'QRF massima 0.75']],
            df_clima_stazione],
            axis=1)

    except KeyError:
        df_previsioni_tot = pd.concat([
            df_previsioni_tot_TEMPM[['QRF media media', modello]],
            df_previsioni_tot_TEMPM[['QRF media 0.25', 'QRF media 0.75']],
            df_clima_stazione],
            axis=1)
        
    df_previsioni_tot = df_previsioni_tot.rename(columns=dict_colonne)
    
    df_previsioni_tot = df_previsioni_tot.astype(float)
    df_previsioni_tot.to_csv(f"{cartella_previsioni}/{stazione}.csv", index=True, header=True, mode='w', na_rep=np.nan)

print('\n\nDone')
