
import os
import sys
import ast
import configparser

import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')

import numpy as np
import pandas as pd

from metpy.calc import wind_direction
from metpy.units import units

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF')
os.chdir(cartella_lavoro)

from funzioni import f_apri_pickle
from funzioni import f_ricomposizione_seno_cose
from danilib import f_log_ciclo_for

config = configparser.ConfigParser()
config.read('./config.ini')

if len(sys.argv) > 1:
    data_arg = ' '.join(sys.argv[1:])
    data = pd.Timestamp(data_arg)
else:
    data = pd.Timestamp('2026-07-11')

print(data)


modello = config.get('COMMON', 'modello')
cartella_modelli_allenati = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_modelli_allenati')}", f'vento/modelli_allenati/{modello}')
cartella_dataset = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_dataset')}")
quantili = ast.literal_eval(config.get('COMMON', 'quantili'))

percorso_data = f"{data.strftime('%Y/%m/%d')}"

cartella_previsioni = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_previsioni')}/vento/{modello}/{percorso_data}")
os.makedirs(cartella_previsioni, exist_ok=True)

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/vento/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index:
    if os.path.exists(f"{cartella_previsioni}/{stazione}.csv"):
        continue
    
    f_log_ciclo_for([['Stazione ', stazione, df_stazioni.index.tolist()]])
    
    try:
        tipo_di_previsioni = set([x.split('_')[-1].split('.')[0] for x in os.listdir(f'{cartella_modelli_allenati}/{stazione}') if x.endswith('.pkl')])
    except FileNotFoundError:
        continue
    
    df_previsioni_tot_WSPDM = pd.DataFrame()
    df_previsioni_tot_WDIRP = pd.DataFrame()
    df_previsioni_tot_WSPDX = pd.DataFrame()
    
    for intervallo in ['0_24', '24_48', '48_72']:
        ### La WSPDM c'è sempre
        dict_WSPDM = f_apri_pickle(f'{cartella_modelli_allenati}/{stazione}/QRF_{stazione}_{intervallo}_WSPDM.pkl')
        modello_WSPDM = dict_WSPDM['modello']
        
        X = pd.read_csv(f"{cartella_dataset}/{modello}/df_{stazione}_{intervallo}_{data.strftime('%Y%m%d')}.csv", index_col=0, parse_dates=True)
        X['10gust3max'] = X['10gust3max'].interpolate(method='linear', limit_direction='both')
        X = X[dict_WSPDM['campi']]
        
        def f_fai_le_previsioni(tipo_W, modello_vento, df_previsioni_tot):
            previsioni = modello_vento.predict(X)
            
            if tipo_W == 'WDIRP':
                previsioni = f_ricomposizione_seno_cose(previsioni)
                df_previsioni = pd.DataFrame(previsioni.T, index=X.index)
            else:
                df_previsioni = pd.DataFrame(previsioni.T, index=X.index, columns=quantili)
                df_previsioni['media'] = df_previsioni.mean(axis=1)
            
            
            if tipo_W == 'WSPDM':
                df_previsioni.columns = [f'QRF vento {x}' for x in df_previsioni.columns]
                df_raw = pd.DataFrame(np.sqrt(X['10u'] ** 2 + X['10v'] ** 2), index=X.index, columns=[f'{modello} vento'])
            if tipo_W == 'WDIRP':
                df_previsioni.columns = ['RF direzione']
                df_raw = pd.DataFrame(wind_direction(X['10u'].values * units('m/s'), X['10v'].values * units('m/s')).magnitude, index=X.index, columns=[f'{modello} direzione'])
            if tipo_W == 'WSPDX':
                df_previsioni.columns = [f'QRF raffica {x}' for x in df_previsioni.columns]
                df_raw = pd.DataFrame(X['10gust3max'].values, index=X.index, columns=[f'{modello} raffica'])
                
            df_previsioni = pd.concat([df_previsioni, df_raw], axis=1)
            
            df_previsioni_tot = pd.concat([df_previsioni_tot, df_previsioni], axis=0)
        
            return df_previsioni_tot
        
        df_previsioni_tot_WSPDM = f_fai_le_previsioni('WSPDM', modello_WSPDM, df_previsioni_tot_WSPDM)
        
        if 'WDIRP' in tipo_di_previsioni:
            dict_WDIRP = f_apri_pickle(f'{cartella_modelli_allenati}/{stazione}/RF_{stazione}_{intervallo}_WDIRP.pkl')
            modello_WDIRP = dict_WDIRP['modello']
            df_previsioni_tot_WDIRP = f_fai_le_previsioni('WDIRP', modello_WDIRP, df_previsioni_tot_WDIRP)

        if 'WSPDX' in tipo_di_previsioni:
            dict_WSPDX = f_apri_pickle(f'{cartella_modelli_allenati}/{stazione}/QRF_{stazione}_{intervallo}_WSPDX.pkl')
            modello_WSPDX = dict_WSPDX['modello']
            df_previsioni_tot_WSPDX = f_fai_le_previsioni('WSPDX', modello_WSPDX, df_previsioni_tot_WSPDX)

    df_previsioni_tot_WSPDM = df_previsioni_tot_WSPDM.round(1)
    df_previsioni_tot_WDIRP = df_previsioni_tot_WDIRP.round(1)
    df_previsioni_tot_WSPDX = df_previsioni_tot_WSPDX.round(1)
    
    ####################

    dict_colonne = {'QRF vento media': 'QRF vento', 'QRF raffica media': 'QRF raffica'}

    try:
        df_previsioni_tot = pd.concat([
            df_previsioni_tot_WSPDM[['QRF vento 0.25', 'QRF vento 0.75', 'QRF vento media', f'{modello} vento']],
            df_previsioni_tot_WDIRP[['RF direzione', f'{modello} direzione']],
            df_previsioni_tot_WSPDX[['QRF raffica 0.25', 'QRF raffica 0.75', 'QRF raffica media', f'{modello} raffica']]],
            axis=1)

    except KeyError:
        df_previsioni_tot = pd.concat([
            df_previsioni_tot_WSPDM[['QRF vento 0.25', 'QRF vento 0.75', 'QRF vento media', f'{modello} vento']]],
            axis=1)
        
    df_previsioni_tot = df_previsioni_tot.rename(columns=dict_colonne)
    
    df_previsioni_tot = df_previsioni_tot.astype(float)
    df_previsioni_tot.to_csv(f"{cartella_previsioni}/{stazione}.csv", index=True, header=True, mode='w', na_rep=np.nan)

print('\n\nDone')
