
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

from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

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
cartella_modelli_allenati = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_modelli_allenati')}", f'umidita/modelli_allenati/{modello}')
cartella_dataset = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_dataset')}")
quantili = ast.literal_eval(config.get('COMMON', 'quantili'))

percorso_data = f"{data.strftime('%Y/%m/%d')}"

cartella_previsioni = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_previsioni')}/umidita/{modello}/{percorso_data}")
os.makedirs(cartella_previsioni, exist_ok=True)

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/umidita/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index:
    # if os.path.exists(f"{cartella_previsioni}/{stazione}.csv"):
    #     continue
    
    f_log_ciclo_for([['Stazione ', stazione, df_stazioni.index.tolist()]])
    
    tipo_di_previsioni = set([x.split('_')[-1].split('.')[0] for x in os.listdir(f'{cartella_modelli_allenati}/{stazione}') if x.endswith('.pkl')])
    
    df_previsioni_tot = pd.DataFrame()
    
    for intervallo in ['0_24', '24_48', '48_72']:
        dict_modello = f_apri_pickle(f'{cartella_modelli_allenati}/{stazione}/QRF_{stazione}_{intervallo}_REHUM.pkl')
        
        X = pd.read_csv(f"{cartella_dataset}/{modello}/df_{stazione}_{intervallo}_{data.strftime('%Y%m%d')}.csv", index_col=0, parse_dates=True)
        X['10gust3max'] = X['10gust3max'].interpolate(method='linear', limit_direction='both')
        X = X[dict_modello['campi']]
        
        previsioni = dict_modello['modello'].predict(X)
        
        df_previsioni = pd.DataFrame(previsioni.T, index=X.index, columns=quantili)
        df_previsioni['media'] = df_previsioni.mean(axis=1)
        
        df_previsioni.columns = [f'QRF media {x}' for x in df_previsioni.columns]
        
        if modello == 'ecita':
            df_raw = pd.DataFrame(relative_humidity_from_dewpoint(X['2t'].values * units.degK, X['2d'].values * units.degK).to('percent').magnitude, index=X.index, columns=[modello])
        else:
            raise
            
        df_previsioni = pd.concat([df_previsioni, df_raw], axis=1)
        
        df_previsioni_tot = pd.concat([df_previsioni_tot, df_previsioni], axis=0)
    
    df_previsioni_tot = df_previsioni_tot.round(1)
    
    ####################
    
    dict_colonne = {'QRF media media': 'QRF media'}
    
    df_previsioni_tot = pd.concat([
        df_previsioni_tot[['QRF media media', modello]],
        df_previsioni_tot[['QRF media 0.25', 'QRF media 0.75']]],
        axis=1)
        
    df_previsioni_tot = df_previsioni_tot.rename(columns=dict_colonne)
    
    df_previsioni_tot_grafici_interattivi = df_previsioni_tot.astype(float)
    df_previsioni_tot_grafici_interattivi.to_csv(f"{cartella_previsioni}/{stazione}.csv", index=True, header=True, mode='w', na_rep=np.nan)

print('\n\nDone')
