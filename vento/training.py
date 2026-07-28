
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore', message='IProgress not found.*')

import os
import ast
import sys
import configparser

import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from metpy.calc import wind_direction
from metpy.units import units

plt.rc('font', weight='normal', size=6)

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF')
os.chdir(cartella_lavoro)

# Patch di compatibilità: sklearn 1.9 richiede sample_weight in _generate_sample_indices,
# ma sklearn_quantile (0.1.1) la chiama ancora senza. Va fatto PRIMA di importare sklearn_quantile.
import sklearn.ensemble._forest as _forest_mod
_generate_sample_indices_originale = _forest_mod._generate_sample_indices

def _generate_sample_indices_patch(random_state, n_samples, n_samples_bootstrap, sample_weight=None):
    return _generate_sample_indices_originale(random_state, n_samples, n_samples_bootstrap, sample_weight)

_forest_mod._generate_sample_indices = _generate_sample_indices_patch

from funzioni import QRF_model
from funzioni import RF_model
from funzioni import f_salva_pickle
from funzioni import f_apri_pickle
from funzioni import f_errori_regressione
from funzioni import f_scomposizione_seno_coseno
from funzioni import f_ricomposizione_seno_cose
from funzioni import f_errore_circolare

from danilib import f_log_ciclo_for

config = configparser.ConfigParser()
config.read('./config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = f"{cartella_lavoro}/../{config.get('COMMON', 'cartella_dataset')}/{modello}"
cartella_modelli_allenati = f"{cartella_lavoro}/vento/modelli_allenati"
os.makedirs(f'{cartella_modelli_allenati}/{modello}', exist_ok=True)

colori = {'0_24': 'tab:blue', '24_48': 'tab:orange', '48_72': 'tab:green'}

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/vento/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index:
    try:
        df_0_24 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_0_24.csv', index_col=0, parse_dates=True)
        df_24_48 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_24_48.csv', index_col=0, parse_dates=True)
        df_48_72 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_48_72.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f'\n\nLa stazione {stazione} non ha i dataset\n')
        continue

    df_osservati = pd.read_csv(f'{cartella_lavoro}/osservati/{stazione}.csv', index_col=0, parse_dates=True)[['WSPDM', 'WSPDX', 'WDIRP']]
    df_osservati = df_osservati.dropna(axis=1, how='all')
    
    df_0_24 = pd.concat([df_0_24, df_osservati], axis=1)
    df_24_48 = pd.concat([df_24_48, df_osservati], axis=1)
    df_48_72 = pd.concat([df_48_72, df_osservati], axis=1)
    
    for intervallo, df_int in zip(['0_24', '24_48', '48_72'], [df_0_24, df_24_48, df_48_72]):
            
        for osservato in df_osservati.columns:
            f_log_ciclo_for([
                ['Stazione ', stazione, df_stazioni.index],
                ['Intervallo ', intervallo, ['0_24', '24_48', '48_72']],
                ['Osservato ', osservato, df_osservati.columns.tolist()]
            ])
            df = df_int.drop(columns=[x for x in ['WSPDM', 'WSPDX', 'WDIRP'] if x != osservato], errors='ignore').dropna().copy()
                        
            prefisso_modello = 'RF' if osservato == 'WDIRP' else 'QRF'
            # if os.path.exists(f'{cartella_modelli_allenati}/{modello}/{stazione}/{prefisso_modello}_{stazione}_{intervallo}_{osservato}.pkl') and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
            #     continue

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                X_training = df.loc[:'2024-12-31 21:00:00']
                X_test = df.loc['2025-01-01 00:00:00':]
                y_training = X_training.pop(osservato)
                
                if osservato == 'WDIRP':
                    y_test_dir = X_test.pop(osservato)
                    y_training = f_scomposizione_seno_coseno(y_training)
                    y_test = f_scomposizione_seno_coseno(y_test_dir)
                else:
                    y_test = X_test.pop(osservato)
            
            else:
                X_training = df
                y_training = X_training.pop(osservato)
                if osservato == 'WDIRP':
                    y_training = f_scomposizione_seno_coseno(y_training)

            if osservato == 'WDIRP':
                model = RF_model()
            else:
                model = QRF_model(ast.literal_eval(config.get('COMMON', 'quantili')))
            model.fit(X_training, y_training)
            
            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                previsioni = model.predict(X_test)
                if osservato == 'WDIRP':
                    previsioni = f_ricomposizione_seno_cose(previsioni)
                
                if osservato == 'WDIRP':
                    df_previsioni = pd.DataFrame(previsioni.T, index=X_test.index, columns=['RF'])
                else:
                    df_previsioni = pd.DataFrame(previsioni.T, index=X_test.index, columns=ast.literal_eval(config.get('COMMON', 'quantili')))
                    df_previsioni['media'] = df_previsioni.mean(axis=1)
                    df_previsioni.columns = [f'QRF_{x}' for x in df_previsioni.columns]
            
                if osservato == 'WDIRP':
                    raw = wind_direction(X_test['10u'].values * units('m/s'), X_test['10v'].values * units('m/s')).magnitude
                else:
                    raw = np.sqrt(X_test['10u'] ** 2 + X_test['10v'] ** 2)
                    
                df_raw = pd.DataFrame(raw, index=X_test.index, columns=[f'Raw_{modello}'])
                
                if osservato == 'WDIRP':
                    df_raw_prev_obs = pd.concat([df_raw, df_previsioni, y_test_dir], axis=1).round(2)
                    mae_circolare_raw = float(np.round(np.mean(np.abs(f_errore_circolare(y_test_dir, raw))), 3))
                    mae_circolare_prev = float(np.round(np.mean(np.abs(f_errore_circolare(y_test_dir, previsioni))), 3))
                    print(f'{mae_circolare_raw = }')
                    print(f'{mae_circolare_prev = }')
                    
                else:
                    df_errori_raw = f_errori_regressione(y_test, df_raw, nome_df=f'{stazione}_{intervallo}_raw_test')
                    df_errori_prev = f_errori_regressione(y_test, df_previsioni['QRF_media'], nome_df=f'{stazione}_{intervallo}_prev_test')
                    df_errori = pd.concat([df_errori_raw, df_errori_prev], axis=1).round(2)
                    print(f"\n{tabulate(df_errori, tablefmt='simple', headers='keys')}\n")

            ################################ Salvataggi
            
            os.makedirs(f'{cartella_modelli_allenati}/{modello}/{stazione}', exist_ok=True)
            
            dict_model = {
                'modello': model,
                'campi': X_training.columns.tolist()
            }
            
            if osservato == 'WDIRP':
                f_salva_pickle(dict_model, f'{cartella_modelli_allenati}/{modello}/{stazione}/RF_{stazione}_{intervallo}_{osservato}.pkl')
            else:
                f_salva_pickle(dict_model, f'{cartella_modelli_allenati}/{modello}/{stazione}/QRF_{stazione}_{intervallo}_{osservato}.pkl')
    
    ################################ Feature Importance
    for osservato in df_osservati.columns:
        
        if os.path.exists(f"{cartella_modelli_allenati}/{modello}/{stazione}/FI_{stazione}_{osservato}.png") and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
            continue
        
        serie = {}
        for intervallo in ['0_24', '24_48', '48_72']:
            if osservato == 'WDIRP':
                d = f_apri_pickle(f'{cartella_modelli_allenati}/{modello}/{stazione}/RF_{stazione}_{intervallo}_{osservato}.pkl')
            else:
                d = f_apri_pickle(f'{cartella_modelli_allenati}/{modello}/{stazione}/QRF_{stazione}_{intervallo}_{osservato}.pkl')
            serie[intervallo] = pd.Series(d['modello'].feature_importances_, index=d['campi'])
            
        # allineamento esplicito: forza l'ordine dei campi del primo intervallo
        df = pd.DataFrame(serie).reindex(serie['0_24'].index)
        
        # controllo anti-disallineamento (vedi nota sotto)
        if df.isna().any().any():
            raise ValueError(f"{stazione}: campi non allineati tra intervalli\n{df[df.isna().any(axis=1)]}")
        
        fig, ax = plt.subplots()
        df.plot.bar(ax=ax, zorder=10, width=0.8, color=[colori[c] for c in df.columns])
        ax.set_ylabel("Importance")
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1)
        ax.tick_params(axis='x', labelsize=4.5)
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.4, alpha=0.5, zorder=-10)
        
        nome_stazione = df_stazioni.loc[stazione]['Name']
        lat_lon = f"{df_stazioni.loc[stazione]['Latitude'].round(2)}, {df_stazioni.loc[stazione]['Longitude'].round(2)}"
        quota = int(df_stazioni.loc[stazione]['Altitude'])
        zona = df_stazioni.loc[stazione]['zona_allertamento']
        titolo = f"{nome_stazione}, {quota} metri ({lat_lon}) - Zona {zona}"
        ax.set_title(titolo, loc='right', fontsize=7)
        ax.set_title(f'{modello} - {osservato}', loc='left', fontsize=7)
        
        ax.legend(
            title='Intervallo',
            prop={'size': 6},
            loc='upper right',
            ncols=3,
        )
        
        fig.tight_layout()
        percorso_plot = f"{cartella_modelli_allenati}/{modello}/{stazione}/FI_{stazione}_{osservato}.png"
        plt.savefig(percorso_plot, dpi=300, bbox_inches='tight')
        os.system(f'convert {percorso_plot} -strip -colors 32 PNG8:{percorso_plot}')
        # plt.show()
        plt.close()
            
    # sss
print('\n\nDone')
