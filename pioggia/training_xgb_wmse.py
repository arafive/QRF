import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore', message='IProgress not found.*')

import os
import ast
import configparser

import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

plt.rc('font', weight='normal', size=6)

# cartella_lavoro = '/run/media/daniele.carnevale/Daniele2TB/repo/QRF'
cartella_lavoro = '/media/daniele/Daniele2TB/repo/QRF'
os.chdir(cartella_lavoro)

from funzioni import f_salva_pickle
from funzioni import f_apri_pickle
from funzioni import f_errori_classificazione
from funzioni import f_plot_heatmap_verifica
from funzioni import ModelloWMSE
from funzioni import f_pesi_wmse

from danilib import f_log_ciclo_for

config = configparser.ConfigParser()
config.read('./config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = f"{config.get('COMMON', 'cartella_dataset')}/{modello}"
cartella_modelli_allenati = f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati"
os.makedirs(f'{cartella_modelli_allenati}/{modello}', exist_ok=True)

colori = {'0_24': 'tab:blue', '24_48': 'tab:orange', '48_72': 'tab:green'}

soglie = ast.literal_eval(config.get('COMMON', 'soglie_classificazione', fallback='[20, 30]'))
palette_soglie = ['#f2c744', '#f28c28', '#d62828', '#8b0000', '#4b0082']
colori_soglia = {soglia: palette_soglie[i % len(palette_soglie)] for i, soglia in enumerate(sorted(soglie))}

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/pioggia/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index:
    try:
        df_0_24 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_0_24.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f'\n\nLa stazione {stazione} non ha i dataset\n')
        continue

    df_24_48 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_24_48.csv', index_col=0, parse_dates=True)
    df_48_72 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_48_72.csv', index_col=0, parse_dates=True)

    df_osservati = pd.read_csv(f'{cartella_lavoro}/osservati/{stazione}.csv', index_col=0, parse_dates=True)['RAIN03HX']

    # !!! Per il momento devo togliere gli osservati dai dataframe concatenati,
    # !!! ma devo aggiornare la procedura di concatenzione.
    vecchi_osservati = ['Tmin', 'Tmean', 'Tmax', 'Direzione', 'Modulo', 'Raffica', 'Direzione raffica']
    df_0_24 = df_0_24.drop(columns=vecchi_osservati, errors='ignore')
    df_24_48 = df_24_48.drop(columns=vecchi_osservati, errors='ignore')
    df_48_72 = df_48_72.drop(columns=vecchi_osservati, errors='ignore')

    df_0_24 = pd.concat([df_0_24, df_osservati], axis=1).dropna()
    df_24_48 = pd.concat([df_24_48, df_osservati], axis=1).dropna()
    df_48_72 = pd.concat([df_48_72, df_osservati], axis=1).dropna()

    for intervallo, df_int in zip(['0_24', '24_48', '48_72'], [df_0_24, df_24_48, df_48_72]):

        for osservato in ['RAIN03HX']:
            f_log_ciclo_for([
                ['Stazione ', stazione, df_stazioni.index],
                ['Intervallo ', intervallo, ['0_24', '24_48', '48_72']]
                ])
            df = df_int.copy()

            percorso_salvataggio = f'{cartella_modelli_allenati}/{modello}/{stazione}/XGBWMSE_{stazione}_{intervallo}_{osservato}.pkl'
            if os.path.exists(percorso_salvataggio) and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
                continue

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                X_training = df.loc[:'2024-12-31 21:00:00']
                X_test = df.loc['2025-01-01 00:00:00':]
                y_training = X_training.pop(osservato)
                y_test = X_test.pop(osservato)

            else:
                X_training = df
                y_training = X_training.pop(osservato)

            modello_wmse = ModelloWMSE()
            modello_wmse.fit(X_training, y_training)

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                previsione_wmse = pd.Series(modello_wmse.predict(X_test), index=X_test.index, name='WMSE')

                df_raw = pd.DataFrame(X_test['tp'].values, index=X_test.index, columns=[f'Raw_{modello}'])

                pesi_test = f_pesi_wmse(y_test.values, modello_wmse.p50, modello_wmse.p99, modello_wmse.peso_max)
                wrmse = np.sqrt(np.average((previsione_wmse - y_test) ** 2, weights=pesi_test))
                print(f"\nWeighted RMSE (WMSE model): {wrmse:.3f}\n")

                liste_df_errori = []
                for soglia in soglie:
                    y_bin_test = (y_test > soglia).astype(int)

                    allerta = (previsione_wmse >= soglia).astype(int)
                    df_errori_soglia = f_errori_classificazione(y_bin_test, allerta, nome_df=f'{stazione}_{intervallo}_XGBWMSE_{soglia}mm')
                    liste_df_errori.append(df_errori_soglia)

                    raw_bin = (df_raw[f'Raw_{modello}'] > soglia).astype(int)
                    df_errori_raw_soglia = f_errori_classificazione(y_bin_test, raw_bin, nome_df=f'{stazione}_{intervallo}_RAW_{soglia}mm')
                    liste_df_errori.append(df_errori_raw_soglia)

                df_verifica = pd.concat(liste_df_errori, axis=1).round(3)
                print(f"\n{tabulate(df_verifica, tablefmt='simple', headers='keys')}\n")
                f_plot_heatmap_verifica(df_verifica, titolo=f'{stazione} - {intervallo} - XGB WMSE vs RAW')

                fig, ax_mm = plt.subplots(figsize=(9, 4))

                ax_mm.plot(X_test.index, df_raw[f'Raw_{modello}'], color='tab:blue', lw=1.3, label=f'Raw_{modello}')
                ax_mm.plot(y_test.index, y_test, color='black', lw=1.3, ls='--', label='Obs')
                ax_mm.plot(previsione_wmse.index, previsione_wmse, color='tab:purple', lw=1.3, label='Previsione WMSE')
                for soglia in soglie:
                    ax_mm.axhline(soglia, color=colori_soglia[soglia], lw=0.9, ls=':')
                    ax_mm.text(X_test.index[-1], soglia, f' {soglia}mm', color=colori_soglia[soglia], fontsize=7, va='center')
                ax_mm.set_ylabel('mm / 3h')
                ax_mm.legend(loc='upper left', fontsize=7)
                ax_mm.set_title(f'{stazione} - {intervallo} - XGB WMSE', loc='left', fontsize=8)

                fig.tight_layout()
                plt.show()

                sss

            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello': modello_wmse,
                'campi': X_training.columns.tolist(),
            }

            f_salva_pickle(dict_model, percorso_salvataggio)

print('\n\nDone')
