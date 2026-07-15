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

from sklearn.metrics import brier_score_loss

plt.rc('font', weight='normal', size=6)

# cartella_lavoro = '/run/media/daniele.carnevale/Daniele2TB/repo/QRF'
cartella_lavoro = '/media/daniele/Daniele2TB/repo/QRF'
os.chdir(cartella_lavoro)

from funzioni import f_salva_pickle
from funzioni import f_apri_pickle
from funzioni import f_errori_classificazione
from funzioni import f_plot_heatmap_verifica
from funzioni import ModelloELR

from danilib import f_log_ciclo_for

config = configparser.ConfigParser()
config.read('./config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = f"{config.get('COMMON', 'cartella_dataset')}/{modello}"
cartella_modelli_allenati = f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati"
os.makedirs(f'{cartella_modelli_allenati}/{modello}', exist_ok=True)

colori = {'0_24': 'tab:blue', '24_48': 'tab:orange', '48_72': 'tab:green'}

soglie = ast.literal_eval(config.get('COMMON', 'soglie_classificazione', fallback='[10, 20, 30]'))
palette_soglie = ['#f2c744', '#f28c28', '#d62828', '#8b0000', '#4b0082']
colori_soglia = {soglia: palette_soglie[i % len(palette_soglie)] for i, soglia in enumerate(sorted(soglie))}
soglia_prob_allerta = float(config.get('COMMON', 'soglia_prob_allerta', fallback='0.3'))


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

            percorso_salvataggio = f'{cartella_modelli_allenati}/{modello}/{stazione}/ELR_{stazione}_{intervallo}_{osservato}.pkl'
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

            modello_elr = ModelloELR()
            modello_elr.fit(X_training, y_training, soglie_fit=soglie)

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                df_prob = pd.DataFrame(
                    {soglia: modello_elr.predict_proba_soglia(X_test, soglia) for soglia in soglie},
                    index=X_test.index,
                )

                df_raw = pd.DataFrame(X_test['tp'].values, index=X_test.index, columns=[f'Raw_{modello}'])

                liste_df_errori = []
                for soglia in soglie:
                    y_bin_test = (y_test > soglia).astype(int)

                    prob = df_prob[soglia].values
                    allerta = (prob >= soglia_prob_allerta).astype(int)
                    df_errori_soglia = f_errori_classificazione(y_bin_test, allerta, nome_df=f'{stazione}_{intervallo}_ELR_{soglia}mm')
                    df_errori_soglia.loc['brier'] = brier_score_loss(y_bin_test, prob) if y_bin_test.nunique() > 1 else np.nan
                    liste_df_errori.append(df_errori_soglia)

                    raw_bin = (df_raw[f'Raw_{modello}'] > soglia).astype(int)
                    df_errori_raw_soglia = f_errori_classificazione(y_bin_test, raw_bin, nome_df=f'{stazione}_{intervallo}_RAW_{soglia}mm')
                    df_errori_raw_soglia.loc['brier'] = np.nan
                    liste_df_errori.append(df_errori_raw_soglia)

                df_verifica = pd.concat(liste_df_errori, axis=1).round(3)
                print(f"\n{tabulate(df_verifica, tablefmt='simple', headers='keys')}\n")
                f_plot_heatmap_verifica(df_verifica, titolo=f'{stazione} - {intervallo} - ELR vs RAW')

                fig, (ax_mm, ax_prob) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, height_ratios=[2, 1])

                ax_mm.plot(X_test.index, df_raw[f'Raw_{modello}'], color='tab:blue', lw=1.3, label=f'Raw_{modello}')
                ax_mm.plot(y_test.index, y_test, color='black', lw=1.3, ls='--', label='Obs')
                for soglia in soglie:
                    ax_mm.axhline(soglia, color=colori_soglia[soglia], lw=0.9, ls=':')
                    ax_mm.text(X_test.index[-1], soglia, f' {soglia}mm', color=colori_soglia[soglia], fontsize=7, va='center')
                ax_mm.set_ylabel('mm / 3h')
                ax_mm.legend(loc='upper left', fontsize=7)
                ax_mm.set_title(f'{stazione} - {intervallo} - ELR', loc='left', fontsize=8)

                for soglia in soglie:
                    ax_prob.plot(df_prob.index, df_prob[soglia], color=colori_soglia[soglia], lw=1.5, label=f'P(> {soglia}mm)')
                ax_prob.axhline(soglia_prob_allerta, color='grey', lw=0.8, ls=':')
                ax_prob.set_ylim(0, 1)
                ax_prob.set_ylabel('Probabilita')
                ax_prob.legend(loc='upper right', fontsize=7, ncols=len(soglie))

                fig.tight_layout()
                plt.show()

                sss

            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello': modello_elr,
                'campi': X_training.columns.tolist(),
            }

            f_salva_pickle(dict_model, percorso_salvataggio)

print('\n\nDone')
