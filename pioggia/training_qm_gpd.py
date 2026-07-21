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
from funzioni import f_errori_regressione
from funzioni import CalibratoreEstremi

from danilib import f_log_ciclo_for

config = configparser.ConfigParser()
config.read('./config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = f"{config.get('COMMON', 'cartella_dataset')}/{modello}"
cartella_modelli_allenati = f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati"
os.makedirs(f'{cartella_modelli_allenati}/{modello}', exist_ok=True)

colori = {'0_24': 'tab:blue', '24_48': 'tab:orange', '48_72': 'tab:green'}

# Metodo di calibrazione fissato per questo script (vedi funzioni.py:CalibratoreEstremi
# per la spiegazione di tutti i metodi disponibili).
metodo_calibrazione = 'qm_gpd'
prefisso_salvataggio = 'QMGPD'

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

            percorso_salvataggio = f'{cartella_modelli_allenati}/{modello}/{stazione}/{prefisso_salvataggio}_{stazione}_{intervallo}_{osservato}.pkl'
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

            quantili = ast.literal_eval(config.get('COMMON', 'quantili'))

            calibratore = CalibratoreEstremi(metodo_calibrazione, quantili, soglia_prob=0.95, cap_fisico=150)
            calibratore.fit(X_training, y_training)

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                df_previsioni = calibratore.predict(X_test, quantili)
                df_previsioni['media'] = df_previsioni.mean(axis=1)
                df_previsioni.columns = [f'{metodo_calibrazione}_{x}' for x in df_previsioni.columns]

                df_raw = pd.DataFrame(X_test['tp'].values, index=X_test.index, columns=[f'Raw_{modello}'])

                df_errori_raw = f_errori_regressione(y_test, df_raw, nome_df=f'{stazione}_{intervallo}_raw_test')
                df_errori_prev = f_errori_regressione(y_test, df_previsioni[f'{metodo_calibrazione}_media'], nome_df=f'{stazione}_{intervallo}_prev_test')

                df_errori = pd.concat([df_errori_raw, df_errori_prev], axis=1).round(2)
                print(f"\n{tabulate(df_errori, tablefmt='simple', headers='keys')}\n")

                ################ Errori per fascia di precipitazione osservata
                # Fasce coerenti con la classificazione Debole/Moderata/Forte/Molto Forte
                # (< 15 / 15-55 / 55-75 / > 75 mm/3h).
                fasce = {
                    '0_15': (0, 15),
                    '15_55': (15, 55),
                    '55_75': (55, 75),
                    '>75': (75, np.inf),
                }

                for nome_fascia, (soglia_min, soglia_max) in fasce.items():
                    maschera = (y_test >= soglia_min) & (y_test < soglia_max)
                    if maschera.sum() == 0:
                        print(f"\nFascia {nome_fascia} mm: nessuna osservazione nel test set\n")
                        continue

                    df_errori_raw_fascia = f_errori_regressione(
                        y_test.loc[maschera], df_raw.loc[maschera], nome_df=f'{stazione}_{intervallo}_raw_{nome_fascia}')
                    df_errori_prev_fascia = f_errori_regressione(
                        y_test.loc[maschera], df_previsioni.loc[maschera, f'{metodo_calibrazione}_media'], nome_df=f'{stazione}_{intervallo}_prev_{nome_fascia}')

                    df_errori_fascia = pd.concat([df_errori_raw_fascia, df_errori_prev_fascia], axis=1).round(2)
                    print(f"\nFascia {nome_fascia} mm (n={maschera.sum()})\n{tabulate(df_errori_fascia, tablefmt='simple', headers='keys')}\n")

                col_q_alto = f'{metodo_calibrazione}_{max(quantili)}'
                df_plot = pd.concat([df_previsioni[f'{metodo_calibrazione}_media'], df_previsioni[col_q_alto], df_raw, y_test], axis=1)
                df_plot.columns = [f'{metodo_calibrazione}_media', f'{metodo_calibrazione}_q{max(quantili)}', 'Ecita', 'Obs']
                df_plot.plot(color=['tab:blue', 'tab:cyan', 'tab:red', 'black'], lw=2, style=['-', '-', '--', ':'])
                plt.title(f'{stazione} - {intervallo} - {metodo_calibrazione}')
                plt.tight_layout()
                plt.show()

                sss

            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello': calibratore,
                'campi': X_training.columns.tolist()
            }

            f_salva_pickle(dict_model, percorso_salvataggio)

    ################################ Feature Importance
    for osservato in ['RAIN03HX']:

        if os.path.exists(f"{cartella_modelli_allenati}/{modello}/{stazione}/FI_{prefisso_salvataggio}_{stazione}_{osservato}.png") and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
            continue

        serie = {}
        for intervallo in ['0_24', '24_48', '48_72']:
            percorso_pkl = f'{cartella_modelli_allenati}/{modello}/{stazione}/{prefisso_salvataggio}_{stazione}_{intervallo}_{osservato}.pkl'
            if not os.path.exists(percorso_pkl):
                serie = None
                break
            d = f_apri_pickle(percorso_pkl)
            try:
                serie[intervallo] = pd.Series(d['modello'].feature_importances_, index=d['campi'])
            except AttributeError:
                print(f"\nMetodo '{d['modello'].metodo}' senza feature_importances_: salto il plot FI per {stazione}\n")
                serie = None
                break

        if serie is None:
            continue

        # allineamento esplicito: forza l'ordine dei campi del primo intervallo
        df = pd.DataFrame(serie).reindex(serie['0_24'].index)

        # controllo anti-disallineamento
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
        ax.set_title(f'{modello} - {osservato} - {metodo_calibrazione}', loc='left', fontsize=7)

        ax.legend(
            title='Intervallo',
            prop={'size': 6},
            loc='upper right',
            ncols=3,
        )

        fig.tight_layout()
        percorso_plot = f"{cartella_modelli_allenati}/{modello}/{stazione}/FI_{prefisso_salvataggio}_{stazione}_{osservato}.png"
        plt.savefig(percorso_plot, dpi=300, bbox_inches='tight')
        os.system(f'convert {percorso_plot} -strip -colors 32 PNG8:{percorso_plot}')
        # plt.show()
        plt.close()

print('\n\nDone')
