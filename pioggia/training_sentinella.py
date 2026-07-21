import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore', message='IProgress not found.*')

import os
import ast
import configparser

import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.metrics import precision_recall_curve

plt.rc('font', weight='normal', size=6)

# cartella_lavoro = '/run/media/daniele.carnevale/Daniele2TB/repo/QRF'
cartella_lavoro = '/media/daniele/Daniele2TB/repo/QRF'
os.chdir(cartella_lavoro)

from funzioni import f_salva_pickle
from funzioni import f_apri_pickle
from funzioni import f_errori_classificazione
from funzioni import f_plot_heatmap_verifica

from danilib import f_log_ciclo_for


def f_soglia_ottima_fbeta(y_bin, prob, beta=1.0):
    """
    Cerca, tra le soglie di decisione candidate, quella che massimizza l'F-beta
    score (beta=1 -> F1; beta>1 pesa la recall piu' della precision - utile in
    ottica di Protezione Civile per non perdere eventi intensi, vedi appunti
    del collega). Usa precision_recall_curve invece di richiamare fbeta_score
    soglia per soglia: precision e recall sono gia' quelli esatti a ogni soglia
    candidata, quindi l'F-beta si calcola in un colpo solo su tutto l'array
    (stesso risultato del loop, molto piu' efficiente).
    """
    precisioni, richiami, soglie = precision_recall_curve(y_bin, prob)
    precisioni, richiami = precisioni[:-1], richiami[:-1]  # l'ultimo punto non ha soglia associata
    with np.errstate(divide='ignore', invalid='ignore'):
        fbeta = (1 + beta ** 2) * precisioni * richiami / ((beta ** 2) * precisioni + richiami)
    fbeta = np.nan_to_num(fbeta, nan=0.0)
    idx_ottimo = np.argmax(fbeta)
    return soglie[idx_ottimo], fbeta[idx_ottimo]


config = configparser.ConfigParser()
config.read('./config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = f"{config.get('COMMON', 'cartella_dataset')}/{modello}"
cartella_modelli_allenati_sentinella = f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati_sentinella"
os.makedirs(f'{cartella_modelli_allenati_sentinella}/{modello}', exist_ok=True)

# Soglia dedicata dell'evento estremo (target binario della sentinella). Se non
# presente in config.ini viene usato il fallback, ma e' meglio esplicitarla li'
# sotto [COMMON] come 'soglia_evento = 30.0'.
soglia_evento = float(config.get('COMMON', 'soglia_evento'))

# beta=1 -> F1 (come richiesto). Il collega suggerisce beta piu' alto (2, 3, 4)
# per pesare di piu' la recall quando non ci si puo' permettere di perdere un
# evento intenso, accettando qualche falso allarme in piu': cambialo qui se vuoi
# provarlo, senza toccare il resto dello script.
beta_fbeta = 1.0

# Frazione (cronologica, non casuale) del periodo di training riservata alla
# ricerca della soglia ottima: l'ultima parte in ordine temporale diventa
# validation, il resto resta training.
frazione_validazione = 0.2

# Sotto questi minimi la stima della soglia (o del modello stesso) sarebbe
# poco affidabile: si salta la stazione/intervallo con un avviso.
n_min_positivi_training = 20
n_min_positivi_validation = 5

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

            percorso_salvataggio = f'{cartella_modelli_allenati_sentinella}/{modello}/{stazione}/SENTINELLA_{stazione}_{intervallo}_{osservato}.pkl'
            if os.path.exists(percorso_salvataggio) and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
                continue

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                X_train_full = df.loc[:'2024-12-31 21:00:00']
                X_test = df.loc['2025-01-01 00:00:00':]
                y_train_full = X_train_full.pop(osservato)
                y_test = X_test.pop(osservato)
            else:
                X_train_full = df
                y_train_full = X_train_full.pop(osservato)

            # Split temporale (non casuale) train/validation dentro il periodo di
            # training: l'ultima frazione cronologica serve solo per ottimizzare
            # la soglia decisionale, mai vista in fit dal classificatore.
            n_val = max(int(len(X_train_full) * frazione_validazione), 1)
            X_training = X_train_full.iloc[:-n_val]
            X_val = X_train_full.iloc[-n_val:]
            y_training = y_train_full.iloc[:-n_val]
            y_val = y_train_full.iloc[-n_val:]

            y_bin_training = (y_training >= soglia_evento).astype(int)
            y_bin_val = (y_val >= soglia_evento).astype(int)

            n_pos_training = int(y_bin_training.sum())
            n_pos_val = int(y_bin_val.sum())
            if n_pos_training < n_min_positivi_training or n_pos_val < n_min_positivi_validation:
                print(f"\n{stazione} - {intervallo}: eventi >= {soglia_evento}mm insufficienti "
                      f"(training={n_pos_training}, validation={n_pos_val}, minimi richiesti "
                      f"{n_min_positivi_training}/{n_min_positivi_validation}) - salto la sentinella\n")
                continue

            n_neg_training = len(y_bin_training) - n_pos_training
            scale_pos_weight = n_neg_training / n_pos_training

            sentinella = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1, random_state=0,
            )
            sentinella.fit(X_training, y_bin_training)

            prob_val = sentinella.predict_proba(X_val)[:, 1]
            soglia_ottima, fbeta_ottimo = f_soglia_ottima_fbeta(y_bin_val, prob_val, beta=beta_fbeta)
            print(f"\n{stazione} - {intervallo}: soglia decisionale ottima = {soglia_ottima:.3f} "
                  f"(F{beta_fbeta:g}-score = {fbeta_ottimo:.3f}, su {n_pos_val} eventi in validation)\n")

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                y_bin_test = (y_test >= soglia_evento).astype(int)
                prob_test = sentinella.predict_proba(X_test)[:, 1]
                allerta_test = (prob_test >= soglia_ottima).astype(int)

                df_raw = pd.DataFrame(X_test['tp'].values, index=X_test.index, columns=[f'Raw_{modello}'])
                raw_bin = (df_raw[f'Raw_{modello}'] >= soglia_evento).astype(int)

                df_errori_sentinella = f_errori_classificazione(y_bin_test, allerta_test, nome_df=f'{stazione}_{intervallo}_SENTINELLA_{soglia_evento}mm')
                df_errori_raw = f_errori_classificazione(y_bin_test, raw_bin, nome_df=f'{stazione}_{intervallo}_RAW_{soglia_evento}mm')

                df_verifica = pd.concat([df_errori_sentinella, df_errori_raw], axis=1).round(3)
                print(f"\n{tabulate(df_verifica, tablefmt='simple', headers='keys')}\n")
                f_plot_heatmap_verifica(df_verifica, titolo=f'{stazione} - {intervallo} - Sentinella (soglia evento {soglia_evento}mm) vs RAW')

                fig, (ax_mm, ax_prob) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, height_ratios=[2, 1])

                ax_mm.plot(X_test.index, df_raw[f'Raw_{modello}'], color='tab:blue', lw=1.3, label=f'Raw_{modello}')
                ax_mm.plot(y_test.index, y_test, color='black', lw=1.3, ls='--', label='Obs')
                ax_mm.axhline(soglia_evento, color='tab:red', lw=0.9, ls=':')
                ax_mm.text(X_test.index[-1], soglia_evento, f' {soglia_evento}mm', color='tab:red', fontsize=7, va='center')
                ax_mm.set_ylabel('mm / 3h')
                ax_mm.legend(loc='upper left', fontsize=7)
                ax_mm.set_title(f'{stazione} - {intervallo} - Sentinella', loc='left', fontsize=8)

                ax_prob.plot(X_test.index, prob_test, color='tab:purple', lw=1.2, label='P(evento >= soglia)')
                ax_prob.axhline(soglia_ottima, color='grey', lw=0.8, ls=':', label=f'soglia ottima ({soglia_ottima:.2f})')
                ax_prob.set_ylim(0, 1)
                ax_prob.set_ylabel('Probabilita')
                ax_prob.legend(loc='upper right', fontsize=7)

                fig.tight_layout()
                plt.show()

            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati_sentinella}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello': sentinella,
                'soglia_evento': soglia_evento,
                'soglia_decisionale_ottima': float(soglia_ottima),
                'beta_fbeta': beta_fbeta,
                'campi': X_training.columns.tolist(),
            }

            f_salva_pickle(dict_model, percorso_salvataggio)

print('\n\nDone')
