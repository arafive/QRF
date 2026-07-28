
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore', message='IProgress not found.*')
warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
warnings.filterwarnings('ignore', message='invalid value encountered in power')
warnings.filterwarnings('ignore', message='overflow encountered in exp')
warnings.filterwarnings('ignore', message='overflow encountered in multiply')

import os
import sys
import ast
import configparser
import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

import xgboost as xgb

plt.rc('font', weight='normal', size=6)

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF/pioggia')
os.chdir(cartella_lavoro)

from funzioni import f_salva_pickle
from funzioni import f_errori_classificazione
from funzioni import f_plot_heatmap_verifica
from funzioni import f_soglia_ottima_hss
from funzioni import f_precision_riscalata
from funzioni import f_aggiungi_indici_instabilita
from funzioni import f_aggiungi_rh_quota
from funzioni import f_aggiungi_wind_shear
from funzioni import f_aggiungi_tv_thetae
from funzioni import f_aggiungi_cape_cin
from funzioni import f_aggiungi_bulk_shear_profilo
from funzioni import f_aggiungi_omega_integrato
from funzioni import f_aggiungi_flusso_umidita
from funzioni import f_aggiungi_zero_termico_relativo
from funzioni import f_aggiungi_depressione_rugiada
from funzioni import f_aggiungi_frazione_convettiva
from funzioni import f_aggiungi_rapporto_raffica

from danilib import f_log_ciclo_for

from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.under_sampling import RandomUnderSampler

config = configparser.ConfigParser()
config.read('./../config.ini')

nome_modello = 'logReg'
soglia_pioggia = 0.2
frazione_validazione = 0.2 # non cambiare
colonne_da_logtrasformare = ['tp', 'cp']

modello = config.get('COMMON', 'modello')
cartella_dataset = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_dataset')}/{modello}")
cartella_modelli_allenati = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati")
os.makedirs(f'{cartella_modelli_allenati}/{modello}', exist_ok=True)

colori = {'0_24': 'tab:blue', '24_48': 'tab:orange', '48_72': 'tab:green'}

# %%

def f_modelli_cla(nome, scale_pos_weight=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # RandomUnderSampler scarta casualmente osservazioni della classe
    # maggioritaria (dry) fino a pareggiare le due classi - stesso approccio
    # di Cavaiola et al. 2024 (FlashNet), che tra oversampling/undersampling/SMOTE
    # ha trovato l'undersampling il piu' efficace su un problema analogo al tuo
    # (classificazione binaria rara da feature NWP).
    sampler = RandomUnderSampler(random_state=777)

    if nome == 'logReg':
        return make_pipeline_imb(
            sampler,
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(class_weight='balanced', max_iter=1000, penalty='elasticnet', solver='saga', l1_ratio=0.5)
        )
    elif nome == 'xgb':
        return make_pipeline_imb(
            sampler,
            xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                n_jobs=1, random_state=777
            )
        )
    else:
        raise ValueError('Modello non trovato')
    
df_stazioni = pd.read_csv(f'{cartella_lavoro}/../pioggia/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index:
    try:
        df_0_24 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_0_24.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f'\n\nLa stazione {stazione} non ha i dataset\n')
        continue

    df_24_48 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_24_48.csv', index_col=0, parse_dates=True)
    df_48_72 = pd.read_csv(f'{cartella_dataset}/df_{stazione}_48_72.csv', index_col=0, parse_dates=True)

    df_osservati = pd.read_csv(f'{cartella_lavoro}/../osservati/{stazione}.csv', index_col=0, parse_dates=True)['RAIN03HX']

    df_0_24 = pd.concat([df_0_24, df_osservati], axis=1).dropna()
    df_24_48 = pd.concat([df_24_48, df_osservati], axis=1).dropna()
    df_48_72 = pd.concat([df_48_72, df_osservati], axis=1).dropna()
    
    for intervallo, df_int in zip(['0_24', '24_48', '48_72'], [df_0_24, df_24_48, df_48_72]):

        for osservato in ['RAIN03HX']:
            f_log_ciclo_for([['Stazione ', stazione, df_stazioni.index]])
            df = df_int.copy()
            df = f_aggiungi_indici_instabilita(df)
            df = f_aggiungi_rh_quota(df)
            df = f_aggiungi_wind_shear(df, livello_basso='925', livello_alto='500')
            df = f_aggiungi_tv_thetae(df)
            df = f_aggiungi_cape_cin(df)
            df = f_aggiungi_bulk_shear_profilo(df)
            df = f_aggiungi_omega_integrato(df)
            df = f_aggiungi_flusso_umidita(df)
            df = f_aggiungi_zero_termico_relativo(df, df_stazioni.loc[stazione,'Altitude'])
            df = f_aggiungi_depressione_rugiada(df)
            df = f_aggiungi_frazione_convettiva(df)
            df = f_aggiungi_rapporto_raffica(df)

            colonne_numeriche = df.select_dtypes(include='number').columns
            n_inf = np.isinf(df[colonne_numeriche]).sum().sum()
            if n_inf > 0:
                raise ValueError(f"{stazione} - {intervallo}: {n_inf} valori infiniti trovati nelle feature")

            percorso_salvataggio = f'{cartella_modelli_allenati}/{modello}/{stazione}/{nome_modello}_{stazione}_{intervallo}_{osservato}.pkl'
            if os.path.exists(percorso_salvataggio) and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
                continue

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                X_train_full = df.loc[:'2024-12-31 21:00:00'].copy()
                X_test = df.loc['2025-01-01 00:00:00':].copy()
                y_train_full = X_train_full.pop(osservato)
                y_test = X_test.pop(osservato)
                y_test = (y_test >= soglia_pioggia).astype(int)
            else:
                X_train_full = df.copy()
                y_train_full = X_train_full.pop(osservato)

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                tp_raw_mm_test = X_test['tp'].copy()  # valore originale in mm, salvato PRIMA della trasformazione

            tp_raw_mm_train_full = X_train_full['tp'].copy()  # idem, serve per tarare la soglia del raw su X_val

            X_train_full[colonne_da_logtrasformare] = np.log1p(X_train_full[colonne_da_logtrasformare].clip(lower=0))
            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                X_test[colonne_da_logtrasformare] = np.log1p(X_test[colonne_da_logtrasformare].clip(lower=0))

            y_bin_train_full = (y_train_full >= soglia_pioggia).astype(int)

            # Split temporale (non casuale) training/validation: l'ultima frazione
            # cronologica serve solo per ottimizzare la soglia decisionale, mai
            # vista in fit dal classificatore.
            n_val = max(int(len(X_train_full) * frazione_validazione), 1)
            X_training = X_train_full.iloc[:-n_val]
            X_val = X_train_full.iloc[-n_val:]
            y_training = y_bin_train_full.iloc[:-n_val]
            y_val = y_bin_train_full.iloc[-n_val:]

            modello_base = f_modelli_cla(nome_modello)

            n_pos_training = int(y_training.sum())
            n_neg_training = len(y_training) - n_pos_training
            scale_pos_weight = n_neg_training / n_pos_training if n_pos_training > 0 else 1.0

            modello_base = f_modelli_cla(nome_modello, scale_pos_weight=scale_pos_weight)

            def f_hss_score(y_true, y_pred):
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                denominatore = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
                return 2 * (tp * tn - fp * fn) / denominatore if denominatore != 0 else 0.0

            scorer_hss = make_scorer(f_hss_score)

            griglia = {
                'xgb': {
                    'xgbclassifier__max_depth': [3, 5, 7, None],
                    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
                },
                'logReg': {
                    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
                    'logisticregression__l1_ratio': [0.0, 0.5, 1.0],
                },
            }[nome_modello]
            tscv = TimeSeriesSplit(n_splits=5)
            ricerca = GridSearchCV(modello_base, griglia, cv=tscv, scoring=scorer_hss, n_jobs=-1)
            ricerca.fit(X_training, y_training)
            modello_cla = ricerca.best_estimator_
            if nome_modello == 'logReg':
                print(f"{stazione} - {intervallo}: miglior C = {ricerca.best_params_['logisticregression__C']}")

            # Calibrazione delle probabilita' su X_val, mai visto dal modello
            # ne' durante il fit ne' durante la grid search sopra.
            modello_cla = CalibratedClassifierCV(FrozenEstimator(modello_cla), method='sigmoid')
            modello_cla.fit(X_val, y_val)

            prob_val = modello_cla.predict_proba(X_val)[:, 1]
            soglia_ottima, hss_ottimo = f_soglia_ottima_hss(y_val, prob_val)
            print(f"\n{stazione} - {intervallo}: soglia decisionale ottima ({nome_modello}) = {soglia_ottima:.3f} "
                  f"(HSS = {hss_ottimo:.3f}, su {int(y_val.sum())} eventi in validation)\n")

            tp_raw_mm_val = tp_raw_mm_train_full.iloc[-n_val:]
            soglia_ottima_raw, hss_ottimo_raw = f_soglia_ottima_hss(y_val, tp_raw_mm_val.values)
            print(f"{stazione} - {intervallo}: soglia decisionale ottima (RAW) = {soglia_ottima_raw:.3f}mm "
                  f"(HSS = {hss_ottimo_raw:.3f})\n")
            
            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                prob_test = modello_cla.predict_proba(X_test)[:, 1]
                previsione_cla = pd.Series((prob_test >= soglia_ottima).astype(int), index=X_test.index, name=nome_modello)

                df_raw = pd.DataFrame(tp_raw_mm_test.values, index=X_test.index, columns=[f'Raw_{modello}'])
                raw_bin = (df_raw[f'Raw_{modello}'] >= soglia_ottima_raw).astype(int)

                df_errori_cla = f_errori_classificazione(y_test, previsione_cla, nome_df=f'{stazione}_{intervallo}_{nome_modello}')
                df_errori_raw = f_errori_classificazione(y_test, raw_bin, nome_df=f'{stazione}_{intervallo}_RAW')

                df_verifica = pd.concat([df_errori_cla, df_errori_raw], axis=1).round(3)

                prevalenza_target = 0.5
                df_verifica.loc['precision[1]_risc_50'] = df_verifica.loc['precision[1]'].combine(
                    df_verifica.loc['prevalence[1]'],
                    lambda precision, prevalenza: f_precision_riscalata(precision, prevalenza, prevalenza_target)
                ).round(3)

                print(f"\n{tabulate(df_verifica, tablefmt='simple', headers='keys')}\n")
                f_plot_heatmap_verifica(df_verifica, titolo=f'{stazione} - {intervallo} - {nome_modello} (pioggia/no-pioggia) vs RAW')
                # from sklearn.calibration import calibration_curve
                
                # frac_pos, mean_pred = calibration_curve(y_val, prob_val, n_bins=10, strategy='uniform')
                # plt.figure(figsize=(4, 4))
                # plt.plot([0, 1], [0, 1], 'k--', lw=1)
                # plt.plot(mean_pred, frac_pos, 'o-', color='tab:blue', markersize=3)
                # plt.xlabel('Probabilità prevista (media per bin)')
                # plt.ylabel('Frequenza osservata')
                # plt.title(f'{stazione} - {intervallo} - Reliability diagram')
                # plt.show()
                # plt.close()

                # sss
            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello': modello_cla,
                'soglia_decisionale_ottima': float(soglia_ottima),
                'campi': X_training.columns.tolist()
            }

            f_salva_pickle(dict_model, percorso_salvataggio)
            sss

print('\n\nDone')
