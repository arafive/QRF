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
from funzioni import f_apri_pickle
from funzioni import f_errori_regressione
from funzioni import f_errori_classificazione
from funzioni import f_plot_heatmap_verifica
from funzioni import f_soglia_ottima_hss
from funzioni import f_precision_riscalata
from funzioni import GPDBoosting
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

modello = config.get('COMMON', 'modello')
cartella_dataset = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_dataset')}/{modello}")
cartella_modelli_allenati = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati")
os.makedirs(f'{cartella_modelli_allenati}/{modello}', exist_ok=True)

colori = {'0_24': 'tab:blue', '24_48': 'tab:orange', '48_72': 'tab:green'}

# Prefisso unico di salvataggio per questa coppia classificatore+GPD (diverso
# da 'logReg'/'GPD' usati dagli script separati, per non sovrascriverli).
prefisso_salvataggio = 'CLAGPD'
nome_modello_cla = 'logReg'  # 'logReg' o 'xgb', stesse opzioni di training_cla.py

# ============================================================================
# Script unico: classificatore (pioggia estrema si/no) + GPD (magnitudo
# condizionata) + combinazione, entrambi tarati sulla STESSA soglia POT -
# a differenza del test diagnostico fatto in chat con il classificatore di
# training_cla.py (tarato sulla soglia wet/dry 0.2mm), che produceva un bias
# sistematico perche' le due soglie non combaciavano.
#
# Soglia POT: percentile SOGLIA_GPD_PERCENTILE della serie osservata di
# TRAINING (mai del test, per evitare leakage) - non un mm fisso.
# ============================================================================

SOGLIA_GPD_PERCENTILE = 0.90
N_MIN_ECCEDENZE_TRAINING = 20       # minimo eccedenze per il fit GPD (su tutto X_train_full)
N_MIN_POSITIVI_TRAINING = 20        # minimo eventi positivi per il classificatore (dopo lo split interno)
N_MIN_POSITIVI_VALIDATION = 5
QUANTILI_GPD = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # quantili della GPD condizionata da prevedere/salvare
QUANTILE_COMBINATO = 0.7            # quale colonna di QUANTILI_GPD (o 'media') usare nella combinazione finale
frazione_validazione = 0.2           # split temporale training/validation per il classificatore - non cambiare
colonne_da_logtrasformare = ['tp', 'cp']


def f_modelli_cla(nome, scale_pos_weight=1.0):
    """Identica a quella di training_cla.py."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

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


def f_hss_score(y_true, y_pred):
    """Identica a quella di training_cla.py - usata come scoring della GridSearchCV."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominatore = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return 2 * (tp * tn - fp * fn) / denominatore if denominatore != 0 else 0.0


scorer_hss = make_scorer(f_hss_score)


# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/../pioggia/df_coordinate.csv', index_col=0).iloc[25:30,:]

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
            f_log_ciclo_for([
                ['Stazione ', stazione, df_stazioni.index],
                ['Intervallo ', intervallo, ['0_24', '24_48', '48_72']]
                ])
            df = df_int.copy()
            df = f_aggiungi_indici_instabilita(df)
            df = f_aggiungi_rh_quota(df)
            df = f_aggiungi_wind_shear(df, livello_basso='925', livello_alto='500')
            df = f_aggiungi_tv_thetae(df)
            df = f_aggiungi_cape_cin(df)
            df = f_aggiungi_bulk_shear_profilo(df)
            df = f_aggiungi_omega_integrato(df)
            df = f_aggiungi_flusso_umidita(df)
            df = f_aggiungi_zero_termico_relativo(df, df_stazioni.loc[stazione, 'Altitude'])
            df = f_aggiungi_depressione_rugiada(df)
            df = f_aggiungi_frazione_convettiva(df)
            df = f_aggiungi_rapporto_raffica(df)

            colonne_numeriche = df.select_dtypes(include='number').columns
            n_inf = np.isinf(df[colonne_numeriche]).sum().sum()
            if n_inf > 0:
                raise ValueError(f"{stazione} - {intervallo}: {n_inf} valori infiniti trovati nelle feature")

            percorso_salvataggio = f'{cartella_modelli_allenati}/{modello}/{stazione}/{prefisso_salvataggio}_{stazione}_{intervallo}_{osservato}.pkl'
            if os.path.exists(percorso_salvataggio) and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
                continue

            fai_il_test = ast.literal_eval(config.get('COMMON', 'fai_il_test'))

            if fai_il_test:
                X_train_full = df.loc[:'2024-12-31 21:00:00'].copy()
                X_test = df.loc['2025-01-01 00:00:00':].copy()
                y_train_full = X_train_full.pop(osservato)
                y_test = X_test.pop(osservato)
                tp_raw_mm_test = X_test['tp'].copy()  # valore originale in mm, salvato PRIMA della trasformazione
            else:
                X_train_full = df.copy()
                y_train_full = X_train_full.pop(osservato)

            tp_raw_mm_train_full = X_train_full['tp'].copy()  # idem, serve per tarare la soglia raw su X_val_cla

            X_train_full[colonne_da_logtrasformare] = np.log1p(X_train_full[colonne_da_logtrasformare].clip(lower=0))
            if fai_il_test:
                X_test[colonne_da_logtrasformare] = np.log1p(X_test[colonne_da_logtrasformare].clip(lower=0))

            # ================================================================
            # Soglia POT unica, usata SIA dal classificatore SIA dal GPD -
            # questa e' la correzione rispetto al test diagnostico fatto in
            # chat (dove il classificatore usava 0.2mm e il GPD il 90 perc.).
            # ================================================================
            soglia_gpd_mm = float(np.quantile(y_train_full, SOGLIA_GPD_PERCENTILE))
            y_bin_train_full = (y_train_full > soglia_gpd_mm).astype(int)

            n_eccedenze_totali = int(y_bin_train_full.sum())
            if n_eccedenze_totali < N_MIN_ECCEDENZE_TRAINING:
                print(f"\n{stazione} - {intervallo}: solo {n_eccedenze_totali} eccedenze sopra {soglia_gpd_mm:.1f}mm "
                      f"(< {N_MIN_ECCEDENZE_TRAINING} richiesti) - salto stazione/intervallo\n")
                continue

            # ================================================================
            # STADIO 1 - Classificatore (P(y > soglia_gpd_mm | X))
            # Stesso schema di training_cla.py: split temporale training/
            # validation, GridSearchCV+TimeSeriesSplit su HSS, calibrazione
            # sigmoide su X_val, soglia decisionale ottima via HSS.
            # ================================================================
            n_val = max(int(len(X_train_full) * frazione_validazione), 1)
            X_training_cla = X_train_full.iloc[:-n_val]
            X_val_cla = X_train_full.iloc[-n_val:]
            y_training_cla = y_bin_train_full.iloc[:-n_val]
            y_val_cla = y_bin_train_full.iloc[-n_val:]

            n_pos_training = int(y_training_cla.sum())
            n_pos_val = int(y_val_cla.sum())
            if n_pos_training < N_MIN_POSITIVI_TRAINING or n_pos_val < N_MIN_POSITIVI_VALIDATION:
                print(f"\n{stazione} - {intervallo}: eventi > {soglia_gpd_mm:.1f}mm insufficienti per il classificatore "
                      f"(training={n_pos_training}, validation={n_pos_val}) - salto stazione/intervallo\n")
                continue

            n_neg_training = len(y_training_cla) - n_pos_training
            scale_pos_weight = n_neg_training / n_pos_training if n_pos_training > 0 else 1.0

            modello_base = f_modelli_cla(nome_modello_cla, scale_pos_weight=scale_pos_weight)

            griglia = {
                'xgb': {
                    'xgbclassifier__max_depth': [3, 5, 7],
                    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
                },
                'logReg': {
                    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
                    'logisticregression__l1_ratio': [0.0, 0.5, 1.0],
                },
            }[nome_modello_cla]
            tscv = TimeSeriesSplit(n_splits=5)
            ricerca = GridSearchCV(modello_base, griglia, cv=tscv, scoring=scorer_hss, n_jobs=-1)
            ricerca.fit(X_training_cla, y_training_cla)
            modello_cla = ricerca.best_estimator_

            modello_cla = CalibratedClassifierCV(FrozenEstimator(modello_cla), method='sigmoid')
            modello_cla.fit(X_val_cla, y_val_cla)

            prob_val_cla = modello_cla.predict_proba(X_val_cla)[:, 1]
            soglia_ottima_cla, hss_ottimo_cla = f_soglia_ottima_hss(y_val_cla, prob_val_cla)
            print(f"\n{stazione} - {intervallo}: soglia POT = {soglia_gpd_mm:.1f}mm ({SOGLIA_GPD_PERCENTILE:.0%} perc.), "
                  f"soglia decisionale classificatore = {soglia_ottima_cla:.3f} (HSS = {hss_ottimo_cla:.3f})\n")

            # ================================================================
            # STADIO 2 - GPD (magnitudo condizionata sopra soglia_gpd_mm)
            # Allenato su TUTTE le eccedenze di X_train_full (non solo la
            # sotto-porzione X_training_cla), per non sprecare dati scarsi -
            # il GPD non ha bisogno di un validation set proprio (nessuna
            # soglia da tarare, nessuna calibrazione di probabilita').
            # ================================================================
            maschera_eccedenza = y_train_full > soglia_gpd_mm
            X_eccedenze = X_train_full.loc[maschera_eccedenza]
            z_eccedenze = (y_train_full.loc[maschera_eccedenza] - soglia_gpd_mm).values

            modello_gpd = GPDBoosting(n_estimators=200, max_depth=3, learning_rate=0.05)
            modello_gpd.fit(X_eccedenze, z_eccedenze)
            print(f"{stazione} - {intervallo}: {n_eccedenze_totali} eccedenze in training, gamma stimato = {modello_gpd.gamma_:.3f}\n")

            if fai_il_test:
                # ============================================================
                # Verifica 1 - Classificatore da solo (stesso stile di training_cla.py)
                # ============================================================
                y_test_bin = (y_test > soglia_gpd_mm).astype(int)
                prob_test_cla = modello_cla.predict_proba(X_test)[:, 1]
                previsione_cla = pd.Series((prob_test_cla >= soglia_ottima_cla).astype(int), index=X_test.index, name=nome_modello_cla)

                tp_raw_mm_val = tp_raw_mm_train_full.iloc[-n_val:]
                soglia_ottima_raw, hss_ottimo_raw = f_soglia_ottima_hss(y_val_cla, tp_raw_mm_val.values)
                raw_bin = (tp_raw_mm_test >= soglia_ottima_raw).astype(int)

                df_errori_cla = f_errori_classificazione(y_test_bin, previsione_cla, nome_df=f'{stazione}_{intervallo}_{nome_modello_cla}')
                df_errori_raw_cla = f_errori_classificazione(y_test_bin, raw_bin, nome_df=f'{stazione}_{intervallo}_RAW')
                df_verifica_cla = pd.concat([df_errori_cla, df_errori_raw_cla], axis=1).round(3)

                prevalenza_target = 0.5
                df_verifica_cla.loc['precision[1]_risc_50'] = df_verifica_cla.loc['precision[1]'].combine(
                    df_verifica_cla.loc['prevalence[1]'],
                    lambda precision, prevalenza: f_precision_riscalata(precision, prevalenza, prevalenza_target)
                ).round(3)

                print(f"\nVerifica classificatore (evento > {soglia_gpd_mm:.1f}mm):\n{tabulate(df_verifica_cla, tablefmt='simple', headers='keys')}\n")
                f_plot_heatmap_verifica(df_verifica_cla, titolo=f'{stazione} - {intervallo} - {nome_modello_cla} (evento > {soglia_gpd_mm:.1f}mm) vs RAW')

                # ============================================================
                # Verifica 2 - GPD da solo, sulle VERE eccedenze del test set
                # (stesso stile di training_gpd.py)
                # ============================================================
                maschera_test_eccedenza = y_test > soglia_gpd_mm
                n_test_eccedenze = int(maschera_test_eccedenza.sum())

                df_previsioni_gpd_full = modello_gpd.predict_quantili(X_test, QUANTILI_GPD) + soglia_gpd_mm
                df_previsioni_gpd_full.columns = [f'gpd_{c}' if c != 'media' else 'gpd_media' for c in df_previsioni_gpd_full.columns]

                if n_test_eccedenze == 0:
                    print(f"{stazione} - {intervallo}: nessuna eccedenza nel test set sopra {soglia_gpd_mm:.1f}mm - salto la verifica GPD pura\n")
                else:
                    X_test_ecc = X_test.loc[maschera_test_eccedenza]
                    y_test_ecc = y_test.loc[maschera_test_eccedenza]
                    df_previsioni_ecc = df_previsioni_gpd_full.loc[maschera_test_eccedenza]

                    df_raw_ecc = pd.DataFrame(tp_raw_mm_test.loc[maschera_test_eccedenza].values, index=X_test_ecc.index, columns=[f'Raw_{modello}'])

                    df_errori_raw_gpd = f_errori_regressione(y_test_ecc, df_raw_ecc, nome_df=f'{stazione}_{intervallo}_raw_gpd')
                    df_errori_gpd = f_errori_regressione(y_test_ecc, df_previsioni_ecc['gpd_media'], nome_df=f'{stazione}_{intervallo}_gpd')
                    df_errori_gpd_tot = pd.concat([df_errori_raw_gpd, df_errori_gpd], axis=1).round(2)
                    print(f"\nVerifica GPD sulle vere eccedenze di test (n={n_test_eccedenze}):\n{tabulate(df_errori_gpd_tot, tablefmt='simple', headers='keys')}\n")

                    fasce = {
                        '0_15': (0, 15),
                        '15_55': (15, 55),
                        '55_75': (55, 75),
                        '>75': (75, np.inf),
                    }
                    for nome_fascia, (soglia_min, soglia_max) in fasce.items():
                        maschera_fascia = (y_test_ecc >= soglia_min) & (y_test_ecc < soglia_max)
                        if maschera_fascia.sum() == 0:
                            print(f"\nFascia {nome_fascia} mm: nessuna osservazione nel test set\n")
                            continue
                        df_errori_raw_fascia = f_errori_regressione(
                            y_test_ecc.loc[maschera_fascia], df_raw_ecc.loc[maschera_fascia], nome_df=f'{stazione}_{intervallo}_raw_{nome_fascia}')
                        df_errori_gpd_fascia = f_errori_regressione(
                            y_test_ecc.loc[maschera_fascia], df_previsioni_ecc.loc[maschera_fascia, 'gpd_media'], nome_df=f'{stazione}_{intervallo}_gpd_{nome_fascia}')
                        df_errori_fascia = pd.concat([df_errori_raw_fascia, df_errori_gpd_fascia], axis=1).round(2)
                        print(f"\nFascia {nome_fascia} mm (n={maschera_fascia.sum()})\n{tabulate(df_errori_fascia, tablefmt='simple', headers='keys')}\n")

                # ============================================================
                # Verifica 3 - Combinazione su TUTTO il test set: dove il
                # classificatore dice "evento estremo", si usa la previsione
                # GPD (colonna QUANTILE_COMBINATO); altrove si usa il raw
                # (nessun modello per la magnitudo "ordinaria" e' stato
                # allenato qui - il raw resta la miglior stima disponibile
                # per quel regime).
                # ============================================================
                previsione_estremo_test = prob_test_cla >= soglia_ottima_cla
                col_combinata = 'gpd_media' if QUANTILE_COMBINATO == 'media' else f'gpd_{QUANTILE_COMBINATO}'

                previsione_combinata = pd.Series(
                    np.where(previsione_estremo_test, df_previsioni_gpd_full[col_combinata].values, tp_raw_mm_test.values),
                    index=X_test.index, name='combinato'
                )

                df_errori_raw_full = f_errori_regressione(y_test, pd.DataFrame(tp_raw_mm_test), nome_df=f'{stazione}_{intervallo}_raw_full')
                df_errori_combinato = f_errori_regressione(y_test, previsione_combinata, nome_df=f'{stazione}_{intervallo}_combinato')
                df_errori_full = pd.concat([df_errori_raw_full, df_errori_combinato], axis=1).round(2)
                print(f"\nVerifica su TUTTO il test set (classificatore + GPD combinati, quantile={QUANTILE_COMBINATO}):\n"
                      f"{tabulate(df_errori_full, tablefmt='simple', headers='keys')}\n")

                df_plot = pd.concat([previsione_combinata, tp_raw_mm_test, y_test], axis=1)
                df_plot.columns = ['combinato', f'Raw_{modello}', 'Obs']
                df_plot.plot(color=['tab:blue', 'tab:red', 'black'], lw=2, style=['-', '--', ':'])
                plt.title(f'{stazione} - {intervallo} - Classificatore+GPD (soglia {soglia_gpd_mm:.1f}mm, {SOGLIA_GPD_PERCENTILE:.0%} perc.)')
                plt.tight_layout()
                plt.show()

                sss

            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello_cla': modello_cla,
                'soglia_decisionale_cla': float(soglia_ottima_cla),
                'modello_gpd': modello_gpd,
                'soglia_gpd_mm': soglia_gpd_mm,
                'soglia_gpd_percentile': SOGLIA_GPD_PERCENTILE,
                'quantile_combinato': QUANTILE_COMBINATO,
                'campi': X_training_cla.columns.tolist()
            }

            f_salva_pickle(dict_model, percorso_salvataggio)

    ################################ Feature Importance (solo GPD - il
    # classificatore logReg non ha un equivalente diretto di feature_importances_
    # comparabile in scala con quello del GPD; se serve un giorno si puo'
    # aggiungere un blocco separato basato sui coefficienti standardizzati)
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
            serie[intervallo] = d['modello_gpd'].feature_importances()

        if serie is None:
            continue

        df = pd.DataFrame(serie).reindex(serie['0_24'].index)
        df = df.fillna(0.0)

        fig, ax = plt.subplots()
        df.plot.bar(ax=ax, zorder=10, width=0.8, color=[colori[c] for c in df.columns])
        ax.set_ylabel("Importance (gain)")
        ax.tick_params(axis='x', labelsize=4.5)
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.4, alpha=0.5, zorder=-10)

        nome_stazione = df_stazioni.loc[stazione]['Name']
        lat_lon = f"{df_stazioni.loc[stazione]['Latitude'].round(2)}, {df_stazioni.loc[stazione]['Longitude'].round(2)}"
        quota = int(df_stazioni.loc[stazione]['Altitude'])
        zona = df_stazioni.loc[stazione]['zona_allertamento']
        titolo = f"{nome_stazione}, {quota} metri ({lat_lon}) - Zona {zona}"
        ax.set_title(titolo, loc='right', fontsize=7)
        ax.set_title(f'{modello} - {osservato} - {prefisso_salvataggio} (sigma GPD boosting)', loc='left', fontsize=7)

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
        plt.close()

print('\n\nDone')
