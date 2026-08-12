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

plt.rc('font', weight='normal', size=6)

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF/pioggia')
os.chdir(cartella_lavoro)

from funzioni import f_salva_pickle
from funzioni import f_apri_pickle
from funzioni import f_errori_regressione
from funzioni import f_gpd_quantile
from funzioni import f_gpd_media
from funzioni import f_obiettivo_gpd_sigma
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

config = configparser.ConfigParser()
config.read('./../config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_dataset')}/{modello}")
cartella_modelli_allenati = os.path.join(CARTELLA_REPO_ROOT, f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati")
os.makedirs(f'{cartella_modelli_allenati}/{modello}', exist_ok=True)

colori = {'0_24': 'tab:blue', '24_48': 'tab:orange', '48_72': 'tab:green'}

# Prefisso di salvataggio, sulla falsa riga di training_qrf.py (prefisso 'QRF').
prefisso_salvataggio = 'GPD'

# ============================================================================
# GPD via gradient boosting, ispirato a gbex (Velthoen, Dombry, Cai, Engelke -
# "Gradient boosting for extreme quantile regression", Extremes 2023).
# Il pacchetto originale e' R (github.com/JVelthoen/gbex); qui si reimplementa
# la logica centrale in Python con un obiettivo custom di XGBoost.
#
# SEMPLIFICAZIONE DELIBERATA rispetto al gbex originale: la forma (gamma)
# della GPD e' stimata UNA VOLTA SOLA via MLE sul pool degli eccessi di
# training (non dipende dalle covariate), mentre solo la scala (sigma) viene
# "boostata" in funzione delle covariate con un obiettivo custom (log-link,
# gradiente/hessiana della devianza GPD derivati analiticamente). Il gbex
# originale boosta ANCHE gamma in funzione delle covariate in modo coordinato -
# omesso qui perche' gamma e' notoriamente difficile da stimare con pochi dati
# (farlo dipendere dalle covariate richiede molti piu' eccessi di quanti se ne
# abbiano tipicamente per stazione/intervallo). Facilmente estendibile in un
# secondo momento se il volume di dati lo giustifica.
#
# Metodo: peaks-over-threshold. Si modella solo la coda oltre 'soglia_gpd_mm',
# definita come percentile SOGLIA_GPD_PERCENTILE della serie OSSERVATA DI
# TRAINING (mai del test, per evitare leakage).
# ============================================================================

SOGLIA_GPD_PERCENTILE = 0.90  # percentile della serie osservata sopra cui si modella la coda
N_MIN_ECCEDENZE_TRAINING = 30  # sotto questa soglia il fit GPD e' troppo rumoroso, si salta
QUANTILI_GPD = [0.5, 0.8, 0.9, 0.95]  # quantili della GPD condizionata da prevedere
colonne_da_logtrasformare = ['tp', 'cp']


# %%
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
                X_training = df.loc[:'2024-12-31 21:00:00'].copy()
                X_test = df.loc['2025-01-01 00:00:00':].copy()
                y_training = X_training.pop(osservato)
                y_test = X_test.pop(osservato)
                tp_raw_mm_test = X_test['tp'].copy()  # valore originale in mm, salvato PRIMA della trasformazione
            else:
                X_training = df.copy()
                y_training = X_training.pop(osservato)

            X_training[colonne_da_logtrasformare] = np.log1p(X_training[colonne_da_logtrasformare].clip(lower=0))
            if fai_il_test:
                X_test[colonne_da_logtrasformare] = np.log1p(X_test[colonne_da_logtrasformare].clip(lower=0))

            # Soglia POT: percentile della serie osservata DI TRAINING (mai del
            # test, per evitare leakage) - come richiesto, non un mm fisso.
            soglia_gpd_mm = float(np.quantile(y_training, SOGLIA_GPD_PERCENTILE))

            maschera_eccedenza = y_training > soglia_gpd_mm
            n_eccedenze = int(maschera_eccedenza.sum())
            if n_eccedenze < N_MIN_ECCEDENZE_TRAINING:
                print(f"\n{stazione} - {intervallo}: solo {n_eccedenze} eccedenze sopra {soglia_gpd_mm:.1f}mm "
                      f"(< {N_MIN_ECCEDENZE_TRAINING} richiesti) - salto\n")
                continue

            X_eccedenze = X_training.loc[maschera_eccedenza]
            z_eccedenze = (y_training.loc[maschera_eccedenza] - soglia_gpd_mm).values

            modello_gpd = GPDBoosting(n_estimators=200, max_depth=3, learning_rate=0.05)
            modello_gpd.fit(X_eccedenze, z_eccedenze)
            print(f"\n{stazione} - {intervallo}: soglia POT = {soglia_gpd_mm:.1f}mm ({SOGLIA_GPD_PERCENTILE:.0%} percentile), "
                  f"{n_eccedenze} eccedenze in training, gamma stimato = {modello_gpd.gamma_:.3f}\n")

            if fai_il_test:
                maschera_test_eccedenza = y_test > soglia_gpd_mm
                n_test_eccedenze = int(maschera_test_eccedenza.sum())

                if n_test_eccedenze == 0:
                    print(f"{stazione} - {intervallo}: nessuna eccedenza nel test set sopra {soglia_gpd_mm:.1f}mm - salto la verifica\n")
                else:
                    X_test_ecc = X_test.loc[maschera_test_eccedenza]
                    y_test_ecc = y_test.loc[maschera_test_eccedenza]

                    df_previsioni = modello_gpd.predict_quantili(X_test_ecc, QUANTILI_GPD)
                    df_previsioni = df_previsioni + soglia_gpd_mm  # da eccesso a mm assoluti
                    df_previsioni.columns = [f'gpd_{c}' if c != 'media' else 'gpd_media' for c in df_previsioni.columns]

                    df_raw = pd.DataFrame(tp_raw_mm_test.loc[maschera_test_eccedenza].values, index=X_test_ecc.index, columns=[f'Raw_{modello}'])

                    df_errori_raw = f_errori_regressione(y_test_ecc, df_raw, nome_df=f'{stazione}_{intervallo}_raw_test_gpd')
                    df_errori_gpd = f_errori_regressione(y_test_ecc, df_previsioni['gpd_media'], nome_df=f'{stazione}_{intervallo}_gpd_test')

                    df_errori = pd.concat([df_errori_raw, df_errori_gpd], axis=1).round(2)
                    print(f"\n{tabulate(df_errori, tablefmt='simple', headers='keys')}\n")

                    ################ Errori per fascia di precipitazione osservata
                    # Stesse fasce di training_qrf.py. Essendo il dataset gia'
                    # filtrato sopra la soglia POT, le fasce piu' basse (es.
                    # 0_15) possono risultare vuote a seconda del percentile
                    # scelto - gestito gia' dal controllo sotto.
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
                            y_test_ecc.loc[maschera_fascia], df_raw.loc[maschera_fascia], nome_df=f'{stazione}_{intervallo}_raw_{nome_fascia}')
                        df_errori_gpd_fascia = f_errori_regressione(
                            y_test_ecc.loc[maschera_fascia], df_previsioni.loc[maschera_fascia, 'gpd_media'], nome_df=f'{stazione}_{intervallo}_gpd_{nome_fascia}')

                        df_errori_fascia = pd.concat([df_errori_raw_fascia, df_errori_gpd_fascia], axis=1).round(2)
                        print(f"\nFascia {nome_fascia} mm (n={maschera_fascia.sum()})\n{tabulate(df_errori_fascia, tablefmt='simple', headers='keys')}\n")

                    ################ Combinazione con il classificatore pioggia si/no gia' allenato
                    # ATTENZIONE: il classificatore e' tarato sulla soglia wet/dry
                    # (0.2mm in training_cla.py), NON sulla soglia POT (90 percentile)
                    # del GPD - vedi spiegazione in chat. Questo e' un test diagnostico,
                    # non ancora la combinazione formalmente corretta.
                    nome_modello_classificatore = 'logReg'  # deve combaciare con nome_modello in training_cla.py
                    percorso_classificatore = f'{cartella_modelli_allenati}/{modello}/{stazione}/{nome_modello_classificatore}_{stazione}_{intervallo}_{osservato}.pkl'

                    if not os.path.exists(percorso_classificatore):
                        print(f"\n{stazione} - {intervallo}: nessun classificatore trovato in {percorso_classificatore} - salto la combinazione\n")
                    else:
                        dict_cla = f_apri_pickle(percorso_classificatore)
                        modello_cla = dict_cla['modello']
                        soglia_decisionale_cla = dict_cla['soglia_decisionale_ottima']
                        campi_cla = dict_cla['campi']

                        try:
                            prob_pioggia_test = modello_cla.predict_proba(X_test[campi_cla])[:, 1]
                        except KeyError as e:
                            print(f"\n{stazione} - {intervallo}: il classificatore si aspetta colonne non presenti in X_test ({e}) - salto la combinazione\n")
                            prob_pioggia_test = None

                        if prob_pioggia_test is not None:
                            previsione_wet_test = prob_pioggia_test >= soglia_decisionale_cla

                            # GPD applicato a TUTTO X_test (non solo alle eccedenze vere)
                            # per simulare un uso da previsione reale, dove non si conosce
                            # in anticipo se l'osservazione superera' la soglia POT.
                            df_previsioni_full = modello_gpd.predict_quantili(X_test, QUANTILI_GPD) + soglia_gpd_mm

                            previsione_combinata = pd.Series(
                                np.where(previsione_wet_test, df_previsioni_full['media'].values, 0.0),
                                index=X_test.index, name='gpd_combinato'
                            )

                            df_errori_raw_full = f_errori_regressione(y_test, pd.DataFrame(tp_raw_mm_test), nome_df=f'{stazione}_{intervallo}_raw_full')
                            df_errori_combinato = f_errori_regressione(y_test, previsione_combinata, nome_df=f'{stazione}_{intervallo}_combinato')
                            df_errori_full = pd.concat([df_errori_raw_full, df_errori_combinato], axis=1).round(2)
                            print(f"\nVerifica su TUTTO il test set (classificatore + GPD combinati):\n{tabulate(df_errori_full, tablefmt='simple', headers='keys')}\n")

                    df_plot = pd.concat([previsione_combinata, tp_raw_mm_test, y_test], axis=1)
                    df_plot.columns = ['gpd_combinato', f'Raw_{modello}', 'Obs']
                    df_plot.plot(color=['tab:blue', 'tab:cyan', 'tab:red', 'black'], lw=2, style=['-', '-', '--', ':'])
                    plt.title(f'{stazione} - {intervallo} - GPD (sopra {soglia_gpd_mm:.1f}mm, {SOGLIA_GPD_PERCENTILE:.0%} perc.)')
                    plt.tight_layout()
                    plt.show()

                    sss

            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello': modello_gpd,
                'soglia_gpd_mm': soglia_gpd_mm,
                'soglia_gpd_percentile': SOGLIA_GPD_PERCENTILE,
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
            serie[intervallo] = d['modello'].feature_importances()

        if serie is None:
            continue

        # allineamento esplicito: forza l'ordine dei campi del primo intervallo
        df = pd.DataFrame(serie).reindex(serie['0_24'].index)

        # a differenza di training_qrf.py, qui un NaN non segnala un
        # disallineamento anomalo: una feature puo' semplicemente non essere
        # mai stata scelta per uno split nel sigma-model di un dato
        # intervallo (importance 0 legittima), quindi si riempie invece di
        # sollevare eccezione.
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
        ax.set_title(f'{modello} - {osservato} - GPD (sigma boosting)', loc='left', fontsize=7)

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