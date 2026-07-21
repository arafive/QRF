"""
Variante "pooled" di training_cla_umido_soglia.py: invece di allenare un
classificatore per ogni singola stazione (che per soglie estreme puo' avere
troppo pochi positivi per essere affidabile), si concatenano i dataset di TUTTE
le stazioni in un unico grande dataset per ciascun intervallo (0_24/24_48/48_72),
e si allena un solo modello per intervallo su quel pool.

Due accortezze necessarie, che nella versione per-stazione non servivano:

1. Lo split temporale (train / validation / test) va fatto PER STAZIONE, sul suo
   indice datetime nativo, e SOLO DOPO si concatenano i pezzi tra stazioni. Se si
   concatenasse prima e si tagliasse dopo per posizione (.iloc[-n:]), il taglio
   prenderebbe stazioni intere invece che il periodo temporale piu' recente.

2. Stazioni diverse hanno righe con lo stesso timestamp: dopo la concatenazione
   l'indice ha duplicati. Un .loc[] su un indice duplicato ripesca le righe di
   TUTTE le stazioni con quel timestamp, non solo quella di origine (verificato:
   e' un bug reale, non teorico). Percio' l'indice viene resettato a interi subito
   dopo ogni concatenazione, prima di qualunque .loc[] successivo.

Nessun regressore/specialista, stesso obiettivo della versione per-stazione: un
classificatore binario con soglia decisionale ottimizzata per F1 (precision e
recall pesate allo stesso modo).
"""

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
    """Identica alle altre due script; duplicata qui per tenerlo autonomo."""
    precisioni, richiami, soglie = precision_recall_curve(y_bin, prob)
    precisioni, richiami = precisioni[:-1], richiami[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        fbeta = (1 + beta ** 2) * precisioni * richiami / ((beta ** 2) * precisioni + richiami)
    fbeta = np.nan_to_num(fbeta, nan=0.0)
    idx_ottimo = np.argmax(fbeta)
    return soglie[idx_ottimo], fbeta[idx_ottimo]


def f_soglia_massima_recall_con_precision_minima(y_bin, prob, precision_minima=0.7):
    """
    Tra le soglie di decisione candidate, sceglie quella che massimizza la recall
    fra tutte quelle che rispettano precision >= precision_minima. E' un punto
    diverso sulla STESSA curva precision-recall di f_soglia_ottima_fbeta - non
    serve riallenare il modello, solo scegliere diversamente la soglia.
    Se nessuna soglia raggiunge la precision richiesta, restituisce quella con
    la precision piu' alta ottenibile (e lo segnala con obiettivo_raggiungibile
    = False) invece di restituire un numero silenziosamente fuorviante.
    """
    precisioni, richiami, soglie = precision_recall_curve(y_bin, prob)
    precisioni, richiami = precisioni[:-1], richiami[:-1]

    maschera_ammissibili = precisioni >= precision_minima
    if not maschera_ammissibili.any():
        idx_migliore = np.argmax(precisioni)
        return soglie[idx_migliore], precisioni[idx_migliore], richiami[idx_migliore], False

    richiami_ammissibili = np.where(maschera_ammissibili, richiami, -1.0)
    idx_migliore = np.argmax(richiami_ammissibili)
    return soglie[idx_migliore], precisioni[idx_migliore], richiami[idx_migliore], True


config = configparser.ConfigParser()
config.read('./config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = f"{config.get('COMMON', 'cartella_dataset')}/{modello}"
cartella_modelli_allenati_cla_umido_soglia = f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati_cla_umido_soglia"
os.makedirs(f'{cartella_modelli_allenati_cla_umido_soglia}/{modello}/tutte_stazioni', exist_ok=True)

soglia_evento = float(config.get('COMMON', 'soglia_evento', fallback='30.0'))
soglia_raw_umido = 1.0
rapporto_secchi_su_informativi = 1.0
# Obiettivo: massima recall possibile con precision >= questo valore (invece di
# bilanciare precision/recall alla pari come faceva l'F1). Cambia qui se vuoi
# provare un altro compromesso.
precision_minima_target = 0.7
frazione_validazione = 0.2

# Con il pooling ci si aspettano molti piu' positivi di prima: questi minimi sono
# comunque tenuti come rete di sicurezza (es. se soglia_evento fosse alzata molto).
n_min_positivi_training = 50
n_min_positivi_validation = 20

# Aggiunge lat/lon/quota della stazione come feature: senza, il modello pooled
# non avrebbe alcun modo di sapere "dove" si trova (l'effetto orografico
# stazione-specifico andrebbe perso). Disattivabile con un flag se non lo vuoi.
includi_coordinate_stazione = True

rng = np.random.default_rng(0)
fai_il_test = ast.literal_eval(config.get('COMMON', 'fai_il_test'))
osservato = 'RAIN03HX'

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/pioggia/df_coordinate.csv', index_col=0)

################################################################################
# PASSO 1: carico e splitto (per stazione, cronologicamente) i dataset di TUTTE
# le stazioni, accumulando i pezzi separati per intervallo. Non alleno ancora
# nulla qui: serve prima raccogliere tutto per poter concatenare correttamente.
################################################################################

intervalli = ['0_24', '24_48', '48_72']
dati_per_intervallo = {i: {'X_training': [], 'y_training': [], 'X_val': [], 'y_val': [],
                            'X_test': [], 'y_test': []} for i in intervalli}
colonne_riferimento = {}

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

    f_log_ciclo_for([['Stazione (caricamento) ', stazione, df_stazioni.index]])

    for intervallo, df_int in zip(intervalli, [df_0_24, df_24_48, df_48_72]):
        df = df_int.copy()

        if includi_coordinate_stazione:
            df['lat_stazione'] = df_stazioni.loc[stazione, 'Latitude']
            df['lon_stazione'] = df_stazioni.loc[stazione, 'Longitude']
            df['quota_stazione'] = df_stazioni.loc[stazione, 'Altitude']

        # Controllo di coerenza: tutte le stazioni devono avere lo stesso set di
        # colonne per lo stesso intervallo, altrimenti pd.concat riempirebbe di
        # NaN in silenzio le colonne mancanti.
        if intervallo not in colonne_riferimento:
            colonne_riferimento[intervallo] = df.columns.tolist()
        elif df.columns.tolist() != colonne_riferimento[intervallo]:
            differenze = set(colonne_riferimento[intervallo]) ^ set(df.columns)
            print(f"\nATTENZIONE: {stazione} - {intervallo} ha colonne diverse dalle altre "
                  f"stazioni ({differenze}) - alllineo alle colonne comuni, ma controlla "
                  f"a monte se questo capita spesso\n")
            colonne_comuni = [c for c in colonne_riferimento[intervallo] if c in df.columns]
            df = df[colonne_comuni]

        if fai_il_test:
            X_train_full = df.loc[:'2024-12-31 21:00:00']
            X_test_st = df.loc['2025-01-01 00:00:00':]
            y_train_full = X_train_full.pop(osservato)
            y_test_st = X_test_st.pop(osservato)
        else:
            X_train_full = df
            y_train_full = X_train_full.pop(osservato)
            X_test_st = None
            y_test_st = None

        # Split temporale (non casuale) train/validation, per STAZIONE (vedi nota
        # in cima al file sul perche' non si puo' fare dopo la concatenazione).
        n_val = max(int(len(X_train_full) * frazione_validazione), 1)
        X_training_st = X_train_full.iloc[:-n_val]
        X_val_st = X_train_full.iloc[-n_val:]
        y_training_st = y_train_full.iloc[:-n_val]
        y_val_st = y_train_full.iloc[-n_val:]

        dati_per_intervallo[intervallo]['X_training'].append(X_training_st)
        dati_per_intervallo[intervallo]['y_training'].append(y_training_st)
        dati_per_intervallo[intervallo]['X_val'].append(X_val_st)
        dati_per_intervallo[intervallo]['y_val'].append(y_val_st)
        if fai_il_test:
            dati_per_intervallo[intervallo]['X_test'].append(X_test_st)
            dati_per_intervallo[intervallo]['y_test'].append(y_test_st)

################################################################################
# PASSO 2: per ciascun intervallo, concateno le stazioni e alleno UN SOLO
# modello sul pool completo.
################################################################################

for intervallo in intervalli:
    f_log_ciclo_for([['Intervallo (training pooled) ', intervallo, intervalli]])

    percorso_salvataggio = f'{cartella_modelli_allenati_cla_umido_soglia}/{modello}/tutte_stazioni/CLA_UMIDO_SOGLIA_{intervallo}_{osservato}.pkl'
    if os.path.exists(percorso_salvataggio) and not ast.literal_eval(config.get('COMMON', 'rifai_il_training')):
        continue

    if not dati_per_intervallo[intervallo]['X_training']:
        print(f"\n{intervallo}: nessuna stazione disponibile - salto\n")
        continue

    # Concateno tra stazioni e resetto SUBITO l'indice (vedi nota in cima al file:
    # senza reset, indici duplicati tra stazioni farebbero ripescare righe sbagliate
    # nei successivi .loc[]).
    X_training = pd.concat(dati_per_intervallo[intervallo]['X_training'], axis=0).reset_index(drop=True)
    y_training = pd.concat(dati_per_intervallo[intervallo]['y_training'], axis=0).reset_index(drop=True)
    X_val = pd.concat(dati_per_intervallo[intervallo]['X_val'], axis=0).reset_index(drop=True)
    y_val = pd.concat(dati_per_intervallo[intervallo]['y_val'], axis=0).reset_index(drop=True)

    y_bin_training = (y_training >= soglia_evento).astype(int)
    y_bin_val = (y_val >= soglia_evento).astype(int)

    n_pos_training = int(y_bin_training.sum())
    n_pos_val = int(y_bin_val.sum())
    print(f"\n{intervallo}: pool di {len(X_training)} righe training + {len(X_val)} validation "
          f"da {len(dati_per_intervallo[intervallo]['X_training'])} stazioni "
          f"({n_pos_training} positivi training, {n_pos_val} positivi validation)\n")

    if n_pos_training < n_min_positivi_training or n_pos_val < n_min_positivi_validation:
        print(f"\n{intervallo}: eventi >= {soglia_evento}mm insufficienti anche dopo il pooling "
              f"(training={n_pos_training}, validation={n_pos_val}) - salto\n")
        continue

    ################ Costruzione del training set bilanciato (stesso principio
    ################ della versione per-stazione, solo su dati pooled)

    idx_positivi = y_bin_training[y_bin_training == 1].index
    idx_negativi = y_bin_training[y_bin_training == 0].index

    raw_negativi = X_training.loc[idx_negativi, 'tp']
    idx_negativi_umidi = raw_negativi[raw_negativi >= soglia_raw_umido].index
    idx_negativi_secchi = raw_negativi[raw_negativi < soglia_raw_umido].index

    n_secchi_da_tenere = int(rapporto_secchi_su_informativi * (len(idx_positivi) + len(idx_negativi_umidi)))
    n_secchi_da_tenere = min(n_secchi_da_tenere, len(idx_negativi_secchi))
    if n_secchi_da_tenere > 0:
        idx_secchi_campionati = rng.choice(idx_negativi_secchi, size=n_secchi_da_tenere, replace=False)
    else:
        idx_secchi_campionati = np.array([], dtype=idx_negativi_secchi.dtype)

    idx_bilanciato = np.concatenate([idx_positivi.values, idx_negativi_umidi.values, idx_secchi_campionati])

    X_bilanciato = X_training.loc[idx_bilanciato]
    y_bin_bilanciato = y_bin_training.loc[idx_bilanciato]

    print(f"\n{intervallo}: training bilanciato -> {len(idx_positivi)} positivi, "
          f"{len(idx_negativi_umidi)} negativi umidi (tenuti tutti), {len(idx_secchi_campionati)} "
          f"negativi secchi campionati (su {len(idx_negativi_secchi)} disponibili)\n")

    ################ Training

    n_pos_bilanciato = int(y_bin_bilanciato.sum())
    n_neg_bilanciato = len(y_bin_bilanciato) - n_pos_bilanciato
    scale_pos_weight = n_neg_bilanciato / n_pos_bilanciato if n_pos_bilanciato > 0 else 1.0

    classificatore = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1, random_state=0,
    )
    classificatore.fit(X_bilanciato, y_bin_bilanciato)

    ################ Ricerca della soglia decisionale ottima su X_val (prevalenza reale)

    prob_val = classificatore.predict_proba(X_val)[:, 1]
    soglia_ottima, precision_val, recall_val, obiettivo_raggiungibile = f_soglia_massima_recall_con_precision_minima(
        y_bin_val, prob_val, precision_minima=precision_minima_target)

    if not obiettivo_raggiungibile:
        print(f"\nATTENZIONE {intervallo}: nessuna soglia sul validation raggiunge precision >= "
              f"{precision_minima_target} (massima precision ottenibile: {precision_val:.3f}). Uso "
              f"comunque la soglia a precision massima - il modello non ha abbastanza potere "
              f"discriminante per l'obiettivo richiesto, non e' un problema di soglia.\n")

    print(f"\n{intervallo}: soglia decisionale = {soglia_ottima:.3f} -> sul validation "
          f"precision={precision_val:.3f}, recall={recall_val:.3f} "
          f"(obiettivo: precision >= {precision_minima_target}, su {n_pos_val} eventi)\n")

    if fai_il_test:
        X_test = pd.concat(dati_per_intervallo[intervallo]['X_test'], axis=0).reset_index(drop=True)
        y_test = pd.concat(dati_per_intervallo[intervallo]['y_test'], axis=0).reset_index(drop=True)

        y_bin_test = (y_test >= soglia_evento).astype(int)
        prob_test = classificatore.predict_proba(X_test)[:, 1]
        allerta_test = (prob_test >= soglia_ottima).astype(int)

        raw_bin = (X_test['tp'] >= soglia_evento).astype(int)

        df_errori_cla = f_errori_classificazione(y_bin_test, allerta_test, nome_df=f'{intervallo}_CLA_UMIDO_{soglia_evento}mm')
        df_errori_raw = f_errori_classificazione(y_bin_test, raw_bin, nome_df=f'{intervallo}_RAW_{soglia_evento}mm')

        df_verifica = pd.concat([df_errori_cla, df_errori_raw], axis=1).round(3)
        print(f"\n{intervallo}: verifica sul test set pooled ({len(X_test)} punti, "
              f"{int(y_bin_test.sum())} eventi >= {soglia_evento}mm)\n"
              f"{tabulate(df_verifica, tablefmt='simple', headers='keys')}\n")

        precision_test = df_errori_cla.loc['precision[1]'].iloc[0]
        recall_test = df_errori_cla.loc['recall[1]'].iloc[0]
        obiettivo_raggiunto = 'OK' if precision_test >= precision_minima_target else f'sotto obiettivo (precision dovrebbe essere >= {precision_minima_target})'
        print(f"\n{intervallo}: sul test set pooled -> precision={precision_test:.3f}, "
              f"recall={recall_test:.3f} [{obiettivo_raggiunto}]\n")

        f_plot_heatmap_verifica(df_verifica, titolo=f'{intervallo} - Classificatore umido/soglia (pooled tutte stazioni) vs RAW')
        plt.show()

        sss

    ################################ Salvataggi

    os.makedirs(f'{cartella_modelli_allenati_cla_umido_soglia}/{modello}/tutte_stazioni', exist_ok=True)

    dict_model = {
        'modello': classificatore,
        'soglia_evento': soglia_evento,
        'soglia_raw_umido': soglia_raw_umido,
        'rapporto_secchi_su_informativi': rapporto_secchi_su_informativi,
        'soglia_decisionale_ottima': float(soglia_ottima),
        'strategia_soglia': 'massima_recall_con_precision_minima',
        'precision_minima_target': precision_minima_target,
        'includi_coordinate_stazione': includi_coordinate_stazione,
        'campi': X_bilanciato.columns.tolist(),
        'n_stazioni_pool': len(dati_per_intervallo[intervallo]['X_training']),
    }

    f_salva_pickle(dict_model, percorso_salvataggio)

print('\n\nDone')
