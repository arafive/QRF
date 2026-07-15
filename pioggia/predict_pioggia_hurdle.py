
import os
import sys
import ast
import configparser

import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')

import numpy as np
import pandas as pd

# cartella_lavoro = '/run/media/daniele.carnevale/Daniele2TB/repo/QRF'
cartella_lavoro = '/media/daniele/Daniele2TB/repo/QRF'
os.chdir(cartella_lavoro)

from funzioni import f_apri_pickle
from funzioni import f_trasforma_target_inversa
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
cartella_modelli_allenati = config.get('COMMON', 'cartella_modelli_allenati')
cartella_dataset = config.get('COMMON', 'cartella_dataset')

# Stessa soglia dedicata usata in training_sentinella.py / training_specialistico.py:
# non serve per predire (le soglie decisionali sono gia' salvate nei pickle), ma la
# tengo qui solo per stamparla nel csv come riferimento.
soglia_evento = float(config.get('COMMON', 'soglia_evento', fallback='30.0'))

percorso_data = f"{data.strftime('%Y/%m/%d')}"

cartella_previsioni = f"{config.get('COMMON', 'cartella_previsioni')}/pioggia_hurdle/{modello}/{percorso_data}"
os.makedirs(cartella_previsioni, exist_ok=True)

# %%
df_stazioni = pd.read_csv(f'{cartella_lavoro}/pioggia/df_coordinate.csv', index_col=0)

for stazione in df_stazioni.index:
    if os.path.exists(f"{cartella_previsioni}/{stazione}.csv"):
        continue

    f_log_ciclo_for([['Stazione ', stazione, df_stazioni.index.tolist()]])

    cartella_sentinella_stazione = f"{cartella_modelli_allenati}/pioggia/modelli_allenati_sentinella/{modello}/{stazione}"
    cartella_specialista_stazione = f"{cartella_modelli_allenati}/pioggia/modelli_allenati_specialistico/{modello}/{stazione}"

    df_previsioni_tot = pd.DataFrame()

    for intervallo in ['0_24', '24_48', '48_72']:
        percorso_sentinella = f'{cartella_sentinella_stazione}/SENTINELLA_{stazione}_{intervallo}_RAIN03HX.pkl'
        percorso_specialista = f'{cartella_specialista_stazione}/SPECIALISTA_{stazione}_{intervallo}_RAIN03HX.pkl'

        if not os.path.exists(percorso_sentinella):
            print(f"\n{stazione} - {intervallo}: sentinella non allenata (eventi insufficienti in training?) - salto\n")
            continue

        dict_sentinella = f_apri_pickle(percorso_sentinella)

        percorso_csv = f"{cartella_dataset}/{modello}/df_{stazione}_{intervallo}_{data.strftime('%Y%m%d')}.csv"
        try:
            X = pd.read_csv(percorso_csv, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print(f"\n{stazione} - {intervallo}: dataset non trovato ({percorso_csv}) - salto\n")
            continue

        X = X[dict_sentinella['campi']]

        prob_evento = dict_sentinella['modello'].predict_proba(X)[:, 1]
        soglia_decisionale = dict_sentinella['soglia_decisionale_ottima']
        trigger = (prob_evento >= soglia_decisionale)

        if os.path.exists(percorso_specialista):
            dict_specialista = f_apri_pickle(percorso_specialista)
            previsione_trasf = dict_specialista['modello'].predict(X[dict_specialista['campi']])
            previsione_specialista = f_trasforma_target_inversa(previsione_trasf, dict_specialista['trasformazione_target'])
        else:
            print(f"\n{stazione} - {intervallo}: specialista non allenato - se la sentinella scatta, stima forzata a 0\n")
            previsione_specialista = np.zeros(len(X))

        # Cuore dell'hurdle in inferenza: sotto soglia decisionale -> 0, sopra ->
        # stima dello specialista (che non ha mai visto giorni secchi/deboli in training).
        stima_hurdle = np.where(trigger, previsione_specialista, 0.0)

        df_previsioni = pd.DataFrame({
            'P_evento': prob_evento,
            'Trigger': trigger.astype(int),
            'Hurdle': stima_hurdle,
            f'Raw_{modello}': X['tp'].values,
        }, index=X.index)

        df_previsioni_tot = pd.concat([df_previsioni_tot, df_previsioni], axis=0)

    if df_previsioni_tot.empty:
        print(f"\n{stazione}: nessun intervallo disponibile per {data} - nessun csv scritto\n")
        continue

    df_previsioni_tot = df_previsioni_tot.round(2)
    df_previsioni_tot.attrs['soglia_evento'] = soglia_evento  # solo a titolo informativo

    df_previsioni_tot.to_csv(f"{cartella_previsioni}/{stazione}.csv", index=True, header=True, mode='w', na_rep=np.nan)

print('\n\nDone')
