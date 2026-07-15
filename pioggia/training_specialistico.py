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

plt.rc('font', weight='normal', size=6)

# cartella_lavoro = '/run/media/daniele.carnevale/Daniele2TB/repo/QRF'
cartella_lavoro = '/media/daniele/Daniele2TB/repo/QRF'
os.chdir(cartella_lavoro)

from funzioni import f_salva_pickle
from funzioni import f_apri_pickle
from funzioni import f_errori_regressione
from funzioni import f_trasforma_target
from funzioni import f_trasforma_target_inversa

from danilib import f_log_ciclo_for

config = configparser.ConfigParser()
config.read('./config.ini')

modello = config.get('COMMON', 'modello')
cartella_dataset = f"{config.get('COMMON', 'cartella_dataset')}/{modello}"
cartella_modelli_allenati_specialistico = f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati_specialistico"
cartella_modelli_allenati_sentinella = f"{config.get('COMMON', 'cartella_modelli_allenati')}/pioggia/modelli_allenati_sentinella"
os.makedirs(f'{cartella_modelli_allenati_specialistico}/{modello}', exist_ok=True)

# Stessa soglia dedicata usata dalla sentinella (training_sentinella.py): e' il
# taglio che definisce l'hurdle, quindi deve essere la stessa nei due script.
# Se non presente in config.ini viene usato il fallback, ma e' meglio esplicitarla
# li' sotto [COMMON] come 'soglia_evento = 30.0'.
soglia_evento = float(config.get('COMMON', 'soglia_evento', fallback='30.0'))

# Trasformazione del target per ridurre l'asimmetria (vedi funzioni.f_trasforma_target).
# Qui il training e' gia' filtrato sugli eventi (niente zeri), ma la coda resta
# comunque lunga a destra: la radice cubica aiuta la regressione a non appiattirsi
# sui valori piu' bassi della fascia.
trasformazione_target = 'cbrt'

# Sotto questo minimo di eventi >= soglia_evento nel training, il regressore non
# avrebbe abbastanza dati per essere affidabile: si salta la stazione/intervallo.
n_min_positivi_training = 20

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

            percorso_salvataggio = f'{cartella_modelli_allenati_specialistico}/{modello}/{stazione}/SPECIALISTA_{stazione}_{intervallo}_{osservato}.pkl'
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

            # Cuore dell'hurdle: lo specialista vede SOLO i casi sopra soglia_evento,
            # cosi' non "appiattisce" la stima imparando anche dai giorni secchi o di
            # pioggia debole (vedi appunti - stesso taglio del target della sentinella).
            maschera_training = y_training >= soglia_evento
            X_training_specialista = X_training[maschera_training]
            y_training_specialista = y_training[maschera_training]

            n_positivi_training = len(y_training_specialista)
            if n_positivi_training < n_min_positivi_training:
                print(f"\n{stazione} - {intervallo}: solo {n_positivi_training} eventi >= {soglia_evento}mm nel "
                      f"training (minimo richiesto {n_min_positivi_training}) - salto lo specialista\n")
                continue

            specialista = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42,
            )
            y_training_trasformato = f_trasforma_target(y_training_specialista, trasformazione_target)
            specialista.fit(X_training_specialista, y_training_trasformato)

            if ast.literal_eval(config.get('COMMON', 'fai_il_test')):
                previsione_trasformata = specialista.predict(X_test)
                previsione_specialista = pd.Series(
                    f_trasforma_target_inversa(previsione_trasformata, trasformazione_target),
                    index=X_test.index, name='Specialista',
                )

                df_raw = pd.DataFrame(X_test['tp'].values, index=X_test.index, columns=[f'Raw_{modello}'])

                # Verifica SOLO sui casi realmente estremi nel test set: e' li' che lo
                # specialista viene usato in produzione (a valle del trigger della
                # sentinella). Valutarlo su tutto il test lo penalizzerebbe per errori
                # su giorni secchi che non e' mai stato addestrato a prevedere.
                maschera_test = y_test >= soglia_evento
                n_positivi_test = int(maschera_test.sum())

                if n_positivi_test == 0:
                    print(f"\n{stazione} - {intervallo}: nessun evento >= {soglia_evento}mm nel test set - salto la verifica\n")
                else:
                    df_errori_raw = f_errori_regressione(
                        y_test.loc[maschera_test], df_raw.loc[maschera_test], nome_df=f'{stazione}_{intervallo}_raw_estremi')
                    df_errori_specialista = f_errori_regressione(
                        y_test.loc[maschera_test], previsione_specialista.loc[maschera_test], nome_df=f'{stazione}_{intervallo}_specialista_estremi')

                    df_errori = pd.concat([df_errori_raw, df_errori_specialista], axis=1).round(2)
                    print(f"\n{stazione} - {intervallo}: verifica su {n_positivi_test} eventi >= {soglia_evento}mm nel test set\n"
                          f"{tabulate(df_errori, tablefmt='simple', headers='keys')}\n")

                # Previsione FINALE della pipeline hurdle completa (sentinella + specialista),
                # sullo stesso identico test set (split temporale fisso, invariato tra i due
                # script): sotto soglia decisionale -> 0, sopra -> stima dello specialista.
                # A differenza della verifica sopra (solo sui veri positivi), qui la valutazione
                # e' sull'intero test set: e' il numero che descrive cosa succede davvero in
                # produzione, incluso l'effetto filtro della sentinella sui falsi allarmi.
                percorso_sentinella = f'{cartella_modelli_allenati_sentinella}/{modello}/{stazione}/SENTINELLA_{stazione}_{intervallo}_{osservato}.pkl'

                if not os.path.exists(percorso_sentinella):
                    stima_hurdle = None
                    print(f"\n{stazione} - {intervallo}: sentinella non trovata in {percorso_sentinella} "
                          f"(allenala con training_sentinella.py) - nessuna previsione finale calcolabile\n")
                else:
                    dict_sentinella = f_apri_pickle(percorso_sentinella)
                    prob_evento_test = dict_sentinella['modello'].predict_proba(X_test[dict_sentinella['campi']])[:, 1]
                    soglia_decisionale = dict_sentinella['soglia_decisionale_ottima']
                    trigger_test = (prob_evento_test >= soglia_decisionale)

                    stima_hurdle = pd.Series(
                        np.where(trigger_test, previsione_specialista, 0.0),
                        index=X_test.index, name='Hurdle',
                    )

                    df_errori_raw_completo = f_errori_regressione(y_test, df_raw, nome_df=f'{stazione}_{intervallo}_raw_completo')
                    df_errori_hurdle_completo = f_errori_regressione(y_test, stima_hurdle, nome_df=f'{stazione}_{intervallo}_hurdle_completo')
                    df_errori_completo = pd.concat([df_errori_raw_completo, df_errori_hurdle_completo], axis=1).round(2)
                    print(f"\n{stazione} - {intervallo}: previsione FINALE (sentinella soglia {soglia_decisionale:.3f} "
                          f"+ specialista) sull'intero test set ({len(X_test)} punti)\n"
                          f"{tabulate(df_errori_completo, tablefmt='simple', headers='keys')}\n")

                fig, ax_mm = plt.subplots(figsize=(9, 4))

                ax_mm.plot(X_test.index, df_raw[f'Raw_{modello}'], color='tab:blue', lw=1.3, label=f'Raw_{modello}')
                ax_mm.plot(y_test.index, y_test, color='black', lw=1.3, ls='--', label='Obs')
                # ax_mm.plot(previsione_specialista.index, previsione_specialista, color='tab:red', lw=1.0, ls=':', label='Previsione Specialista (isolata)')
                if stima_hurdle is not None:
                    ax_mm.plot(stima_hurdle.index, stima_hurdle, color='tab:green', lw=1.3, label='Previsione FINALE (hurdle)')
                ax_mm.axhline(soglia_evento, color='grey', lw=0.9, ls=':')
                ax_mm.text(X_test.index[-1], soglia_evento, f' {soglia_evento}mm', color='grey', fontsize=7, va='center')
                ax_mm.set_ylabel('mm / 3h')
                ax_mm.legend(loc='upper left', fontsize=7)
                ax_mm.set_title(f'{stazione} - {intervallo} - Specialista (training solo su eventi >= {soglia_evento}mm)', loc='left', fontsize=8)

                fig.tight_layout()
                plt.show()
                sss

            ################################ Salvataggi

            os.makedirs(f'{cartella_modelli_allenati_specialistico}/{modello}/{stazione}', exist_ok=True)

            dict_model = {
                'modello': specialista,
                'trasformazione_target': trasformazione_target,
                'soglia_evento': soglia_evento,
                'campi': X_training_specialista.columns.tolist(),
            }

            f_salva_pickle(dict_model, percorso_salvataggio)

print('\n\nDone')
