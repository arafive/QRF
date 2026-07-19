
import warnings

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

import os
import sys

import numpy as np
import pandas as pd

from tabulate import tabulate

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF')
os.chdir(cartella_lavoro)

from funzioni import f_query
from danilib import f_log_ciclo_for

# %%
df_tutte_le_coordinate = pd.DataFrame()

for cartella in ['vento', 'temperatura', 'umidita', 'pioggia']:
    df = pd.read_csv(f'./{cartella}/df_coordinate.csv', index_col=0)
    # >>> /media/daniele/Daniele2TB/repo/QRF$ vi */df_* # per toglierle a mano

    df_tutte_le_coordinate = pd.concat([df_tutte_le_coordinate, df], axis=0)

df_tutte_le_coordinate = df_tutte_le_coordinate[~df_tutte_le_coordinate.index.duplicated(keep='first')]
df_tutte_le_coordinate = df_tutte_le_coordinate.sort_index()

for stazione in df_tutte_le_coordinate.index:
    f_log_ciclo_for([['Stazione ', stazione, df_tutte_le_coordinate.index.tolist()]])

    percorso_csv = f'./osservati/{stazione}.csv'

    if os.path.exists(percorso_csv):
        ### Se il file esiste legge le ultime due righe, cancella l'ultima e riparte dalla penultima (Claude)
        with open(percorso_csv, 'rb') as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            chunk_size = min(filesize, 8192)
            while True:
                f.seek(-chunk_size, os.SEEK_END)
                tail = f.read()
                righe = tail.split(b'\n')
                newline_finale = bool(righe and righe[-1] == b'')
                if newline_finale:
                    righe.pop()  # scarta l'elemento vuoto dovuto al newline finale del file
                if len(righe) >= 2 or chunk_size == filesize:
                    break
                chunk_size = min(chunk_size * 2, filesize)

        if len(righe) >= 2:
            ultima_riga_bytes, penultima_riga_bytes = righe[-1], righe[-2]
            penultimo_timestamp = pd.to_datetime(penultima_riga_bytes.decode().split(',')[0])
            t0 = (penultimo_timestamp + pd.Timedelta(hours=1)).strftime('%Y%m%d%H%M')

            # tronca il file rimuovendo l'ultima riga: verrà riscritta dalla nuova query
            # -1 solo se il file terminava davvero con \n; se l'ultima riga era già
            # troncata a metà (scrittura precedente interrotta), non c'è newline da togliere
            nuova_dimensione = filesize - len(ultima_riga_bytes) - (1 if newline_finale else 0)
            with open(percorso_csv, 'r+b') as f:
                f.truncate(nuova_dimensione)
        else:
            # meno di due righe presenti: fallback sul comportamento precedente
            ultimo_timestamp = pd.to_datetime(righe[-1].decode().split(',')[0])
            t0 = (ultimo_timestamp + pd.Timedelta(hours=1)).strftime('%Y%m%d%H%M')

        try:
            df_query = f_query(stazione, t0=t0)
        except IndexError:
            # nessuna riga nuova disponibile per questa stazione
            df_query = pd.DataFrame()

        if not df_query.empty:
            df_query.to_csv(percorso_csv, index=True, header=False, mode='a', na_rep=np.nan)

    else:
        ### Se il file non esiste lo crea dal 2019-01-01 00:00:00 UTC
        df_query = f_query(stazione)

        """
        Se nell'intero periodo l'osservato X ha più del 75% di NaN ogni anno allora lo tolgo dalle stazioni idonee.
        """

        percentuale_nan = (
            df_query.isna()
              .groupby(df_query.index.year)
              .mean()
              .mul(100)
        ).round(1)
        # print(f"\n{tabulate(percentuale_nan, tablefmt='simple', headers='keys')}\n")

        percentuale_idonee = percentuale_nan.loc[:, (percentuale_nan < 75).all(axis=0)]
        percentuale_idonee.index.name = stazione
        # print(f"\n{tabulate(percentuale_idonee, tablefmt='simple', headers='keys')}\n")
        percentuale_NON_idonee = percentuale_nan.loc[:, (percentuale_nan >= 75).all(axis=0)]
        percentuale_NON_idonee.index.name = stazione
        # print(f"\n{tabulate(percentuale_NON_idonee, tablefmt='simple', headers='keys')}\n")

        for non_idonea in percentuale_NON_idonee.columns.tolist():
            if non_idonea == 'REHUM':
                df_coordinate = pd.read_csv('./umidita/df_coordinate.csv', index_col=0)
                df_coordinate = df_coordinate.drop(stazione, errors='ignore')
                df_coordinate.to_csv('./umidita/df_coordinate.csv', index=True, header=True, mode='w', na_rep=np.nan)

            elif non_idonea == 'WSPDM':
                df_coordinate = pd.read_csv('./vento/df_coordinate.csv', index_col=0)
                df_coordinate = df_coordinate.drop(stazione, errors='ignore')
                df_coordinate.to_csv('./vento/df_coordinate.csv', index=True, header=True, mode='w', na_rep=np.nan)

            elif non_idonea == 'TEMPM':
                df_coordinate = pd.read_csv('./temperatura/df_coordinate.csv', index_col=0)
                df_coordinate = df_coordinate.drop(stazione, errors='ignore')
                df_coordinate.to_csv('./temperatura/df_coordinate.csv', index=True, header=True, mode='w', na_rep=np.nan)

            elif non_idonea == 'RAIN01HX':
                df_coordinate = pd.read_csv('./pioggia/df_coordinate.csv', index_col=0)
                df_coordinate = df_coordinate.drop(stazione, errors='ignore')
                df_coordinate.to_csv('./pioggia/df_coordinate.csv', index=True, header=True, mode='w', na_rep=np.nan)

        df_query.to_csv(percorso_csv, index=True, header=True, mode='w', na_rep=np.nan)

print('\n\nDone')
