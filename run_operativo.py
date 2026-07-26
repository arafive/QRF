
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT
from config_percorsi_Daniele import ESEGUIBILE_PYTHON

cartella_estrazioni = os.path.join(CARTELLA_REPO_ROOT, 'estrazioni_puntuali')
cartella_QRF = os.path.join(CARTELLA_REPO_ROOT, 'QRF')

if len(sys.argv) > 1:
    data_arg = ' '.join(sys.argv[1:])
    oggi = pd.Timestamp(data_arg)
else:
    oggi = pd.Timestamp.today().normalize()
    
# for oggi in pd.date_range('2026-06-01', '2026-06-25', freq='1d'):

comando = f"{ESEGUIBILE_PYTHON} {cartella_estrazioni}/estrazioni.py {oggi.strftime('%Y-%m-%d')}"
print(comando, '\n')
os.system(comando)

comando = f"{ESEGUIBILE_PYTHON} {cartella_estrazioni}/concatenazioni.py {oggi.strftime('%Y-%m-%d')}"
print('\n', comando, '\n')
os.system(comando)

comando = f"{ESEGUIBILE_PYTHON} {cartella_QRF}/temperatura/predict.py {oggi.strftime('%Y-%m-%d')}"
print('\n', comando, '\n')
os.system(comando)

comando = f"{ESEGUIBILE_PYTHON} {cartella_QRF}/vento/predict.py {oggi.strftime('%Y-%m-%d')}"
print('\n', comando, '\n')
os.system(comando)

comando = f"{ESEGUIBILE_PYTHON} {cartella_QRF}/umidita/predict.py {oggi.strftime('%Y-%m-%d')}"
print('\n', comando, '\n')
os.system(comando)

print('\n\nDone')
