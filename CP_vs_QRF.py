"""
Creazione: Tue Sep 22 14:13:06 2026
Autore: daniele.carnevale
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~/.config'))
from config_percorsi_Daniele import CARTELLA_REPO_ROOT

cartella_lavoro = os.path.join(CARTELLA_REPO_ROOT, 'QRF')
os.chdir(cartella_lavoro)

# cartella_osservati = "/home/cfmi.arpal.org/daniele.carnevale/Scrivania/QRF/osservati"
cartella_osservati = "/media/daniele/Daniele2TB/repo/QRF/osservati"

def f_errori_regressione(osservato, previsto, nome_df=None):
    """Funzione per gli errori di regressione."""
    df = pd.DataFrame(columns=[nome_df])

    previsto = np.array(previsto).squeeze()
    osservato = np.array(osservato).squeeze()

    try:
        dimensione_dei_dati = len(osservato)
    except TypeError:
        dimensione_dei_dati = 1

    errore = previsto - osservato

    df.loc['bias'] = (1 / dimensione_dei_dati) * np.sum(errore)
    df.loc['rmse']    = np.sqrt(np.sum(errore ** 2) / dimensione_dei_dati)
    df.loc['pearson'] = np.sum((previsto - np.mean(previsto)) * (osservato - np.mean(osservato))) / np.sqrt(np.sum((previsto - np.mean(previsto)) ** 2) * np.sum((osservato - np.mean(osservato)) ** 2))

    return df


def f_trova_outlier_mad(df, soglia=3.5):
    """Trova outlier con il metodo MAD e li segnala esplicitamente."""
    
    mediana = df.median()
    mad = (df - mediana).abs().median()
    
    z_robusto = 0.6745 * (df - mediana) / mad
    outlier_mask = z_robusto.abs() > soglia
    
    if outlier_mask.any().any():
        righe, colonne = np.where(outlier_mask)
        for r, c in zip(righe, colonne):
            idx = df.index[r]
            col = df.columns[c]
            print(f"Outlier: indice={idx}, colonna={col}, valore={df.iloc[r, c]:.2f}, z={z_robusto.iloc[r, c]:.2f}")
    else:
        print("Nessun outlier trovato.")
    
    return df[outlier_mask.any(axis=1)]


df_coordinate = pd.read_csv('temperatura/df_coordinate.csv', index_col=0, parse_dates=True)

# %%
# def f_settaggio_db_arpal():
#     """
#     Ritorna il settaggio per il collegamento al database.

#     Returns
#     -------
#     connessione : oracledb.Connection
#         Connessione da passare a pd.read_sql().

#     """
#     import oracledb
#     dsnStr = oracledb.makedsn('cfmi_db.regione.liguria.it', '1522', 'cfmi')
#     connessione = oracledb.connect(user='cmi', password='cmi', dsn=dsnStr)

#     return connessione

# connessione = f_settaggio_db_arpal()

# query = """
# SELECT      kss.omirl_code,
#             a.label,
#             b.publication_date,
#             t.day,
#             t.temperature_max,
#             t.temperature_min  
# FROM        ligurian_bulletin.temperatures AS t
# JOIN        ligurian_bulletin.areas AS a ON t.area_id = a.id
# JOIN        ligurian_bulletin.bulletins AS b ON t.bulletin_id = b.id
# LEFT JOIN   ligurian_bulletin.kalman_standard_stations AS kss ON a.kalman_code_id = kss.id
# WHERE       t.day >= '2026-01-01' AND
#             b.publication_date IS NOT NULL
# ORDER BY    t.bulletin_id, t.day;

# """

# df_query = pd.read_sql(query, con=connessione)

# %%

with open('temperatures_202609221247.json', 'r') as f:
    testo = f.read().strip()

testo = testo[1:-1]
dati = json.loads(testo)
previsioni_CP = pd.DataFrame(dati)

previsioni_CP['publication_date'] = pd.to_datetime(previsioni_CP['publication_date']).dt.floor('D').dt.tz_localize(None)
previsioni_CP['day'] = pd.to_datetime(previsioni_CP['day'])

stazioni = previsioni_CP['omirl_code'].unique()
emissioni = [pd.Timestamp(x) for x in previsioni_CP['publication_date'].unique()][2:]

dict_CP = {
    x: 
        {y: pd.DataFrame(
            np.nan,
            index=emissioni,
            columns=['d0', 'd1', 'd2']
            ) for y in ['tmin', 'tmax']
         } for x in stazioni
    }
    
for data in emissioni:
    
    for s in stazioni:

        df = previsioni_CP[previsioni_CP['publication_date'] == data]
        df = df[df['omirl_code'] == s]
        df = df.sort_values('day')
        
        for i in range(min(3, len(df))):
            dict_CP[s]['tmin'].loc[data, f'd{i}'] = float(df.iloc[i]['temperature_min'])
            dict_CP[s]['tmax'].loc[data, f'd{i}'] = float(df.iloc[i]['temperature_max'])

# %%

dict_OBS = {
    x: 
        {y: pd.DataFrame(
            np.nan,
            index=emissioni,
            columns=['d0', 'd1', 'd2']
            ) for y in ['tmin', 'tmax']
         } for x in stazioni
    }
    
for s in stazioni:
    
    file = f"{cartella_osservati}/{s}.csv"
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    
    gruppi = {giorno_data: giorno for giorno_data, giorno in df.groupby(df.index.normalize())}
    
    for data in emissioni:
        for offset in range(3):
            giorno_data = data + pd.Timedelta(days=offset)
            if giorno_data in gruppi:
                giorno = gruppi[giorno_data]
                dict_OBS[s]['tmin'].loc[data, f'd{offset}'] = giorno.between_time('00:00', '08:00')['TEMPN'].min()
                dict_OBS[s]['tmax'].loc[data, f'd{offset}'] = giorno.between_time('08:00', '18:00')['TEMPX'].max()

# %%

dict_QRF = {
    x: 
        {y: pd.DataFrame(
            np.nan,
            index=emissioni,
            columns=['d0', 'd1', 'd2']
            ) for y in ['tmin', 'tmax']
         } for x in stazioni
    }

for data in emissioni:
    
    for s in stazioni:
        try:
            file = f"../MeteoBricchi/dati1D/temperatura/ecita/{data.strftime('%Y/%m/%d')}/{s}/{s}.csv"
            df = pd.read_csv(file, index_col=0, parse_dates=True)
        except FileNotFoundError:
            try:
                file = f"../MeteoBricchi/dati1D/temperatura/ecita/{data.strftime('%Y/%m/%d')}/{s}.csv"
                df = pd.read_csv(file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                continue

        obs_min = dict_OBS[s]['tmin'].loc[data].mean()
        obs_max = dict_OBS[s]['tmax'].loc[data].mean()
        
        colonne_temp = [c for c in df.columns if 'Tmin' in c or 'Tmax' in c or 'minima' in c or 'massima' in c]
        df_temp = df[colonne_temp] if colonne_temp else df
        
        mediana_df = df_temp.median().median()
        mediana_obs = np.nanmedian([obs_min, obs_max])
        
        ratio = mediana_df / mediana_obs if mediana_obs != 0 else np.nan
        
        if pd.notna(ratio) and 5 < ratio < 20:
            df = df / 10
        elif pd.notna(ratio) and not (0.5 < ratio < 2 or 5 < ratio < 20):
            print(f"ATTENZIONE: rapporto scala insolito per {s}, {data}: ratio={ratio:.2f}")
            
        df_giornaliero = df.groupby((df.index - pd.Timedelta(hours=1)).date)
        
        for i, (_, giorno) in enumerate(df_giornaliero):
            try:
                dict_QRF[s]['tmin'].loc[data, f'd{i}'] = giorno.between_time('00:00', '08:00')['QRF Tmin'].min()
                dict_QRF[s]['tmax'].loc[data, f'd{i}'] = giorno.between_time('08:00', '18:00')['QRF Tmax'].max()
            except KeyError:
                dict_QRF[s]['tmin'].loc[data, f'd{i}'] = giorno.between_time('00:00', '08:00')['QRF minima'].min()
                dict_QRF[s]['tmax'].loc[data, f'd{i}'] = giorno.between_time('08:00', '18:00')['QRF massima'].max()

# %% Calcolo errori

dict_errori = {x: {y: pd.DataFrame() for y in ['tmin', 'tmax']} for x in stazioni}

for s in stazioni:
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    for i, (ax, t) in enumerate(zip(axes, ['tmin', 'tmax'])):
        
        CP = dict_CP[s][t].copy()
        QRF = dict_QRF[s][t].copy()
        OBS = dict_OBS[s][t].copy()
        
        ### Controllo che non ci siano valori anomali
        outlier = f_trova_outlier_mad(CP)
        outlier = f_trova_outlier_mad(QRF)
        outlier = f_trova_outlier_mad(OBS)
        
        # a = pd.concat([CP.iloc[:,2], QRF.iloc[:,2], OBS.iloc[:,2]], axis=1)
        # a.columns = ['CP', 'QRF', 'OBS']
        # a.plot()
        # plt.title(f'{s} {t}', loc='left')
        # plt.ylim(-10, 40)
        # plt.show()
        # plt.close()
        
        # if s == 'CAIRM':
        #     stop
        
        CP.columns = [f'{s}_{t}_{x}_CP' for x in CP.columns]
        QRF.columns = [f'{s}_{t}_{x}_QRF' for x in QRF.columns]
        OBS.columns = [f'{s}_{t}_{x}_OBS' for x in OBS.columns]
        
        df_tot = pd.concat([CP, QRF, OBS], axis=1)
        df_tot = df_tot.dropna()
        
        for d in [0, 1, 2]:
            errori_CP = f_errori_regressione(df_tot[f'{s}_{t}_d{d}_OBS'], df_tot[f'{s}_{t}_d{d}_CP'], nome_df=f'd{d}_CP')
            errori_QRF = f_errori_regressione(df_tot[f'{s}_{t}_d{d}_OBS'], df_tot[f'{s}_{t}_d{d}_QRF'], nome_df=f'd{d}_QRF')
            
            dict_errori[s][t] = pd.concat([dict_errori[s][t], errori_CP, errori_QRF], axis=1)
            
        x = np.arange(3)
        width = 0.13
        
        # Bias
        ax.bar(
            x - 2*width,
            dict_errori[s][t].loc['bias', ['d0_CP', 'd1_CP', 'd2_CP']],
            width,
            color='tab:blue',
            edgecolor='black',
            linewidth=0.5,
            label='Bias CP'
        )
        
        ax.bar(
            x - width,
            dict_errori[s][t].loc['bias', ['d0_QRF', 'd1_QRF', 'd2_QRF']],
            width,
            color='tab:blue',
            hatch='///',
            edgecolor='black',
            linewidth=0.5,
            label='Bias QRF'
        )
        
        # RMSE
        ax.bar(
            x,
            dict_errori[s][t].loc['rmse', ['d0_CP', 'd1_CP', 'd2_CP']],
            width,
            color='tab:orange',
            edgecolor='black',
            linewidth=0.5,
            label='RMSE CP'
        )
        
        ax.bar(
            x + width,
            dict_errori[s][t].loc['rmse', ['d0_QRF', 'd1_QRF', 'd2_QRF']],
            width,
            color='tab:orange',
            hatch='///',
            edgecolor='black',
            linewidth=0.5,
            label='RMSE QRF'
        )
        
        # Pearson
        ax.bar(
            x + 2*width,
            dict_errori[s][t].loc['pearson', ['d0_CP', 'd1_CP', 'd2_CP']],
            width,
            color='tab:red',
            edgecolor='black',
            linewidth=0.5,
            label='Pearson CP'
        )
        
        ax.bar(
            x + 3*width,
            dict_errori[s][t].loc['pearson', ['d0_QRF', 'd1_QRF', 'd2_QRF']],
            width,
            color='tab:red',
            hatch='///',
            edgecolor='black',
            linewidth=0.5,
            label='Pearson QRF'
        )
        
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(['Oggi', 'Domani', 'Dopodomani'])
        
        ax.set_ylim(-1.5, 3.2)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1, 2])
        ax.axhline(0, color='black', linewidth=0.8)
        
        ax.set_axisbelow(True)
        ax.grid(axis='y', linewidth=0.5, ls='-')
        
        ax.set_ylabel('Errore')
        ax.set_title(f"{t.capitalize()}", fontweight='bold')
        ax.axhline(1, color='black', linewidth=0.5, zorder=-10)
        
        if i == 0:
            ax.legend(ncol=2, ncols=3, loc='lower center')
        if i == 0:
            ax.set_title("Analisi: 1 gen - 22 sett 2026", loc='right')
            ax.set_title(f"{df_coordinate.loc[s]['Name']}", loc='left')
        
    plt.tight_layout()
    plt.savefig(f'errori_{s}.png', dpi=300, bbox_inches='tight')
    plt.show()
    # sss
    
# %% Errori per CP
p_cp = pd.read_csv('../../test/errori_vigilanza/p_cp.csv', index_col=0, parse_dates=True)
p_cp['CP'] = p_cp['CP'].fillna(p_cp['P'])

p_cp = p_cp.loc[emissioni, 'CP']

idx_previsore = {previsore: p_cp[p_cp == previsore].index for previsore in p_cp.unique()}

for errore in ['bias', 'rmse', 'pearson']:
    dict_errori_CP = {x: {'d0': [], 'd1': [], 'd2': []} for x in p_cp.unique()}
    
    for s in stazioni:
    
        for previsore, idx in idx_previsore.items():
            
            errori_d = {'d0': [], 'd1': [], 'd2': []}
            
            for t in ['tmin', 'tmax']:
                CP_temp = dict_CP[s][t].loc[idx]
                OBS_temp = dict_OBS[s][t].loc[idx]
                
                a = pd.concat([CP_temp, OBS_temp], axis=1).dropna()
                CP_temp = a.iloc[:, :3]
                OBS_temp = a.iloc[:, 3:]
                
                errori_d['d0'].append(f_errori_regressione(OBS_temp['d0'], CP_temp['d0']).loc[errore].iloc[0])
                errori_d['d1'].append(f_errori_regressione(OBS_temp['d1'], CP_temp['d1']).loc[errore].iloc[0])
                errori_d['d2'].append(f_errori_regressione(OBS_temp['d2'], CP_temp['d2']).loc[errore].iloc[0])
            
            dict_errori_CP[previsore]['d0'].append(np.mean(errori_d['d0']))
            dict_errori_CP[previsore]['d1'].append(np.mean(errori_d['d1']))
            dict_errori_CP[previsore]['d2'].append(np.mean(errori_d['d2']))
    
    df_errori_previsori = pd.DataFrame({
        previsore: {d: np.mean(valori) for d, valori in dd.items()}
        for previsore, dd in dict_errori_CP.items()
    }).T
    
    ### Plot
    x = np.arange(len(df_errori_previsori))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, df_errori_previsori['d0'], width, color='tab:blue', label='Oggi', edgecolor='black', linewidth=0.5)
    ax.bar(x, df_errori_previsori['d1'], width, color='tab:orange', label='Domani', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, df_errori_previsori['d2'], width, color='tab:red', label='Dopodomani', edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    nomi_completi = ['Daniele Carnevale', 'Daniele Luppi', 'Federico Cassola', 'Federico Buscemi']
    nomi_trasformati = [f"{x.split()[0]} {x.split()[1][0]}." if x in nomi_completi else x.split()[0]for x in df_errori_previsori.index]
    # ax.set_xticklabels(df_errori_previsori.index, rotation=45, ha='right')
    ax.set_xticklabels(nomi_trasformati, rotation=45, ha='right')
    
    ax.set_ylabel(f'{errore.capitalize()} medio')
    ax.set_title('Errore medio per previsore')
    ax.legend(loc='upper left')
    ax.set_axisbelow(True)
    ax.grid(axis='y', linewidth=0.5, ls='-')
    if errore == 'rmse':
        ax.set_ylim(0, 2.75)
        ax.set_yticks([0.75, 1, 1.25, 1.5, 2, 2.5])
    elif errore == 'bias':
        ax.set_ylim(-0.3, 0.4)
        ax.legend(loc='lower right')
        ax.set_ylabel(f'{errore.capitalize()} medio\n(bias negativo: sottostima nella previsione)')
    elif errore == 'pearson':
        ax.set_ylim(0.94, 1.01)
        ax.set_yticks([0.94, 0.96, 0.98, 1])
    
    plt.tight_layout()
    plt.savefig(f'errori_per_CP_{errore}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

print('\n\nDone.')
