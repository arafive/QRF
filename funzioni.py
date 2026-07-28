"""
Modulo unico di funzioni/classi condivise dagli script di training della pioggia
(training_elr.py, training_rf_soglie.py, training_xgb_q95.py, training_xgb_wmse.py,
training_xgb_cbrt.py, training_qrf.py, training_qrf_gpd.py, training_gbm_pinball.py,
training_gbm_tweedie_gpd.py, training_csgd.py, training_qm_gpd.py, training_flow.py)
e da altre parti della pipeline (query DB, post-processing vento).

Tutto quello che prima viveva in moduli separati (plot_heatmap_verifica.py,
calibrazione_estremi.py) e' stato accorpato qui per non avere piu' script
di supporto sparsi in giro.
"""

import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import xgboost as xgb
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

from danilib import f_settaggio_db_arpal
connessione = f_settaggio_db_arpal()


# ============================================================================
# Query database (osservazioni stazioni)
# ============================================================================

def f_query(stazione, t0='201901010000'):
    query_1 = f"""
    SELECT
        DTRF,
        CASE WHEN WSPDM IS NOT NULL AND V_WSPDM < 'I' THEN WSPDM/10 END AS WSPDM,
        CASE WHEN WSPDM IS NOT NULL AND V_WSPDM < 'I' THEN WSPDX/10 END AS WSPDX,
        CASE WHEN WSPDM IS NOT NULL AND V_WSPDM < 'I' THEN WDIRP END AS WDIRP,
        CASE WHEN WSPDM IS NOT NULL AND V_WSPDM < 'I' THEN WDIRX END AS WDIRX,
    
        CASE WHEN TEMPM IS NOT NULL AND V_TEMPM < 'I' THEN TEMPN/10 END AS TEMPN,
        CASE WHEN TEMPM IS NOT NULL AND V_TEMPM < 'I' THEN TEMPM/10 END AS TEMPM,
        CASE WHEN TEMPM IS NOT NULL AND V_TEMPM < 'I' THEN TEMPX/10 END AS TEMPX,
    
        CASE WHEN REHUM IS NOT NULL AND V_REHUM < 'I' THEN REHUM END AS REHUM
    
    FROM
        ANAG
    
    JOIN
        DATA ON ANAG.CODE = DATA.CODE
    
    WHERE
        ANAG.CODE IN ('{stazione}')
        AND DTRF >= TO_DATE('{t0}', 'YYYYMMDDHH24MI')
        AND DTRF = TRUNC(DTRF, 'HH')
    
    ORDER BY DTRF
    """

    query_2 = f"""
    SELECT
        DTRF,
        CASE WHEN RAIN01HX IS NOT NULL AND V_RAIN01HX < 'I' THEN RAIN01HX/10 END AS RAIN01HX,
        CASE WHEN RAIN03HX IS NOT NULL AND V_RAIN03HX < 'I' THEN RAIN03HX/10 END AS RAIN03HX
    
    FROM
        ANAG
    
    JOIN
        DATA_1H ON ANAG.CODE = DATA_1H.CODE
    
    WHERE
        ANAG.CODE IN ('{stazione}')
        AND DTRF >= TO_DATE('{t0}', 'YYYYMMDDHH24MI')
    
    ORDER BY DTRF
    """

    df_query_1 = pd.read_sql(query_1, con=connessione)
    df_query_2 = pd.read_sql(query_2, con=connessione)

    df_query_1 = df_query_1.set_index('DTRF')
    df_query_2 = df_query_2.set_index('DTRF')
    df_query_1.index.name = ''
    df_query_2.index.name = ''

    df_query_1 = df_query_1.reindex(pd.date_range(f'{pd.Timestamp(t0)}', df_query_1.index[-1], freq='1h'))
    df_query_2 = df_query_2.reindex(pd.date_range(f'{pd.Timestamp(t0)}', df_query_2.index[-1], freq='1h'))

    return pd.concat([df_query_1, df_query_2], axis=1)


# ============================================================================
# Errori di verifica (regressione / classificazione)
# ============================================================================

def f_errori_regressione(osservato, previsto, nome_df=None):
    """Funzione per gli errori di regressione."""
    df = pd.DataFrame(columns=[nome_df])

    previsto = np.array(previsto).squeeze()
    osservato = np.array(osservato).squeeze()

    try:
        dimensione_dei_dati = len(osservato)
    except TypeError:
        dimensione_dei_dati = 1  # Ha dimensione 1

    errore = previsto - osservato

    try:
        df.loc['bias'] = (1 / dimensione_dei_dati) * np.sum(errore)
    except ZeroDivisionError:
        df.loc['bias'] = np.nan

    try:
        df.loc['mae'] = (1 / dimensione_dei_dati) * np.sum(abs(errore))
    except ZeroDivisionError:
        df.loc['mae'] = np.nan

    df.loc['nbias']   = 100 * (np.sum(errore) / np.sum(osservato))
    df.loc['nmae']    = 100 * (np.sum(abs(errore)) / np.sum(abs(osservato)))
    df.loc['rmse']    = np.sqrt(np.sum(errore ** 2) / dimensione_dei_dati)
    df.loc['nrmse']   = 100 * (np.sqrt(np.sum(errore ** 2) / np.sum(osservato ** 2)))

    if previsto.size == 0 or previsto.size == 1:
        # Se è 0 vuol dire che, ad esempio, non ci sono TP
        df.loc['pearson'] = 0
    else:
        df.loc['pearson'] = np.sum((previsto - np.mean(previsto)) * (osservato - np.mean(osservato))) / np.sqrt(np.sum((previsto - np.mean(previsto)) ** 2) * np.sum((osservato - np.mean(osservato)) ** 2))

    if np.isnan(df.loc['pearson'].iloc[0]):
        # Oppure o l'osservato o la previsione sono costanti.
        # Metto 0 per proseguire.
        df.loc['pearson'] = 0

    return df


def f_errori_classificazione(osservato, previsto, nome_df=None):
    """Funzione per gli errori di classificazione."""
    df = pd.DataFrame(columns=[nome_df])
    previsto = np.array(previsto).squeeze()
    osservato = np.array(osservato).squeeze()
    df_report = pd.DataFrame.from_dict(classification_report(osservato, previsto, zero_division=0, output_dict=True, labels=[0, 1]), orient='columns').rename(columns={'0.0': '0', '1.0': '1'})
    tn, fp, fn, tp = confusion_matrix(osservato, previsto, labels=[0, 1]).ravel()
    df.loc['tn'] = tn
    df.loc['fp'] = fp
    df.loc['fn'] = fn
    df.loc['tp'] = tp
    df.loc['num_positive'] = tp + fn
    df.loc['num_negative'] = fp + tn
    df.loc['num_pred_pos'] = tp + fp
    df.loc['num_pred_neg'] = fn + tn
    df.loc['tss'] = (tp / (fn + tp)) + (tn / (fp + tn)) - 1
    df.loc['hss'] = 2 * (tp * tn - fp * fn) / (df.loc['num_positive'].iloc[0] * df.loc['num_pred_neg'].iloc[0] + df.loc['num_pred_pos'].iloc[0] * df.loc['num_negative'].iloc[0])
    df.loc['csi'] = tp / (tp + fp + fn)
    df.loc['tnr'] = tn / df.loc['num_negative'].iloc[0]
    df.loc['fpr'] = fp / df.loc['num_negative'].iloc[0]
    df.loc['fnr'] = fn / df.loc['num_positive'].iloc[0]
    df.loc['tpr'] = tp / df.loc['num_positive'].iloc[0]
    df.loc['tot_population'] = df.loc['num_positive'].iloc[0] + df.loc['num_negative'].iloc[0]
    df.loc['balanced_accuracy'] = (df.loc['tpr'].iloc[0] + df.loc['tnr'].iloc[0]) / 2 * 100
    df.loc['prevalence[1]'] = df.loc['num_positive'].iloc[0] / df.loc['tot_population'].iloc[0]
    df.loc['precision[1]'] = df_report.loc['precision', '1']
    df.loc['recall[1]'] = df_report.loc['recall', '1']
    df.loc['f1-score[1]'] = df_report.loc['f1-score', '1']
    df.loc['informedness'] = df.loc['tpr'].iloc[0] + df.loc['tnr'].iloc[0] - 1
    df.loc['prevalence_threshold'] = (np.sqrt(df.loc['tpr'].iloc[0] * df.loc['fpr'].iloc[0]) - df.loc['fpr'].iloc[0]) / (df.loc['tpr'].iloc[0] - df.loc['fpr'].iloc[0])

    return df


# ============================================================================
# Pickle
# ============================================================================

def f_apri_pickle(percorso):
    with open(percorso, "rb") as f:
        a = pickle.load(f)
    return a


def f_salva_pickle(oggetto, percorso):
    with open(percorso, 'wb') as f:
        pickle.dump(oggetto, f)


# ============================================================================
# Modelli baseline ad alberi (Random Forest classica / quantilica)
# ============================================================================

def QRF_model(quantili):
    model = RandomForestQuantileRegressor(
        n_estimators=500,
        max_features=0.33,
        min_samples_leaf=3,
        max_depth=10,
        min_samples_split=5,
        max_samples=0.8,
        bootstrap=True,
        random_state=777,
        q=quantili,
        n_jobs=-1
    )

    return model


def RF_model():
    model = RandomForestRegressor(
        n_estimators=500,
        max_features=0.33,
        min_samples_leaf=3,
        max_depth=10,
        min_samples_split=5,
        max_samples=0.8,
        bootstrap=True,
        random_state=777,
        n_jobs=-1
    )

    return model


# ============================================================================
# Trasformazioni angolari (direzione vento)
# ============================================================================

def f_scomposizione_seno_coseno(a):
    return np.column_stack([np.sin(np.radians(a)), np.cos(np.radians(a))])
    # return np.column_stack([-np.sin(np.radians(a)), -np.cos(np.radians(a))])


def f_ricomposizione_seno_cose(a):
    return np.degrees(np.arctan2(a[:, 0], a[:, 1])) % 360


def f_errore_circolare(obs_deg, pred_deg):
    return (pred_deg - obs_deg + 180) % 360 - 180
    # return np.degrees(np.arctan2(-a[:, 0], -a[:, 1])) % 360

# ============================================================================
# Correzione prior per classificatori allenati su dati ribilanciati
# ============================================================================

def f_correggi_probabilita_prior(p_oversampled, prevalenza_train_oversampled, prevalenza_vera):
    """
    Corregge le probabilita' di un classificatore allenato con prior artificiale
    (es. dopo bootstrap oversampling) riportandole alla probabilita' vera rispetto
    alla prevalenza originale. Correzione esatta (Elkan 2001; Saerens, Latinne,
    Decaestecker 2002): assume che il ricampionamento non cambi la distribuzione
    delle feature per classe (vero per costruzione, duplica solo osservazioni
    reali), quindi la trasformazione e' monotona -> l'AUC non cambia, solo il
    valore assoluto della probabilita'.
    """
    p = np.clip(np.asarray(p_oversampled, dtype=float), 1e-9, 1 - 1e-9)
    a = (prevalenza_vera / (1 - prevalenza_vera)) / (prevalenza_train_oversampled / (1 - prevalenza_train_oversampled))
    odds_corretti = (p / (1 - p)) * a
    return odds_corretti / (1 + odds_corretti)


class ClassificatoreConCorrezionePrior:
    """Wrapper che applica automaticamente la correzione di prior alle probabilita'
    di un classificatore allenato su dati ribilanciati via bootstrap oversampling."""
    def __init__(self, classificatore, prevalenza_vera, prevalenza_train_oversampled):
        self.classificatore = classificatore
        self.prevalenza_vera = prevalenza_vera
        self.prevalenza_train_oversampled = prevalenza_train_oversampled

    def predict_proba(self, X):
        p_grezzo = self.classificatore.predict_proba(X)[:, 1]
        p_corretto = f_correggi_probabilita_prior(p_grezzo, self.prevalenza_train_oversampled, self.prevalenza_vera)
        return np.column_stack([1 - p_corretto, p_corretto])

    @property
    def feature_importances_(self):
        return self.classificatore.feature_importances_


def f_plot_precision_recall_vs_soglia(y_bin_test, prob, titolo=None, percorso_salvataggio=None):
    precision, recall, soglie_prob = precision_recall_curve(y_bin_test, prob)
    precision, recall = precision[:-1], recall[:-1]  # l'ultimo punto non ha soglia associata

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(soglie_prob, precision, color='tab:blue', lw=1.8, label='precision')
    ax.plot(soglie_prob, recall, color='tab:orange', lw=1.8, label='recall')
    ax.set_xlabel('soglia di probabilita di allerta')
    ax.set_ylabel('valore')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='center right')
    ax.grid(alpha=0.3)
    if titolo:
        ax.set_title(titolo, fontsize=9)
    fig.tight_layout()
    if percorso_salvataggio:
        fig.savefig(percorso_salvataggio, dpi=150, bbox_inches='tight')
    return fig, ax


# ============================================================================
# ELR - Extended Logistic Regression (Wilks, 2009)
# ============================================================================

class ModelloELR:
    """
    Extended Logistic Regression (Wilks, 2009): una singola regressione logistica
    per tutte le soglie, con la soglia stessa (trasformata con radice quadrata,
    la trasformazione che funziona meglio per la pioggia in letteratura) come
    covariata aggiuntiva. Garantisce probabilita' coerenti (monotone) tra soglie
    diverse, a differenza di classificatori separati per soglia, e permette di
    interrogare anche soglie non usate in fase di fit.
    """
    def __init__(self, trasformazione_soglia=np.sqrt, C=1.0):
        self.trasformazione_soglia = trasformazione_soglia
        self.C = C
        self.modello = None
        self.scaler = None
        self.campi = None

    def fit(self, X, y, soglie_fit):
        self.campi = X.columns.tolist()
        blocchi_X, blocchi_y = [], []
        for soglia in soglie_fit:
            X_blocco = X.copy()
            X_blocco['_soglia_trasf'] = self.trasformazione_soglia(soglia)
            blocchi_X.append(X_blocco)
            blocchi_y.append((y > soglia).astype(int))
        X_pool = pd.concat(blocchi_X, axis=0, ignore_index=True)
        y_pool = pd.concat(blocchi_y, axis=0, ignore_index=True)

        self.scaler = StandardScaler()
        X_pool_scaled = self.scaler.fit_transform(X_pool[self.campi + ['_soglia_trasf']])

        self.modello = LogisticRegression(max_iter=2000, C=self.C)
        self.modello.fit(X_pool_scaled, y_pool)

        coef_soglia = self.modello.coef_[0][-1]
        if coef_soglia >= 0:
            print(f"ATTENZIONE: coefficiente della soglia non negativo ({coef_soglia:.3f}) "
                  "- le probabilita' potrebbero non essere monotone tra soglie diverse.")
        return self

    def predict_proba_soglia(self, X, soglia):
        X_query = X.copy()
        X_query['_soglia_trasf'] = self.trasformazione_soglia(soglia)
        X_query_scaled = self.scaler.transform(X_query[self.campi + ['_soglia_trasf']])
        return self.modello.predict_proba(X_query_scaled)[:, 1]


# ============================================================================
# XGBoost - quantile diretto (es. Q95) via objective='reg:quantileerror'
# ============================================================================

class ModelloQuantile95:
    """
    Regressione quantilica (XGBoost, objective='reg:quantileerror') che prevede
    direttamente il percentile (default 95) della precipitazione condizionato
    ai predittori: il valore previsto e' quello per cui c'e' il 5% di probabilita'
    che la pioggia osservata lo superi. A differenza della ELR non serve scalare
    i dati (modello ad alberi) ne' iterare sulle soglie in fase di fit.
    """
    def __init__(self, quantile_alpha=0.95, n_estimators=400, max_depth=4,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, random_state=42):
        self.modello = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile_alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )
        self.campi = None

    def fit(self, X, y):
        self.campi = X.columns.tolist()
        self.modello.fit(X[self.campi], y)
        return self

    def predict(self, X):
        return self.modello.predict(X[self.campi])


# ============================================================================
# XGBoost - Weighted MSE (pesi lineari crescenti sopra la mediana)
# ============================================================================

def f_pesi_wmse(y, p50, p99, peso_max=10.0):
    """
    Peso 1 fino alla mediana (p50) del training, poi crescita lineare
    che raggiunge peso_max esattamente al percentile 99 (p99). Oltre p99
    la crescita prosegue con la stessa pendenza (nessun cap), per continuare
    a penalizzare, in proporzione, gli errori sugli eventi piu' estremi.
    """
    y = np.asarray(y, dtype=float)
    pesi = np.ones_like(y)
    mask = y > p50
    delta = (p99 - p50) if p99 > p50 else 1e-9
    pesi[mask] = 1.0 + (peso_max - 1.0) * (y[mask] - p50) / delta
    return pesi


class ModelloWMSE:
    """
    Regressione XGBoost con obiettivo custom Weighted MSE: gli errori sui
    valori di pioggia sopra la mediana del training pesano progressivamente
    di piu' (fino a peso_max al percentile 99), spingendo il modello a
    essere piu' accurato sugli eventi intensi a scapito della precisione
    sui valori bassi/nulli, molto piu' numerosi. Non e' un quantile in
    senso stretto: e' una previsione puntuale "spinta" verso la coda alta.
    """
    def __init__(self, peso_max=10.0, n_estimators=400, max_depth=4,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, random_state=42):
        self.peso_max = peso_max
        self.p50 = None
        self.p99 = None
        self.campi = None
        self.modello = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )

    def _obiettivo_wmse(self, y_true, y_pred):
        pesi = f_pesi_wmse(y_true, self.p50, self.p99, self.peso_max)
        grad = 2.0 * pesi * (y_pred - y_true)
        hess = 2.0 * pesi
        return grad, hess

    def fit(self, X, y):
        self.campi = X.columns.tolist()
        self.p50 = np.percentile(y, 50)
        self.p99 = np.percentile(y, 99)
        self.modello.set_params(objective=self._obiettivo_wmse)
        self.modello.fit(X[self.campi], y)
        return self

    def predict(self, X):
        return self.modello.predict(X[self.campi])


# ============================================================================
# XGBoost - target trasformato (radice cubica / log1p) per ridurre l'asimmetria
# ============================================================================

def f_trasforma_target(y, tipo='cbrt'):
    if tipo == 'cbrt':
        return np.cbrt(y)
    elif tipo == 'log1p':
        return np.log1p(y)
    else:
        raise ValueError(f"Trasformazione '{tipo}' non supportata (usare 'cbrt' o 'log1p')")


def f_trasforma_target_inversa(y_trasf, tipo='cbrt'):
    if tipo == 'cbrt':
        return np.power(y_trasf, 3)
    elif tipo == 'log1p':
        return np.expm1(y_trasf)
    else:
        raise ValueError(f"Trasformazione '{tipo}' non supportata (usare 'cbrt' o 'log1p')")


class ModelloTrasformato:
    """
    Regressione XGBoost standard (MSE) addestrata sul target trasformato per
    ridurre l'asimmetria della distribuzione della pioggia (molti zeri, coda
    lunga a destra). Default: radice cubica y^(1/3), preferita in letteratura
    meteorologica rispetto al log perche' gestisce meglio gli zeri senza
    bisogno di offset artificiale (log(y+1)). Le predizioni vengono
    ritrasformate (elevate al cubo) prima di essere restituite da predict(),
    cosi' l'output e' gia' in mm, direttamente confrontabile con le osservazioni.
    """
    def __init__(self, trasformazione='cbrt', n_estimators=400, max_depth=4,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, random_state=42):
        self.trasformazione = trasformazione
        self.campi = None
        self.modello = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.campi = X.columns.tolist()
        y_trasf = f_trasforma_target(y, self.trasformazione)
        self.modello.fit(X[self.campi], y_trasf)
        return self

    def predict(self, X):
        y_trasf_pred = self.modello.predict(X[self.campi])
        return f_trasforma_target_inversa(y_trasf_pred, self.trasformazione)


# ============================================================================
# CalibratoreEstremi: 7 metodi per la calibrazione multi-quantile con gestione
# esplicita della coda estrema (qrf, qrf_gpd, gbm_pinball, gbm_tweedie_gpd,
# csgd, qm_gpd, flow). Interfaccia comune: fit(X, y) / predict(X, quantili).
# ============================================================================

METODI_DISPONIBILI = ['qrf', 'qrf_gpd', 'gbm_pinball',
                      'gbm_tweedie_gpd', 'csgd', 'qm_gpd', 'flow']


# ---------------------------------------------------------------- utility EVT

def _fit_coda_gpd(eccedenze, n_min=20):
    """Fit GPD (scipy genpareto, loc=0) sulle eccedenze positive sopra soglia."""
    eccedenze = np.asarray(eccedenze, dtype=float)
    eccedenze = eccedenze[eccedenze > 0]
    if len(eccedenze) < n_min:
        raise ValueError(
            f"Solo {len(eccedenze)} eccedenze sopra soglia (minimo {n_min}): "
            "abbassa 'soglia_prob' o aumenta il dataset."
        )
    forma, _, scala = stats.genpareto.fit(eccedenze, floc=0)
    return forma, scala


def _quantile_coda_gpd(q, soglia_prob, forma, scala):
    """Quantile GPD (da sommare alla soglia u(x)) per q > soglia_prob."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    p_coda = np.clip((q - soglia_prob) / (1 - soglia_prob), 1e-9, 1 - 1e-9)
    return stats.genpareto.ppf(p_coda, forma, loc=0, scale=scala)


def _ecdf_interpolata(campione):
    """ECDF come funzione interpolabile (per il corpo della qm_gpd)."""
    campione = np.sort(np.asarray(campione, dtype=float))
    n = len(campione)
    p = (np.arange(1, n + 1) - 0.5) / n
    return campione, p


class CalibratoreEstremi:
    """
    Interfaccia comune: fit(X, y) / predict(X, quantili) / salva(percorso) / carica(percorso).
    X: DataFrame di feature. y: Series dell'osservato (es. RAIN03HX).
    predict restituisce un DataFrame (index=X.index, colonne=quantili richiesti).

    NOTA su 'flow': è una versione semplificata (Bernoulli + Box-Cox condizionato),
    non un deep normalizing flow multistrato: quest'ultimo richiederebbe pytorch.
    Se serve la versione con rete neurale vera, va installato torch e va preparata
    la variante apposita.
    """

    def __init__(self, metodo, quantili, soglia_prob=0.95, cap_fisico=None, **kwargs):
        if metodo not in METODI_DISPONIBILI:
            raise ValueError(f"Metodo '{metodo}' non riconosciuto. Scegli tra {METODI_DISPONIBILI}")
        self.metodo = metodo
        self.quantili = sorted(quantili)
        self.soglia_prob = soglia_prob
        self.cap_fisico = cap_fisico  # mm: tetto di plausibilita' fisica, rete di sicurezza
        self.kwargs = kwargs
        self.campi = None
        self._stato = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        self.campi = X.columns.tolist()
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        getattr(self, f'_fit_{self.metodo}')(X, y)
        return self

    def predict(self, X, quantili=None):
        quantili = sorted(quantili) if quantili is not None else self.quantili
        previsioni = getattr(self, f'_predict_{self.metodo}')(X, quantili)
        if self.cap_fisico is not None:
            previsioni = np.clip(previsioni, 0, self.cap_fisico)
        return pd.DataFrame(previsioni, index=X.index, columns=quantili)

    def salva(self, percorso):
        f_salva_pickle(self, percorso)

    @staticmethod
    def carica(percorso):
        return f_apri_pickle(percorso)

    @property
    def feature_importances_(self):
        modello = self._stato.get('modello_corpo')
        if modello is not None and hasattr(modello, 'feature_importances_'):
            return modello.feature_importances_
        raise AttributeError(
            f"feature_importances_ non disponibile per il metodo '{self.metodo}'")

    # ============================================================ 'qrf'
    def _fit_qrf(self, X, y):
        modello = QRF_model(self.quantili)
        modello.fit(X, y)
        self._stato['modello_corpo'] = modello
        self._stato['quantili_fit'] = list(self.quantili)

    def _predict_qrf(self, X, quantili):
        modello = self._stato['modello_corpo']
        quantili_fit = self._stato['quantili_fit']
        mancanti = set(quantili) - set(quantili_fit)
        if mancanti:
            raise ValueError(
                f"Il metodo 'qrf' predice solo i quantili di fit {quantili_fit}; mancano {mancanti}")
        previsioni = np.asarray(modello.predict(X)).T
        idx = [quantili_fit.index(q) for q in quantili]
        return previsioni[:, idx]

    # ============================================================ 'qrf_gpd'
    def _fit_qrf_gpd(self, X, y):
        quantili_fit = sorted(set(self.quantili) | {self.soglia_prob})
        modello = QRF_model(quantili_fit)
        modello.fit(X, y)
        self._stato['modello_corpo'] = modello
        self._stato['quantili_fit'] = quantili_fit

        idx_soglia = quantili_fit.index(self.soglia_prob)
        u_train = np.asarray(modello.predict(X)).T[:, idx_soglia]
        eccedenze = y.values - u_train
        forma, scala = _fit_coda_gpd(eccedenze)
        self._stato['gpd'] = (forma, scala)

    def _predict_qrf_gpd(self, X, quantili):
        modello = self._stato['modello_corpo']
        quantili_fit = self._stato['quantili_fit']
        forma, scala = self._stato['gpd']
        previsioni_corpo = np.asarray(modello.predict(X)).T
        idx_soglia = quantili_fit.index(self.soglia_prob)
        u = previsioni_corpo[:, idx_soglia]

        out = np.empty((len(X), len(quantili)))
        for j, q in enumerate(quantili):
            if q <= self.soglia_prob:
                if q not in quantili_fit:
                    raise ValueError(
                        f"Quantile {q} <= soglia: deve appartenere a self.quantili in fit.")
                out[:, j] = previsioni_corpo[:, quantili_fit.index(q)]
            else:
                out[:, j] = u + \
                    _quantile_coda_gpd(q, self.soglia_prob, forma, scala)
        return out

    # ============================================================ 'gbm_pinball'
    def _peso_intensita(self, y):
        """Peso crescente con l'intensità osservata (default: peso lineare fino a 3x)."""
        peso_max = self.kwargs.get('peso_max', 3.0)
        riferimento = np.quantile(y, 0.99) if np.quantile(
            y, 0.99) > 0 else y.max() + 1e-6
        return 1.0 + (peso_max - 1.0) * np.clip(y / riferimento, 0, 1)

    def _fit_gbm_pinball(self, X, y):
        pesi = self._peso_intensita(y.values)
        modello = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=np.array(self.quantili),
            n_estimators=self.kwargs.get('n_estimators', 400),
            max_depth=self.kwargs.get('max_depth', 5),
            learning_rate=self.kwargs.get('learning_rate', 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
        )
        modello.fit(X, y, sample_weight=pesi)
        self._stato['modello_corpo'] = modello
        self._stato['quantili_fit'] = list(self.quantili)

    def _predict_gbm_pinball(self, X, quantili):
        modello = self._stato['modello_corpo']
        quantili_fit = self._stato['quantili_fit']
        mancanti = set(quantili) - set(quantili_fit)
        if mancanti:
            raise ValueError(
                f"Il metodo 'gbm_pinball' predice solo {quantili_fit}; mancano {mancanti}")
        previsioni = np.asarray(modello.predict(X))  # (n, n_quantili_fit)
        if previsioni.ndim == 1:
            previsioni = previsioni[:, None]
        # i quantili sono stimati da alberi indipendenti e possono incrociarsi:
        # riordino monotono post-hoc (tecnica standard, "quantile crossing fix")
        previsioni = np.sort(previsioni, axis=1)
        idx = [quantili_fit.index(q) for q in quantili]
        return previsioni[:, idx]

    # ============================================================ 'gbm_tweedie_gpd'
    def _fit_gbm_tweedie_gpd(self, X, y):
        modello = xgb.XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=self.kwargs.get(
                'tweedie_variance_power', 1.5),
            n_estimators=self.kwargs.get('n_estimators', 400),
            max_depth=self.kwargs.get('max_depth', 5),
            learning_rate=self.kwargs.get('learning_rate', 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
        )
        modello.fit(X, y)
        pred_train = np.clip(modello.predict(X), 1e-6, None)
        rapporto = y.values / pred_train  # residuo moltiplicativo

        u_train = pred_train * np.quantile(rapporto, self.soglia_prob)
        eccedenze = y.values - u_train
        forma, scala = _fit_coda_gpd(eccedenze)

        self._stato['modello_corpo'] = modello
        self._stato['quantili_rapporto'] = {q: np.quantile(
            rapporto, q) for q in self.quantili if q <= self.soglia_prob}
        self._stato['gpd'] = (forma, scala)

    def _predict_gbm_tweedie_gpd(self, X, quantili):
        modello = self._stato['modello_corpo']
        forma, scala = self._stato['gpd']
        pred = np.clip(np.asarray(modello.predict(X)), 1e-6, None)
        u = pred * np.quantile(list(self._stato['quantili_rapporto'].values()), self.soglia_prob) \
            if self.soglia_prob not in self._stato['quantili_rapporto'] else pred * self._stato['quantili_rapporto'][self.soglia_prob]

        out = np.empty((len(X), len(quantili)))
        for j, q in enumerate(quantili):
            if q <= self.soglia_prob:
                if q not in self._stato['quantili_rapporto']:
                    raise ValueError(
                        f"Quantile {q} <= soglia deve appartenere a self.quantili in fit.")
                out[:, j] = pred * self._stato['quantili_rapporto'][q]
            else:
                out[:, j] = u + \
                    _quantile_coda_gpd(q, self.soglia_prob, forma, scala)
        return out

    # ============================================================ 'csgd'
    @staticmethod
    def _csgd_parametri(theta, Xs):
        n_feat = Xs.shape[1]
        # clip delle feature standardizzate: limita l'estrapolazione della funzione
        # lineare quando il test set contiene combinazioni fuori dal range di training
        # (bound solo sui coefficienti NON basta: coefficiente*feature_enorme esplode comunque)
        Xs = np.clip(Xs, -4, 4)
        w_k, w_theta, w_delta = np.split(theta, 3)
        log_k = w_k[0] + Xs @ w_k[1:]
        log_scale = w_theta[0] + Xs @ w_theta[1:]
        delta = w_delta[0] + Xs @ w_delta[1:]
        # floor su k: shape < ~0.5 rende la gamma fortemente asimmetrica e amplifica
        # di molto il quantile 99esimo rispetto alla media (a parita' di scale) - lo evito
        k = np.exp(np.clip(log_k, np.log(0.5), 4))
        scale = np.exp(np.clip(log_scale, -6, 4))
        return k, scale, delta

    @classmethod
    def _csgd_nll(cls, theta, Xs, y, lambda_L2=0.0):
        k, scale, delta = cls._csgd_parametri(theta, Xs)
        secco = (y <= 1e-6)
        nll = 0.0
        # giorni secchi: -log P(Z <= delta)
        p0 = stats.gamma.cdf(np.clip(delta[secco], 1e-9, None), a=k[secco], scale=scale[secco])
        nll -= np.sum(np.log(np.clip(p0, 1e-12, None)))
        # giorni piovosi: -log pdf_gamma(y+delta)
        z = y[~secco] + delta[~secco]
        z = np.clip(z, 1e-9, None)
        dens = stats.gamma.pdf(z, a=k[~secco], scale=scale[~secco])
        nll -= np.sum(np.log(np.clip(dens, 1e-300, None)))
        penalita = lambda_L2 * np.sum(theta ** 2)  # ridge: tiene i coefficienti contenuti
        return nll / len(y) + penalita

    def _fit_csgd(self, X, y):
        media = X.mean()
        std = X.std().replace(0, 1)
        Xs = ((X - media) / std).values
        self._stato['standardizzazione'] = (media, std)

        n_feat = Xs.shape[1]
        y_pos = y.values[y.values > 1e-6]
        media_y = y_pos.mean() if len(y_pos) else 1.0
        scala_tipica = max(y.values.std(), 1e-3)

        theta0 = np.zeros(3 * (n_feat + 1))
        theta0[0] = np.log(max(media_y, 1e-2))     # log_k iniziale
        theta0[n_feat + 1] = np.log(1.0)            # log_scale iniziale
        theta0[2 * (n_feat + 1)] = 0.0               # delta iniziale

        # bound sui coefficienti (non sulle intercette): con feature standardizzate,
        # un coefficiente oltre questi limiti significa che una singola feature anomala
        # nel test set puo' far esplodere k/scale (via exp) o lo shift delta.
        lim_delta = self.kwargs.get('lim_delta_std', 5) * scala_tipica
        bounds = (
            [(-8, 8)] + [(-3, 3)] * n_feat +      # w_k: intercetta libera, coeff. limitati
            [(-8, 8)] + [(-3, 3)] * n_feat +      # w_theta: idem
            [(-lim_delta, lim_delta)] * (n_feat + 1)  # w_delta: in unita' fisiche (mm)
        )

        risultato = minimize(
            self._csgd_nll, theta0, args=(Xs, y.values, self.kwargs.get('lambda_L2', 0.01)),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': self.kwargs.get('maxiter', 500)}
        )
        self._stato['theta'] = risultato.x
        self._stato['successo_fit'] = risultato.success
        self._stato['modello_corpo'] = None  # nessun feature_importances_ per questo metodo

    def _predict_csgd(self, X, quantili):
        media, std = self._stato['standardizzazione']
        Xs = ((X - media) / std).values
        k, scale, delta = self._csgd_parametri(self._stato['theta'], Xs)

        out = np.empty((len(X), len(quantili)))
        for j, q in enumerate(quantili):
            z_q = stats.gamma.ppf(q, a=k, scale=scale)
            out[:, j] = np.clip(z_q - delta, 0, None)
        return out

    # ============================================================ 'qm_gpd'
    def _fit_qm_gpd(self, X, y):
        modello = xgb.XGBRegressor(
            n_estimators=self.kwargs.get('n_estimators', 400),
            max_depth=self.kwargs.get('max_depth', 5),
            learning_rate=self.kwargs.get('learning_rate', 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
        )
        modello.fit(X, y)
        pred_train = np.clip(np.asarray(modello.predict(X)), 0, None)
        y_train = y.values

        # corpo: ecdf empiriche fino a soglia_prob; coda: GPD oltre soglia_prob
        soglia_pred = np.quantile(pred_train, self.soglia_prob)
        soglia_obs = np.quantile(y_train, self.soglia_prob)
        forma_s, scala_s = _fit_coda_gpd(
            pred_train[pred_train > soglia_pred] - soglia_pred)
        forma_o, scala_o = _fit_coda_gpd(
            y_train[y_train > soglia_obs] - soglia_obs)

        campione_pred, p_pred = _ecdf_interpolata(
            pred_train[pred_train <= soglia_pred])
        campione_obs, p_obs = _ecdf_interpolata(y_train[y_train <= soglia_obs])

        self._stato['modello_corpo'] = modello
        self._stato.update(dict(
            soglia_pred=soglia_pred, soglia_obs=soglia_obs,
            forma_s=forma_s, scala_s=scala_s, forma_o=forma_o, scala_o=scala_o,
            campione_pred=campione_pred, p_pred=p_pred,
            campione_obs=campione_obs, p_obs=p_obs,
        ))

        # mediana calibrata sul training, per costruire i residui (banda di incertezza).
        # I residui vengono conservati come campione (non solo pochi quantili fissi) e
        # dotati a loro volta di una coda GPD, cosi' il metodo puo' rispondere a
        # qualunque quantile richiesto in predict, non solo a quelli usati in fit.
        p = self._mappa_probabilita(pred_train)
        y_calib_mediana = self._mappa_valore(p)
        residui = y_train - y_calib_mediana
        self._stato['residui'] = residui
        soglia_res = np.quantile(residui, self.soglia_prob)
        forma_r, scala_r = _fit_coda_gpd(
            residui[residui > soglia_res] - soglia_res)
        self._stato['soglia_res'] = soglia_res
        self._stato['gpd_res'] = (forma_r, scala_r)

    def _quantile_residuo(self, q):
        residui = self._stato['residui']
        soglia_res = self._stato['soglia_res']
        if q <= self.soglia_prob:
            return np.quantile(residui, q)
        forma_r, scala_r = self._stato['gpd_res']
        return soglia_res + _quantile_coda_gpd(q, self.soglia_prob, forma_r, scala_r)[0]

    def _mappa_probabilita(self, pred):
        st = self._stato
        p = np.where(
            pred <= st['soglia_pred'],
            np.interp(pred, st['campione_pred'], st['p_pred']),
            self.soglia_prob + (1 - self.soglia_prob) * stats.genpareto.cdf(
                np.clip(pred - st['soglia_pred'], 0, None), st['forma_s'], loc=0, scale=st['scala_s']),
        )
        return np.clip(p, 1e-6, 1 - 1e-6)

    def _mappa_valore(self, p):
        st = self._stato
        y = np.where(
            p <= self.soglia_prob,
            np.interp(p, st['p_obs'], st['campione_obs']),
            st['soglia_obs'] +
            _quantile_coda_gpd(p, self.soglia_prob,
                               st['forma_o'], st['scala_o']),
        )
        return y

    def _predict_qm_gpd(self, X, quantili):
        modello = self._stato['modello_corpo']
        pred = np.clip(np.asarray(modello.predict(X)), 0, None)
        p = self._mappa_probabilita(pred)
        y_calib_mediana = self._mappa_valore(p)

        out = np.empty((len(X), len(quantili)))
        for j, q in enumerate(quantili):
            out[:, j] = np.clip(
                y_calib_mediana + self._quantile_residuo(q), 0, None)
        return out

    # ============================================================ 'flow'
    @staticmethod
    def _flow_parametri(theta, Xs):
        n_feat = Xs.shape[1]
        Xs = np.clip(Xs, -4, 4)  # stessa protezione applicata a 'csgd', vedi nota li'
        w_p0, w_mu, w_logsigma, w_lambda = np.split(theta, 4)
        logit_p0 = w_p0[0] + Xs @ w_p0[1:]
        mu = w_mu[0] + Xs @ w_mu[1:]
        log_sigma = w_logsigma[0] + Xs @ w_logsigma[1:]
        lam = w_lambda[0] + Xs @ w_lambda[1:]
        p0 = 1 / (1 + np.exp(-np.clip(logit_p0, -30, 30)))
        sigma = np.exp(np.clip(log_sigma, -10, 10))
        # range ragionevole per Box-Cox su pioggia
        lam = np.clip(lam, -1.0, 1.5)
        return p0, mu, sigma, lam

    @classmethod
    def _flow_nll(cls, theta, Xs, y):
        p0, mu, sigma, lam = cls._flow_parametri(theta, Xs)
        secco = (y <= 1e-6)
        nll = -np.sum(np.log(np.clip(p0[secco], 1e-12, None)))

        y_h = y[~secco]
        lam_h, mu_h, sigma_h, p0_h = lam[~secco], mu[~secco], sigma[~secco], p0[~secco]
        vicino_zero = np.abs(lam_h) < 1e-3
        z = np.where(vicino_zero, np.log(y_h), (y_h ** lam_h -
                     1) / np.where(vicino_zero, 1, lam_h))
        log_jacobiano = (lam_h - 1) * np.log(y_h)  # log|dz/dy|
        log_dens_normale = stats.norm.logpdf(z, loc=mu_h, scale=sigma_h)
        nll -= np.sum(np.log(np.clip(1 - p0_h, 1e-12, None)) +
                      log_dens_normale + log_jacobiano)
        return nll / len(y)

    def _fit_flow(self, X, y):
        media = X.mean()
        std = X.std().replace(0, 1)
        Xs = ((X - media) / std).values
        self._stato['standardizzazione'] = (media, std)

        n_feat = Xs.shape[1]
        theta0 = np.zeros(4 * (n_feat + 1))
        theta0[2 * (n_feat + 1)] = np.log(1.0)   # log_sigma iniziale
        # lambda iniziale (Box-Cox moderato)
        theta0[3 * (n_feat + 1)] = 0.3

        risultato = minimize(
            self._flow_nll, theta0, args=(Xs, y.values),
            method='L-BFGS-B', options={'maxiter': self.kwargs.get('maxiter', 300)}
        )
        self._stato['theta'] = risultato.x
        self._stato['successo_fit'] = risultato.success
        self._stato['modello_corpo'] = None

    def _predict_flow(self, X, quantili):
        media, std = self._stato['standardizzazione']
        Xs = ((X - media) / std).values
        p0, mu, sigma, lam = self._flow_parametri(self._stato['theta'], Xs)

        out = np.empty((len(X), len(quantili)))
        for j, q in enumerate(quantili):
            q_wet = np.clip((q - p0) / (1 - p0), 1e-9, 1 - 1e-9)
            z_q = mu + sigma * stats.norm.ppf(q_wet)
            vicino_zero = np.abs(lam) < 1e-3
            y_q = np.where(vicino_zero, np.exp(z_q), np.clip(
                lam * z_q + 1, 1e-9, None) ** (1 / np.where(vicino_zero, 1, lam)))
            out[:, j] = np.where(q <= p0, 0.0, y_q)
        return out


# ============================================================================
# Heatmap di verifica (tabella colorata per soglie/metodi di classificazione)
# ============================================================================

CMAP_SEMAFORO = LinearSegmentedColormap.from_list('rosso_bianco_verde', ['firebrick', 'white', 'forestgreen'])
CMAP_SEMAFORO.set_bad('lightgrey')

# indici esclusi dalla colorazione perche' non sono metriche di qualita' ma conteggi
# o descrittori del campione (non hanno un "meglio/peggio" indipendente dal contesto)
RIGHE_ESCLUSE_DEFAULT = [
    'tn', 'fp', 'fn', 'tp', 'num_positive', 'num_negative',
    'num_pred_pos', 'num_pred_neg', 'tot_population',
    'prevalence[1]', 'prevalence[0]',
]

# indici bounded in modo naturale: (centro_neutro, minimo, massimo, direzione)
CONFIG_INDICI_BOUNDED = {
    'tss':                  (0,    -1, 1,   'max'),
    'hss':                  (0,    -1, 1,   'max'),
    'informedness':         (0,    -1, 1,   'max'),
    'matthews_corr':        (0,    -1, 1,   'max'),
    'markedness':           (0,    -1, 1,   'max'),
    'csi':                  (0.5,   0, 1,   'max'),
    'tnr':                  (0.5,   0, 1,   'max'),
    'tpr':                  (0.5,   0, 1,   'max'),
    'fpr':                  (0.5,   0, 1,   'min'),
    'fnr':                  (0.5,   0, 1,   'min'),
    'fdr':                  (0.5,   0, 1,   'min'),
    'for':                  (0.5,   0, 1,   'min'),
    'ppv':                  (0.5,   0, 1,   'max'),
    'npv':                  (0.5,   0, 1,   'max'),
    'precision[0]':         (0.5,   0, 1,   'max'),
    'recall[0]':            (0.5,   0, 1,   'max'),
    'f1-score[0]':          (0.5,   0, 1,   'max'),
    'precision[1]':         (0.5,   0, 1,   'max'),
    'recall[1]':            (0.5,   0, 1,   'max'),
    'f1-score[1]':          (0.5,   0, 1,   'max'),
    'fowlkes_mallows':      (0.5,   0, 1,   'max'),
    'accuracy':             (50,    0, 100, 'max'),
    'balanced_accuracy':    (50,    0, 100, 'max'),
    'prevalence_threshold': (0.5,   0, 1,   'min'),
    'precision[1]_risc_50': (0.5, 0, 1, 'max')
}

# indici a rapporto (non bounded, valore neutro/non-informativo = 1): scala logaritmica
CONFIG_INDICI_RAPPORTO = {
    'lr+': 'max',
    'lr-': 'min',
    'dor': 'max',
}


def _bonta_bounded(valori, centro, vmin, vmax, direzione):
    valori = np.asarray(valori, dtype=float)
    x = np.clip(valori, vmin, vmax)
    bonta = np.where(
        x >= centro,
        0.5 + 0.5 * (x - centro) / (vmax - centro) if vmax > centro else 0.5,
        0.5 - 0.5 * (centro - x) / (centro - vmin) if centro > vmin else 0.5,
    )
    if direzione == 'min':
        bonta = 1 - bonta
    return np.clip(bonta, 0, 1)


def _bonta_rapporto(valori, direzione, satura_a=10):
    """Rapporti (lr+, lr-, dor): 1 = non informativo (bianco), scala log oltre/sotto."""
    valori = np.asarray(valori, dtype=float)
    valori_safe = np.where(valori <= 0, 1e-6, valori)
    log_rapporto = np.log10(valori_safe) / np.log10(satura_a)
    log_rapporto = np.clip(log_rapporto, -1, 1)
    bonta = 0.5 + 0.5 * log_rapporto
    if direzione == 'min':
        bonta = 1 - bonta
    return np.clip(bonta, 0, 1)


def _bonta_brier(valori, prevalenze_pos):
    """Brier confrontato con il riferimento climatologico p*(1-p) della stessa colonna."""
    valori = np.asarray(valori, dtype=float)
    riferimento = np.asarray(prevalenze_pos, dtype=float) * (1 - np.asarray(prevalenze_pos, dtype=float))
    riferimento = np.where(riferimento <= 0, 1e-6, riferimento)
    rapporto = valori / riferimento
    bonta = np.where(
        rapporto <= 1,
        0.5 + 0.5 * (1 - rapporto),
        0.5 - 0.5 * np.clip(rapporto - 1, 0, 1),
    )
    return np.clip(bonta, 0, 1)


def f_plot_heatmap_verifica(df_verifica, righe=None, escludi_righe=RIGHE_ESCLUSE_DEFAULT,
                             titolo=None, percorso_salvataggio=None):
    """
    Plot tabellare colorato di un df_verifica (output di f_errori_classificazione
    concatenato su piu' colonne, es. soglie/metodi diversi).
    Verde = valore ottimo, rosso = valore pessimo, bianco = valore neutro/non informativo.
    Il colore e' calcolato per-indice secondo la direzione e i bound naturali della metrica;
    il testo in cella mostra sempre il valore originale, non il punteggio di bonta'.
    """
    if righe is None:
        righe = [r for r in df_verifica.index if r not in escludi_righe]

    prevalenze_pos = df_verifica.loc['prevalence[1]'] if 'prevalence[1]' in df_verifica.index else None

    matrice_bonta = pd.DataFrame(index=righe, columns=df_verifica.columns, dtype=float)
    for indice in righe:
        valori = df_verifica.loc[indice]
        if indice in CONFIG_INDICI_BOUNDED:
            centro, vmin, vmax, direzione = CONFIG_INDICI_BOUNDED[indice]
            matrice_bonta.loc[indice] = _bonta_bounded(valori, centro, vmin, vmax, direzione)
        elif indice in CONFIG_INDICI_RAPPORTO:
            matrice_bonta.loc[indice] = _bonta_rapporto(valori, CONFIG_INDICI_RAPPORTO[indice])
        elif indice == 'brier' and prevalenze_pos is not None:
            matrice_bonta.loc[indice] = _bonta_brier(valori, prevalenze_pos)
        else:
            matrice_bonta.loc[indice] = 0.5

    fig, ax = plt.subplots(figsize=(0.9 * len(df_verifica.columns) + 2, 0.32 * len(righe) + 1))
    ax.imshow(matrice_bonta.values.astype(float), cmap=CMAP_SEMAFORO, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(df_verifica.columns)))
    ax.set_xticklabels(df_verifica.columns, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(righe)))
    ax.set_yticklabels(righe, fontsize=6)

    for i, indice in enumerate(righe):
        for j, colonna in enumerate(df_verifica.columns):
            valore = df_verifica.loc[indice, colonna]
            testo = '-' if pd.isna(valore) else f'{valore:.3g}'
            bonta = matrice_bonta.loc[indice, colonna]
            colore_testo = 'white' if (bonta < 0.2 or bonta > 0.85) else 'black'
            ax.text(j, i, testo, ha='center', va='center', fontsize=5.5, color=colore_testo)

    ax.set_xticks(np.arange(-0.5, len(df_verifica.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(righe), 1), minor=True)
    ax.grid(which='minor', color='lightgrey', linewidth=0.4)
    ax.tick_params(which='minor', length=0)

    if titolo:
        ax.set_title(titolo, fontsize=8)

    fig.tight_layout()
    if percorso_salvataggio:
        fig.savefig(percorso_salvataggio, dpi=200, bbox_inches='tight')
    return fig, ax


def f_eda_calibrazione(df, nome="", soglia_wet=0.2):
    prev, obs = df['prev'], df['obs']
    n = len(df)

    frac_wet_obs = (obs >= soglia_wet).mean()
    frac_wet_prev = (prev >= soglia_wet).mean()
    print(f"--- {nome} (n={n}) ---")
    print(f"Frazione wet - obs: {frac_wet_obs:.3f}  prev: {frac_wet_prev:.3f}")

    obs_wet = obs[obs >= soglia_wet]
    prev_wet = prev[prev >= soglia_wet]
    print(f"Obs wet  - mean: {obs_wet.mean():.2f}  std: {obs_wet.std():.2f}  skew: {stats.skew(obs_wet):.2f}")
    print(f"Prev wet - mean: {prev_wet.mean():.2f}  std: {prev_wet.std():.2f}  skew: {stats.skew(prev_wet):.2f}")

    rho, pval = stats.spearmanr(prev, obs, nan_policy='omit')
    print(f"Spearman rho: {rho:.3f} (p={pval:.1e})")

    for q in [0.95, 0.99, 0.995, 0.999]:
        soglia_q = obs.quantile(q)
        print(f"  q{q}: obs={soglia_q:.1f}mm  prev={prev.quantile(q):.1f}mm  n_oltre_obs={int((obs > soglia_q).sum())}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0,0].hist(np.log1p(obs_wet), bins=30, alpha=0.6, label='obs', density=True)
    axes[0,0].hist(np.log1p(prev_wet), bins=30, alpha=0.6, label='prev', density=True)
    axes[0,0].set_xlabel('log1p(mm)')
    axes[0,0].set_title('Distribuzione (solo wet)')
    axes[0,0].legend()

    axes[0,1].hexbin(prev, obs, gridsize=30, mincnt=1, bins='log')
    lim = max(prev.max(), obs.max())
    axes[0,1].plot([0, lim], [0, lim], 'r--', lw=1)
    axes[0,1].set_xlabel('prev (mm)')
    axes[0,1].set_ylabel('obs (mm)')
    axes[0,1].set_title('Obs vs Prev (hexbin, scala log)')

    bins = [-0.01, soglia_wet, 1, 5, 10, 20, 50, np.inf]
    labels = ['0', '0-1', '1-5', '5-10', '10-20', '20-50', '50+']
    df_tmp = df.copy()
    df_tmp['bin_prev'] = pd.cut(df_tmp['prev'], bins=bins, labels=labels)
    df_tmp.boxplot(column='obs', by='bin_prev', ax=axes[1,0])
    axes[1,0].set_xlabel('bin prev (mm)')
    axes[1,0].set_ylabel('obs (mm)')
    axes[1,0].set_title('Obs condizionato al bin di prev')

    fig.suptitle(f"EDA calibrazione — {nome}")

    stats.probplot(obs_wet, dist=stats.gamma, sparams=stats.gamma.fit(obs_wet, floc=0), plot=axes[1,1])

    stats.probplot(obs_wet, dist=stats.gamma, sparams=stats.gamma.fit(obs_wet, floc=0), plot=axes[1,1])
    axes[1,1].set_title('QQ-plot obs wet vs Gamma')

    plt.tight_layout()
    plt.show()
    plt.close()

    return {'n': n, 'frac_wet_obs': frac_wet_obs, 'frac_wet_prev': frac_wet_prev, 'spearman_rho': rho, 'obs_wet': obs_wet, 'prev_wet': prev_wet}


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


def f_soglia_ottima_hss(y_bin, prob):
    """
    Cerca, tra le soglie di decisione candidate, quella che massimizza l'HSS
    (Heidke Skill Score). Vettorizzato: ordina le probabilita' in modo
    decrescente e calcola TP/FP/FN/TN cumulati al variare della soglia,
    senza ricostruire la matrice di confusione punto per punto.
    """
    y_bin = np.asarray(y_bin)
    prob = np.asarray(prob)
    P = y_bin.sum()
    N = len(y_bin) - P

    ordine = np.argsort(-prob)
    y_ordinato = y_bin[ordine]
    prob_ordinato = prob[ordine]

    tp_cum = np.cumsum(y_ordinato)
    fp_cum = np.cumsum(1 - y_ordinato)
    fn_cum = P - tp_cum
    tn_cum = N - fp_cum

    numeratore = 2 * (tp_cum * tn_cum - fp_cum * fn_cum)
    denominatore = (tp_cum + fn_cum) * (fn_cum + tn_cum) + (tp_cum + fp_cum) * (fp_cum + tn_cum)

    with np.errstate(divide='ignore', invalid='ignore'):
        hss = np.where(denominatore != 0, numeratore / denominatore, 0.0)

    idx_ottimo = np.argmax(hss)
    return prob_ordinato[idx_ottimo], hss[idx_ottimo]


def f_precision_riscalata(precision, prevalenza_originale, prevalenza_target):
    """
    Riscala la precision da un dataset con prevalenza 'prevalenza_originale' a
    quella che si otterrebbe con 'prevalenza_target', a parita' di potere
    discriminante del classificatore (Cavaiola et al. 2024, Nat. Commun., Eq. 8).
    Utile per confrontare la precision tra stazioni/intervalli con prevalenza
    wet diversa, o tra raw e modello se le loro soglie implicano prevalenze
    diverse nel test set.
    """
    p, p_target = prevalenza_originale, prevalenza_target
    fattore = (p - p_target) / (p_target * (1 - p))
    return precision / (1 + fattore * (1 - precision))


def f_dewpoint(t_kelvin, q, p_hpa):
    """
    Temperatura di rugiada (°C) da temperatura (K), umidita' specifica (kg/kg)
    e pressione (hPa), via tensione di vapore + formula di Magnus invertita.
    """
    t_celsius = t_kelvin - 273.15
    e = q * p_hpa / (0.622 + 0.378 * q)  # tensione di vapore, hPa
    e = np.clip(e, 1e-6, None)  # evita log(0) o log(negativo) per q~0
    a, b = 17.625, 243.04

    alpha = np.log(e / 6.1094)
    # La formula ha un asintoto per alpha -> a: q anomali (es. unita' sbagliate,
    # g/kg invece di kg/kg) o valori fisicamente estremi mandano il denominatore
    # verso zero, producendo +inf. Si tiene alpha a distanza di sicurezza.
    alpha = np.clip(alpha, None, a - 1e-3)

    td_celsius = (b * alpha) / (a - alpha)
    return td_celsius


def f_aggiungi_indici_instabilita(df):
    """
    Aggiunge lapse rate 850-500, K-index e Total Totals index, calcolati dalle
    colonne t_XXX/q_XXX gia' presenti (nessuna estrazione GRIB aggiuntiva).

    ATTENZIONE UNITA': assume t_XXX in Kelvin e q_XXX in kg/kg. Dato che hai gia'
    avuto un bug di unita' C/K nel calcolo dell'RH in passato, controlla prima
    con df[['t_850','t_700','t_500']].describe(): valori ~250-300 -> Kelvin (ok
    cosi'), valori ~-30/+40 -> Celsius (togli il "- 273.15" da f_dewpoint e da
    qui sotto).
    """
    df = df.copy()

    df['lapse_rate_850_500'] = df['t_850'] - df['t_500']

    df['td_850'] = f_dewpoint(df['t_850'], df['q_850'], 850)
    df['td_700'] = f_dewpoint(df['t_700'], df['q_700'], 700)

    t_850_c = df['t_850'] - 273.15
    t_700_c = df['t_700'] - 273.15
    t_500_c = df['t_500'] - 273.15

    df['k_index'] = (t_850_c - t_500_c) + df['td_850'] - (t_700_c - df['td_700'])
    df['total_totals'] = (t_850_c + df['td_850']) - 2 * t_500_c

    return df


def f_aggiungi_wind_shear(df, livello_basso='925', livello_alto='500'):
    """
    Aggiunge componenti e modulo dello shear del vento tra due livelli di
    pressione (default 925-500hPa, i due livelli piu' estremi che hai tra
    quelli disponibili).
    """
    df = df.copy()

    du = df[f'u_{livello_alto}'] - df[f'u_{livello_basso}']
    dv = df[f'v_{livello_alto}'] - df[f'v_{livello_basso}']

    df[f'shear_u_{livello_basso}_{livello_alto}'] = du
    df[f'shear_v_{livello_basso}_{livello_alto}'] = dv
    df[f'shear_mod_{livello_basso}_{livello_alto}'] = np.sqrt(du**2 + dv**2)

    return df


import numpy as np
from metpy.calc import (
    virtual_temperature, mixing_ratio_from_specific_humidity,
    equivalent_potential_temperature, dewpoint_from_specific_humidity
)
from metpy.units import units

LIVELLI_HPA = [1000, 950, 925, 900, 850, 800, 700, 600, 500]

def f_aggiungi_tv_thetae(df):
    """
    Temperatura virtuale (Tv) e temperatura potenziale equivalente (theta-e)
    per ciascun livello, piu' il gradiente 850-500hPa di theta-e (stesso
    ingrediente della formula CCSI). Assume t_XXX in Kelvin, q_XXX in kg/kg.

    q e' limitata a un range fisico plausibile prima di qualunque calcolo
    (evita mixing_ratio_from_specific_humidity q/(1-q) esploso per q anomali),
    e td viene limitato dopo il calcolo (stesso principio del limite assoluto
    gia' applicato a f_dewpoint) perche' equivalent_potential_temperature usa
    un esponenziale nella formula di Bolton: anche un td finito ma estremo
    puo' produrre overflow -> inf a valle.
    """
    df = df.copy()
    for p in LIVELLI_HPA:
        t = df[f't_{p}'].values * units.kelvin
        q = np.clip(df[f'q_{p}'].values, 0, 0.05) * units('kg/kg')  # 0.05 kg/kg e' gia' generoso per l'atmosfera reale
        mr = mixing_ratio_from_specific_humidity(q)

        td = dewpoint_from_specific_humidity(p * units.hPa, t, q)
        td_sicuro = np.clip(td.to('degC').magnitude, -90.0, 50.0) * units.degC

        df[f'tv_{p}'] = virtual_temperature(t, mr).magnitude
        df[f'thetae_{p}'] = equivalent_potential_temperature(p * units.hPa, t, td_sicuro).magnitude

    df['thetae_grad_850_500'] = df['thetae_500'] - df['thetae_850']
    return df


from metpy.calc import parcel_profile, cape_cin, relative_humidity_from_specific_humidity

def f_aggiungi_cape_cin(df):
    """
    CAPE/CIN surface-based dal profilo t_XXX/q_XXX. Riga per riga (nessuna
    vettorizzazione nativa in MetPy per profili multipli) - testa prima i
    tempi su un sottoinsieme, es. df.iloc[:200], prima di lanciarlo su tutto.
    """
    livelli_p = np.array(LIVELLI_HPA) * units.hPa
    fallito_lista, cape_lista, cin_lista = [], [], []

    for _, riga in df.iterrows():
        t_prof = np.array([riga[f't_{p}'] for p in LIVELLI_HPA]) * units.kelvin
        q_prof = np.array([riga[f'q_{p}'] for p in LIVELLI_HPA]) * units('kg/kg')
        td_prof = dewpoint_from_specific_humidity(livelli_p, t_prof, q_prof)

        try:
            prof_parcella = parcel_profile(livelli_p, t_prof[0], td_prof[0])
            cape, cin = cape_cin(livelli_p, t_prof, td_prof, prof_parcella)
            cape_lista.append(cape.magnitude)
            cin_lista.append(cin.magnitude)
            fallito_lista.append(0)
        except Exception:
            # LFC/EL non trovati -> interpretazione fisica: nessuna
            # instabilita' convettiva rilevata, non un dato mancante.
            cape_lista.append(0.0)
            cin_lista.append(0.0)
            fallito_lista.append(1)

    df = df.copy()
    df['cape_sb'] = cape_lista
    df['cin_sb'] = cin_lista
    df['cape_cin_flag_fallito'] = fallito_lista
    return df


from metpy.calc import bulk_shear

def f_aggiungi_bulk_shear_profilo(df, profondita_m=6000):
    """
    Bulk shear sull'intero profilo verticale (uso di gh_XXX come altezze reali),
    su uno strato di profondita' 'profondita_m' a partire dal livello piu' basso
    disponibile. La profondita' viene limitata dinamicamente riga per riga
    all'estensione reale del profilo: in aria fredda lo spessore 1000-500hPa
    si restringe e una profondita' fissa (es. 6000m) puo' non starci.
    """
    livelli_p = np.array(LIVELLI_HPA) * units.hPa
    fallito_lista, shear_u_lista, shear_v_lista = [], [], []

    for _, riga in df.iterrows():
        u_prof = np.array([riga[f'u_{p}'] for p in LIVELLI_HPA]) * units('m/s')
        v_prof = np.array([riga[f'v_{p}'] for p in LIVELLI_HPA]) * units('m/s')
        h_prof = np.array([riga[f'gh_{p}'] for p in LIVELLI_HPA]) * units.meter

        estensione_disponibile = (h_prof[-1] - h_prof[0]).magnitude
        profondita_effettiva = min(profondita_m, estensione_disponibile * 0.95)

        try:
            su, sv = bulk_shear(livelli_p, u_prof, v_prof, height=h_prof,
                                 depth=profondita_effettiva * units.meter)
            shear_u_lista.append(su.magnitude)
            shear_v_lista.append(sv.magnitude)
            fallito_lista.append(0)
        except Exception:
            shear_u_lista.append(0.0)
            shear_v_lista.append(0.0)
            fallito_lista.append(1)

    df = df.copy()
    df['shear_u_0_6km'] = shear_u_lista
    df['shear_v_0_6km'] = shear_v_lista
    df['shear_mod_0_6km'] = np.hypot(np.array(shear_u_lista), np.array(shear_v_lista))
    df['shear_flag_fallito'] = fallito_lista
    return df


def f_aggiungi_omega_integrato(df):
    """
    Media della velocita' verticale (omega, Pa/s) sullo strato 850-500hPa,
    piu' il valore minimo (massima ascendenza, assumendo la convenzione
    standard: omega negativo = moto ascendente). Verifica la convenzione di
    segno del tuo modello prima di fidarti dell'interpretazione fisica.
    """
    df = df.copy()
    colonne_w = ['w_850', 'w_800', 'w_700', 'w_500']
    df['omega_mean_850_500'] = df[colonne_w].mean(axis=1)
    df['omega_min_850_500'] = df[colonne_w].min(axis=1)  # ascendenza massima
    return df


def f_aggiungi_flusso_umidita(df):
    """Modulo del flusso di umidita' nei bassi livelli (q_925 * velocita' del vento a 925hPa)."""
    df = df.copy()
    vento_925 = np.hypot(df['u_925'], df['v_925'])
    df['flusso_umidita_925'] = df['q_925'] * vento_925
    return df


def f_aggiungi_zero_termico_relativo(df, elevazione_stazione):
    """
    Quota dello zero termico meno l'elevazione della stazione: valori negativi
    indicano che la stazione e' sopra lo zero termico (precipitazione
    potenzialmente nevosa, non piovosa).
    """
    df = df.copy()
    df['zeroT_relativo'] = df['zeroT'] - elevazione_stazione
    return df


def f_aggiungi_depressione_rugiada(df):
    """Differenza t2m - td2m: piu' vicina a zero, piu' l'aria al suolo e' vicina alla saturazione."""
    df = df.copy()
    df['depressione_rugiada_2m'] = df['2t'] - df['2d']
    return df


def f_aggiungi_frazione_convettiva(df):
    """
    Frazione della pioggia totale attribuita a convezione (cp/tp). Quando
    tp=0 la frazione e' concettualmente indefinita, ma qui si usa 0 come
    default (nessuna pioggia -> nessuna frazione convettiva da segnalare),
    con un flag per distinguere questo caso da una frazione convettiva
    davvero nulla in presenza di pioggia.
    """
    df = df.copy()
    tp_zero = df['tp'] <= 0
    df['frazione_convettiva'] = np.where(tp_zero, 0.0, df['cp'] / df['tp'].replace(0, np.nan))
    df['frazione_convettiva_flag_tp_zero'] = tp_zero.astype(int)
    return df


def f_aggiungi_rapporto_raffica(df):
    """
    Raffica massima 3h / velocita' media del vento a 10m. Con vento quasi
    calmo il rapporto esploderebbe (o darebbe NaN a vento esattamente zero)
    pur essendo un caso fisicamente informativo (raffica con flusso medio
    debole = turbolenza locale) - si applica un pavimento minimo alla
    velocita' invece di azzerare o scartare, cosi' il rapporto resta alto
    ma finito invece di NaN.
    """
    df = df.copy()
    vento_10m = np.hypot(df['10u'], df['10v'])
    vento_10m_pavimento = vento_10m.clip(lower=0.5)  # m/s
    df['rapporto_raffica'] = df['10gust3max'] / vento_10m_pavimento
    return df

LIVELLI_HPA_TUTTI = [1000, 950, 925, 900, 850, 800, 700, 600, 500]
LIVELLI_HPA_OMEGA = [925, 900, 850, 800, 700, 500]  # unici livelli con 'w_XXX' disponibile

def f_aggiungi_rh_quota(df):
    """
    Umidita' relativa (RH, %) ai livelli in quota indicati, dalle colonne
    t_XXX (temperatura, K) e q_XXX (umidita' specifica, kg/kg) gia' presenti
    (nessuna estrazione aggiuntiva). Analoga a f_aggiungi_indici_instabilita.
    """
    df = df.copy()
    for p in LIVELLI_HPA_TUTTI:
        t = df[f't_{p}'].values * units.kelvin
        q = df[f'q_{p}'].values * units('g/kg')
        pressione = p * units.hPa
        rh = relative_humidity_from_specific_humidity(pressione, t, q).to('percent').magnitude
        df[f'r_{p}'] = np.clip(rh, 0.0, 100.0)  # limite fisico
    return df


# ============================================================================
# GPD via gradient boosting (ispirato a gbex - Velthoen, Dombry, Cai, Engelke,
# "Gradient boosting for extreme quantile regression", Extremes 2023 - il
# pacchetto originale e' R, github.com/JVelthoen/gbex). Reimplementazione
# Python con obiettivo custom XGBoost. SEMPLIFICAZIONE DELIBERATA: la forma
# (gamma) e' stimata una volta sola via MLE sul pool degli eccessi di
# training (non dipende dalle covariate); solo la scala (sigma) e' "boostata"
# in funzione delle covariate. Usato da training_gpd.py e training_cla_gpd.py.
# ============================================================================

def f_gpd_quantile(sigma, gamma, p):
    """Quantile p della GPD (eccesso oltre soglia), dati sigma (array) e gamma (scalare)."""
    if abs(gamma) < 1e-6:
        return -sigma * np.log(1 - p)
    return (sigma / gamma) * ((1 - p) ** (-gamma) - 1)


def f_gpd_media(sigma, gamma):
    """Media della GPD (eccesso oltre soglia): definita solo per gamma < 1, altrimenti NaN."""
    return np.where(gamma < 1, sigma / (1 - gamma), np.nan)


def f_obiettivo_gpd_sigma(gamma_fisso):
    """
    Obiettivo custom XGBoost: gradiente e hessiana della negative log-
    likelihood GPD rispetto a eta=log(sigma) (link log, garantisce sigma>0),
    a gamma fissato. Derivazione analitica valida in modo uniforme anche nel
    limite gamma->0 (caso esponenziale): nessun ramo speciale necessario.
    """
    def obiettivo(predt, dtrain):
        z = dtrain.get_label()
        sigma = np.exp(predt)
        w = z / sigma
        denom = np.clip(1 + gamma_fisso * w, 1e-6, None)
        grad = 1 - (1 + gamma_fisso) * w / denom
        hess = (1 + gamma_fisso) * w / denom**2
        hess = np.clip(hess, 1e-6, None)  # xgboost richiede hessiana positiva
        return grad, hess
    return obiettivo


class GPDBoosting:
    """
    Modello GPD con scala dipendente dalle covariate via gradient boosting
    (XGBoost, obiettivo custom) e forma fissa (MLE sul pool degli eccessi di
    training). Interfaccia fit/predict, sullo stile del resto della pipeline.
    """

    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma_ = None
        self.booster_ = None
        self.campi_ = None
        self.eta_iniziale_ = None

    def fit(self, X, z):
        """X: covariate (solo righe di eccedenza). z: eccessi (y - soglia), tutti > 0."""
        self.campi_ = X.columns.tolist()

        gamma_stimato, _, _ = stats.genpareto.fit(z, floc=0)
        # Clip di sicurezza: gamma molto negativo implicherebbe un limite
        # superiore fisico vicinissimo alla soglia (poco plausibile per la
        # pioggia), gamma molto positivo rende la coda quasi non-integrabile.
        self.gamma_ = float(np.clip(gamma_stimato, -0.5, 1.5))

        self.eta_iniziale_ = float(np.log(max(z.mean(), 1e-3)))

        dtrain = xgb.DMatrix(X[self.campi_], label=z)
        dtrain.set_base_margin(np.full(len(z), self.eta_iniziale_))

        params = {
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            'disable_default_eval_metric': 1,
            'verbosity': 0,
        }
        self.booster_ = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
            obj=f_obiettivo_gpd_sigma(self.gamma_)
        )
        return self

    def predict_sigma(self, X):
        d = xgb.DMatrix(X[self.campi_])
        d.set_base_margin(np.full(X.shape[0], self.eta_iniziale_))
        eta = self.booster_.predict(d, output_margin=True)
        return np.exp(eta)

    def predict_quantili(self, X, quantili):
        """Ritorna un DataFrame con i quantili richiesti (in mm di ECCESSO, non ancora + soglia) e la media."""
        sigma = self.predict_sigma(X)
        df_out = pd.DataFrame(index=X.index)
        for p in quantili:
            df_out[p] = f_gpd_quantile(sigma, self.gamma_, p)
        df_out['media'] = f_gpd_media(sigma, self.gamma_)
        return df_out

    def feature_importances(self, importance_type='gain'):
        """
        Nota: a differenza di feature_importances_ di sklearn (normalizzata,
        somma a 1), get_score(importance_type='gain') di XGBoost restituisce
        il gain medio grezzo per split - scala arbitraria, non normalizzata.
        """
        punteggi = self.booster_.get_score(importance_type=importance_type)
        return pd.Series({c: punteggi.get(c, 0.0) for c in self.campi_})
