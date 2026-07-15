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


def f_ricomposizione_seno_cose(a):
    return np.degrees(np.arctan2(a[:, 0], a[:, 1])) % 360


def f_errore_circolare(obs_deg, pred_deg):
    return (pred_deg - obs_deg + 180) % 360 - 180


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
