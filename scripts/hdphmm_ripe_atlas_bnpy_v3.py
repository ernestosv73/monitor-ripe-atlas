# -*- coding: utf-8 -*-
"""
=============================================================================
HDP-HMM con bnpy para caracterizacion de latencia end-to-end en RIPE Atlas (v3)
+ correlacion con Path Analysis, analisis de residuos e informe interpretable
=============================================================================
Basado en: Mouchet, Vaton, Chonavel, Aben y den Hertog, "Large-Scale
           Characterization and Segmentation of Internet Path Delays with
           Infinite HMMs" (IEEE Access, 2020; arXiv:1910.12714).
           (bnpy es de Hughes & Sudderth; NO es el paper.)

Entorno: Python 2.7 (o 3) + bnpy + numpy. Solo numpy + biblioteca estandar.

Cambios respecto a v1 (hdphmm_ripe_atlas_bnpy.py)
-------------------------------------------------
 1. Decodificacion: --decodificacion viterbi (default) usa la secuencia de
    estados MAS PROBABLE CONJUNTA, como el paper ("the most likely hidden
    state sequence"); argmax de resp (v1) elige el estado mas probable
    ciclo por ciclo y genera mas chattering. Si no se pueden extraer los
    parametros del HMM de tu build, cae a argmax y lo avisa. El Viterbi es
    propio (numpy) y usa E[log p(x|z)] de bnpy + probabilidades de
    transicion puntuales: es una aproximacion razonable del VB, no el
    Viterbi exacto del Gibbs del paper.
 2. Una sola vista de estados: las tablas se calculan DESPUES del
    suavizado y se muestran tambien los crudos; consola y
    resumen_ejecucion.txt son identicos.
 3. Regimenes renumerados por RTT medio (R1 = el mas rapido), con media,
    desvio, mediana, apariciones, permanencia y arquetipo (estable /
    variable / atipico, como los arquetipos del paper). Aviso de pares de
    estados fusionables (d de Cohen baja).
 4. Cada change-point trae el salto de MEDIA del regimen (ventana antes /
    despues dentro de cada tramo) y de desvio, y un tipo: nivel,
    varianza, nivel+varianza, equivalente (estados casi identicos) o
    atipico. v1 usaba la diferencia entre dos ciclos consecutivos, que no
    mide el cambio de regimen.
 5. Fase 6 con linea base de azar (analitica + permutaciones), cobertura en
    ambos sentidos (CP->evento y evento->CP), emparejamiento 1 a 1 y
    desglose por categoria.
 6. Por defecto se excluyen del ground truth las categorias de cambio de
    ruta (--incluir-cambios-ruta las conserva): el enfoque es RTT end-to-end.
 7. Control del salto final: si en algunos ciclos el ultimo salto que
    respondio no es el habitual (destino sin respuesta), ese RTT no es
    end-to-end. Se avisa; --usar-hop-modal los excluye.
 8. --seed ahora se pasa como algseed/dataorderseed (bnpy ignoraba 'seed').
    Sin --seed se usan las semillas por defecto de bnpy.
 9. --sF por defecto = varianza de la serie (v1: 1.0).
10. Nuevas salidas: estados_resumen, transiciones, linea de tiempo HTML.

Cambios de v3 respecto a v2
---------------------------
 A. Estadisticos ROBUSTOS: desvio robusto = 1.4826*MAD y medianas para el
    arquetipo, la d de Cohen, el tipo de cambio y el salto de ventana (la
    media y el desvio clasicos se inflaban con un solo atipico).
 B. Fusion automatica de estados equivalentes (d < 0.5 y cociente de
    desvios < 1.5, ninguno atipico) ANTES del suavizado; aglomerativa, de
    a un par por vez, recalculando. --no-fusionar la desactiva.
 C. Barrido de --min-dwell (1,2,3,5,8 y el elegido): estados, cambios,
    change-points y coincidencia con las reglas para cada valor.
 D. Patron horario: entradas al regimen variable desde uno estable por
    hora del dia, con test de chi-cuadrado por Monte Carlo.
 E. Episodios en que el ultimo salto con respuesta no es el modal: regimen
    antes/durante/despues, RTT antes/durante/despues y cambios de ruta
    cercanos (test contra ventanas al azar). Prueba si los regimenes de
    RTT siguen a los cambios de ruta.

USO:
    python hdphmm_ripe_atlas_bnpy_v3.py \\
        --crudo historial_traceroute_measurement59176905_probe23108.csv \\
        --categorias eventos_measurement59176905_probe23108_sin_perdida.csv \\
        --output-dir resultados_hdphmm_v3 \\
        --gap-minutes 90 --K 15 --nlap 100 --sF 10
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import argparse
import csv
import re
import io
import math
import bisect
from collections import defaultdict, Counter
from datetime import datetime, timedelta

try:
    import numpy as np
except ImportError:
    sys.stderr.write("Este script necesita numpy instalado.\n")
    sys.exit(1)


# ============================================================================
# CONSTANTES
# ============================================================================
CATEGORIAS_RUTA = ("cambio_ip_asn", "cambio_longitud", "cambio_destino")
UMBRAL_PA_MS = 10.0     # umbral absoluto de degradacion de RTT de Path Analysis
UMBRAL_PA_PCT = 20.0    # umbral relativo
D_FUSIONABLE = 0.5      # d de Cohen por debajo de la cual dos estados son casi iguales
RATIO_STD_VARIANZA = 1.5
STD_ROB_MIN = 0.01      # piso del desvio robusto (ms) para evitar cocientes degenerados
PALETA = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2",
          "#edc948", "#9c755f"]
COLOR_CAT = {"rtt_dark": "#7a1fa2", "rtt_medium": "#ab47bc", "rtt_light": "#ce93d8",
             "cambio_ip_asn": "#d32f2f", "cambio_longitud": "#fbc02d",
             "cambio_destino": "#ef6c00", "perdida_intermedia": "#1976d2"}


# ============================================================================
# UTILIDADES
# ============================================================================
def extraer_ids_de_archivo(filename):
    """measurement_id y probe_id del nombre del archivo (o (None, None))."""
    m = re.search(r"measurement(\d+)_probe(\d+)", os.path.basename(filename))
    if m:
        return m.group(1), m.group(2)
    return None, None


def parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def abrir_csv_w(path):
    if sys.version_info[0] >= 3:
        return open(path, "w", newline="")
    return open(path, "wb")


def escribir_texto(path, texto):
    """Escribe texto UTF-8 tanto en py2 (str = bytes) como en py3."""
    if isinstance(texto, bytes):
        with open(path, "wb") as f:
            f.write(texto)
    else:
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(texto)


def f2(x):
    return "{0:.2f}".format(x)


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ============================================================================
# FASE 1: CARGA DE DATOS
# ============================================================================
def leer_ciclos(input_path):
    """CSV crudo -> {timestamp: {hop: rtt}} y n de lineas."""
    por_ciclo = defaultdict(dict)
    n_lineas = 0
    with open(input_path, "r") as f:
        for row in csv.DictReader(f):
            n_lineas += 1
            ts = row["timestamp"].strip()
            try:
                hop = int(row["hop"])
                rtt = float(row["rtt_ms"])
            except (ValueError, TypeError, KeyError):
                continue
            if rtt != rtt:  # NaN
                continue
            por_ciclo[ts][hop] = rtt
    return por_ciclo, n_lineas


def analizar_salto_final(por_ciclo):
    """Distribucion del ultimo salto que respondio por ciclo y salto modal."""
    cont = Counter(max(h) for h in por_ciclo.values() if h)
    modal = cont.most_common(1)[0][0] if cont else None
    return cont, modal


def construir_serie(por_ciclo, dest_hop=None):
    """Lista ordenada de (timestamp, rtt) del salto de destino de cada ciclo."""
    serie = []
    for ts in sorted(por_ciclo.keys()):
        hops = por_ciclo[ts]
        if not hops:
            continue
        if dest_hop is not None:
            if dest_hop in hops:
                serie.append((ts, hops[dest_hop]))
        else:
            serie.append((ts, hops[max(hops.keys())]))
    return serie


def cargar_categorias(path_csv, excluir=()):
    """
    CSV de ground truth -> {timestamp: {'categorias': set, 'conteo': n}},
    sin las categorias en 'excluir'. Devuelve tambien un resumen.
    """
    d = defaultdict(lambda: {"categorias": set(), "conteo": 0})
    n_lineas = 0
    excluidas = Counter()
    ts_excluidos = defaultdict(set)
    with open(path_csv, "r") as f:
        for row in csv.DictReader(f):
            n_lineas += 1
            ts = row["timestamp"].strip()
            cat = row["categoria"].strip()
            if cat in excluir:
                excluidas[cat] += 1
                ts_excluidos[ts].add(cat)
                continue
            d[ts]["categorias"].add(cat)
            d[ts]["conteo"] += 1
    por_cat = Counter()
    for info in d.values():
        for c in info["categorias"]:
            por_cat[c] += 1
    return dict(d), {"n_lineas": n_lineas, "excluidas": excluidas, "por_cat": por_cat,
                     "ts_excluidos": dict(ts_excluidos)}


# ============================================================================
# FASE 2: SEGMENTACION TEMPORAL
# ============================================================================
def cortar_en_secuencias(serie, gap_minutes):
    if not serie:
        return []
    secuencias, actual = [], [serie[0]]
    for i in range(1, len(serie)):
        delta = (parse_ts(serie[i][0]) - parse_ts(serie[i - 1][0])).total_seconds() / 60.0
        if delta > gap_minutes:
            secuencias.append(actual)
            actual = [serie[i]]
        else:
            actual.append(serie[i])
    secuencias.append(actual)
    return secuencias


def cadencia_mediana_min(secuencias):
    deltas = []
    for s in secuencias:
        for i in range(1, len(s)):
            deltas.append((parse_ts(s[i][0]) - parse_ts(s[i - 1][0])).total_seconds() / 60.0)
    return float(np.median(deltas)) if deltas else 0.0


# ============================================================================
# FASE 3: HDP-HMM CON BNPY + DECODIFICACION
# ============================================================================
def viterbi(log_ev, log_pi0, log_pi):
    """Secuencia de estados mas probable (log-espacio). log_ev: (T,K)."""
    T, K = log_ev.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int64)
    delta[0] = log_pi0 + log_ev[0]
    for t in range(1, T):
        tmp = delta[t - 1][:, None] + log_pi
        psi[t] = tmp.argmax(axis=0)
        delta[t] = tmp.max(axis=0) + log_ev[t]
    z = np.empty(T, dtype=np.int64)
    z[-1] = int(delta[-1].argmax())
    for t in range(T - 2, -1, -1):
        z[t] = psi[t + 1][z[t + 1]]
    return z


def _log(x):
    return np.log(np.maximum(x, 1e-300))


def extraer_parametros_hmm(hmodel, Data, LP):
    """
    Devuelve (log_soft_ev (N,K), pi0 (K,), P (K,K)) o None si el build de
    bnpy no los expone con los nombres esperados.
    """
    try:
        logev = LP.get("E_log_soft_ev") if hasattr(LP, "get") else None
        if logev is None:
            logev = hmodel.obsModel.calcLogSoftEvMatrix_FromPost(Data)
        logev = np.asarray(logev, dtype=float)

        alloc = hmodel.allocModel
        P = pi0 = None
        try:
            P = np.asarray(alloc.get_trans_prob_matrix(), dtype=float)
        except Exception:
            P = None
        try:
            pi0 = np.asarray(alloc.get_init_prob_vector(), dtype=float)
        except Exception:
            pi0 = None
        if P is None and hasattr(alloc, "transTheta"):
            th = np.asarray(alloc.transTheta, dtype=float)
            P = th[:, :th.shape[0]]
        if P is None:
            return None
        K = P.shape[0]
        if P.shape != (K, K) or logev.shape[1] != K:
            return None
        if pi0 is None and hasattr(alloc, "startTheta"):
            pi0 = np.asarray(alloc.startTheta, dtype=float)[:K]
        if pi0 is None or pi0.shape[0] != K:
            pi0 = np.ones(K) / K
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
        pi0 = pi0 / max(pi0.sum(), 1e-300)
        return logev, pi0, P
    except Exception:
        return None


def correr_hdphmm(secuencias, K, nlap, transAlpha, startAlpha, hmmKappa, sF,
                  seed, bnpy_outdir, alg_name, decodificacion):
    """
    Entrena el HDP-HMM. Devuelve un dict con:
      'argmax'   : estados por secuencia (argmax de resp, como v1)
      'viterbi'  : estados por secuencia (Viterbi) o None
      'usados'   : cual de los dos se usara ('viterbi' o 'argmax')
      'P'        : matriz de transicion puntual del modelo (K,K) o None
    o None si falla.
    """
    print("[FASE 3] Entrenando HDP-HMM con bnpy...")
    print("  - Parametros: K={0}, nlap={1}, transAlpha={2}, startAlpha={3}, hmmKappa={4}, sF={5}".format(
        K, nlap, transAlpha, startAlpha, hmmKappa, sF))

    if not os.environ.get("BNPYOUTDIR"):
        os.environ["BNPYOUTDIR"] = bnpy_outdir
    if not os.path.isdir(os.environ["BNPYOUTDIR"]):
        os.makedirs(os.environ["BNPYOUTDIR"])

    try:
        import bnpy
    except ImportError:
        print("  [ERROR] bnpy no esta disponible en este interprete.")
        return None

    X = np.concatenate(
        [np.array([rtt for (_, rtt) in seq], dtype=np.float64).reshape(-1, 1) for seq in secuencias],
        axis=0)
    doc_range = [0]
    for seq in secuencias:
        doc_range.append(doc_range[-1] + len(seq))
    doc_range = np.array(doc_range, dtype=np.int32)

    Data = bnpy.data.GroupXData(X=X, doc_range=doc_range)
    Data.name = "ripe_atlas_rtt"

    extra = {}
    if seed is not None:
        extra["algseed"] = seed
        extra["dataorderseed"] = seed

    try:
        hmodel, RInfo = bnpy.run(
            Data, "HDPHMM", "Gauss", alg_name,
            jobname="hdphmm-segment",
            nLap=nlap, nTask=1, nBatch=1,
            K=K, initname="randexamples",
            transAlpha=transAlpha, startAlpha=startAlpha, hmmKappa=hmmKappa,
            sF=sF, ECovMat="eye",
            printEvery=25, saveEvery=-1, traceEvery=-1,
            doWriteStdOut=False,
            **extra)
    except Exception as e:
        print("  [ERROR] Entrenando HDP-HMM: {0}".format(e))
        return None

    LP = hmodel.calc_local_params(Data)
    est_argmax = LP["resp"].argmax(axis=1)

    est_vit = None
    P = None
    params = extraer_parametros_hmm(hmodel, Data, LP)
    if params is not None:
        logev, pi0, P = params
        try:
            z = np.empty(logev.shape[0], dtype=np.int64)
            for i in range(len(secuencias)):
                a, b = doc_range[i], doc_range[i + 1]
                z[a:b] = viterbi(logev[a:b], _log(pi0), _log(P))
            est_vit = z
        except Exception as e:
            print("  [AVISO] Viterbi fallo ({0}); se usa argmax.".format(e))
            est_vit = None
    else:
        print("  [AVISO] No se pudieron extraer los parametros del HMM de este build; "
              "se usa argmax (como v1).")

    usados = "viterbi" if (decodificacion == "viterbi" and est_vit is not None) else "argmax"
    if decodificacion == "viterbi" and est_vit is None:
        print("  [AVISO] Se pidio Viterbi pero no esta disponible: decodificacion = argmax.")

    def por_secuencia(est):
        return [np.asarray(est[doc_range[i]:doc_range[i + 1]]).astype(int)
                for i in range(len(secuencias))]

    print("  - Entrenamiento terminado (decodificacion: {0}).".format(usados))
    return {"argmax": por_secuencia(est_argmax),
            "viterbi": por_secuencia(est_vit) if est_vit is not None else None,
            "usados": usados, "P": P}


# ============================================================================
# FASE 4: SUAVIZADO POST-HOC Y ESTADISTICAS
# ============================================================================
def suavizar_estados(estados, min_dwell):
    """Fusiona corridas mas cortas que min_dwell con la vecina mas larga."""
    estados = [int(e) for e in estados]
    n = len(estados)
    if n == 0 or min_dwell <= 1:
        return estados
    for _ in range(20):
        runs, i = [], 0
        while i < n:
            j = i
            while j < n and estados[j] == estados[i]:
                j += 1
            runs.append([i, j, estados[i]])
            i = j
        if len(runs) <= 1:
            break
        hubo = False
        for idx in range(len(runs)):
            ini, fin, val = runs[idx]
            if fin - ini >= min_dwell:
                continue
            izq = runs[idx - 1] if idx > 0 else None
            der = runs[idx + 1] if idx < len(runs) - 1 else None
            if izq is not None and der is not None:
                elegido = izq if (izq[1] - izq[0]) >= (der[1] - der[0]) else der
            elif izq is not None:
                elegido = izq
            elif der is not None:
                elegido = der
            else:
                continue
            for k in range(ini, fin):
                estados[k] = elegido[2]
            hubo = True
        if not hubo:
            break
    return estados


def runs_de(est):
    """[(ini, fin, valor)] de corridas consecutivas."""
    est = [int(e) for e in est]
    runs, i, n = [], 0, len(est)
    while i < n:
        j = i
        while j < n and est[j] == est[i]:
            j += 1
        runs.append((i, j, est[i]))
        i = j
    return runs


def contar_cambios(estados_por_secuencia):
    return sum(max(len(runs_de(a)) - 1, 0) for a in estados_por_secuencia)


def estadisticas_por_estado(secuencias, estados_por_secuencia, cadencia_min):
    rtts, largos, total = defaultdict(list), defaultdict(list), 0
    for seq, est in zip(secuencias, estados_por_secuencia):
        r = np.array([x[1] for x in seq])
        total += len(r)
        for ini, fin, val in runs_de(est):
            rtts[val].extend(r[ini:fin].tolist())
            largos[val].append(fin - ini)
    out = {}
    for e, v in rtts.items():
        a = np.array(v)
        med = float(np.median(largos[e]))
        mediana = float(np.median(a))
        mad = float(np.median(np.abs(a - mediana)))
        out[e] = {"estado": e, "n": len(a), "pct": 100.0 * len(a) / max(total, 1),
                  "media": float(a.mean()), "std": float(a.std()),
                  "mediana": mediana, "std_rob": max(1.4826 * mad, STD_ROB_MIN),
                  "apariciones": len(largos[e]),
                  "perm_med_ciclos": med, "perm_med_min": med * cadencia_min}
    return out


def ranking_regimenes(stats_ref):
    """{estado_id: rank 1..n} ordenando por RTT medio ascendente."""
    orden = sorted(stats_ref.keys(), key=lambda e: stats_ref[e]["media"])
    return dict((e, i + 1) for i, e in enumerate(orden))


def arquetipo(st, cv_estable, pct_minimo=1.0):
    """estable / variable / atipico con desvio robusto sobre mediana."""
    if st["pct"] < pct_minimo:
        return "atipico"
    cv = st["std_rob"] / max(st["mediana"], 1e-9)
    return "estable" if cv < cv_estable else "variable"


def d_cohen(a, b):
    """Distancia estandarizada ROBUSTA: |dif. de medianas| / desvio robusto agrupado."""
    pooled = math.sqrt((a["std_rob"] ** 2 + b["std_rob"] ** 2) / 2.0)
    dm = abs(a["mediana"] - b["mediana"])
    if pooled < 1e-9:
        return float("inf") if dm > 1e-9 else 0.0
    return dm / pooled


def ratio_std(a, b):
    lo, hi = sorted([a["std_rob"], b["std_rob"]])
    return hi / max(lo, 1e-9)


def pares_equivalentes(stats, cv_estable, pct_minimo=1.0):
    """Pares de estados no atipicos con d < D_FUSIONABLE y cociente de desvios < RATIO_STD_VARIANZA."""
    ids = sorted(e for e in stats.keys() if stats[e]["pct"] >= pct_minimo)
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = stats[ids[i]], stats[ids[j]]
            d, r = d_cohen(a, b), ratio_std(a, b)
            if d < D_FUSIONABLE and r < RATIO_STD_VARIANZA:
                out.append((ids[i], ids[j], d, r))
    return out


def fusionar_equivalentes(secuencias, crudos, cadencia_min, cv_estable):
    """
    Fusion aglomerativa: une de a un par (el de menor d) de estados equivalentes,
    el menos poblado en el mas poblado, y recalcula. Devuelve
    (estados_fusionados_por_secuencia, stats_fusionados, mapa {estado: representante}, log).
    """
    est = [np.array(a, dtype=int) for a in crudos]
    stats = estadisticas_por_estado(secuencias, est, cadencia_min)
    mapa = dict((e, e) for e in stats)
    log = []
    for _ in range(len(stats)):
        pares = pares_equivalentes(stats, cv_estable)
        if not pares:
            break
        i, j, d, r = min(pares, key=lambda x: x[2])
        dest, absorbido = (i, j) if stats[i]["n"] >= stats[j]["n"] else (j, i)
        log.append({"absorbido": absorbido, "destino": dest, "d": d, "ratio": r,
                    "n_abs": stats[absorbido]["n"], "n_dest": stats[dest]["n"],
                    "med_abs": stats[absorbido]["mediana"], "med_dest": stats[dest]["mediana"],
                    "sig_abs": stats[absorbido]["std_rob"], "sig_dest": stats[dest]["std_rob"]})
        for k in mapa:
            if mapa[k] == absorbido:
                mapa[k] = dest
        est = [np.array([dest if int(x) == absorbido else int(x) for x in a], dtype=int) for a in est]
        stats = estadisticas_por_estado(secuencias, est, cadencia_min)
    return est, stats, mapa, log


# ============================================================================
# FASE 5: CHANGE-POINTS (con salto de regimen)
# ============================================================================
def detectar_change_points(secuencias, suaves):
    cps, base = [], 0
    for sid, (seq, est) in enumerate(zip(secuencias, suaves)):
        rs = runs_de(est)
        for k in range(1, len(rs)):
            ip0, ip1, vprev = rs[k - 1]
            i0, i1, v = rs[k]
            cps.append({"seq_id": sid, "i": i0, "gidx": base + i0, "timestamp": seq[i0][0],
                        "estado_anterior": vprev, "estado_nuevo": v,
                        "run_prev": (ip0, ip1), "run_new": (i0, i1)})
        base += len(seq)
    return cps


def tipo_cambio(sa, sn, ar_a, ar_n):
    if ar_a == "atipico" or ar_n == "atipico":
        return "atipico"
    d = d_cohen(sa, sn)
    ratio = ratio_std(sa, sn)
    nivel = d >= D_FUSIONABLE
    varianza = ratio >= RATIO_STD_VARIANZA
    if nivel and varianza:
        return "nivel+varianza"
    if nivel:
        return "nivel"
    if varianza:
        return "varianza"
    return "equivalente"


def enriquecer_change_points(cps, secuencias, stats, ranking, ventana, cv_estable):
    rtt_seq = [np.array([x[1] for x in s]) for s in secuencias]
    for cp in cps:
        r, i = rtt_seq[cp["seq_id"]], cp["i"]
        ip0, _ = cp["run_prev"]
        _, in1 = cp["run_new"]
        antes = r[max(ip0, i - ventana):i]
        despues = r[i:min(in1, i + ventana)]
        mb, ma = float(np.median(antes)), float(np.median(despues))
        sa, sn = stats[cp["estado_anterior"]], stats[cp["estado_nuevo"]]
        ar_a, ar_n = arquetipo(sa, cv_estable), arquetipo(sn, cv_estable)
        d = ma - mb
        pct = 100.0 * d / max(mb, 1e-3)
        cp.update({
            "regimen_anterior": ranking[cp["estado_anterior"]],
            "regimen_nuevo": ranking[cp["estado_nuevo"]],
            "mediana_antes_ms": mb, "mediana_despues_ms": ma,
            "salto_ventana_ms": d, "pct_ventana": pct,
            "delta_mediana_estados_ms": sn["mediana"] - sa["mediana"],
            "delta_std_rob_estados_ms": sn["std_rob"] - sa["std_rob"],
            "d_cohen": d_cohen(sa, sn),
            "tipo": tipo_cambio(sa, sn, ar_a, ar_n),
            "supera_umbral_pa": bool(abs(d) > UMBRAL_PA_MS and abs(pct) > UMBRAL_PA_PCT),
            "rtt_ciclo_anterior_ms": float(r[i - 1]) if i > 0 else None,
            "rtt_ciclo_ms": float(r[i]),
        })
    return cps


# ============================================================================
# FASE 6: CORRELACION CON PATH ANALYSIS (con linea base de azar)
# ============================================================================
def dist_min(x, ref_sorted):
    x = np.asarray(x, dtype=float)
    if len(ref_sorted) == 0:
        return np.full(x.shape, np.inf)
    n = len(ref_sorted)
    idx = np.searchsorted(ref_sorted, x)
    left = ref_sorted[np.clip(idx - 1, 0, n - 1)]
    right = ref_sorted[np.clip(idx, 0, n - 1)]
    return np.minimum(np.abs(x - left), np.abs(x - right))


def correlacionar(cps, dict_cat, secuencias, tol, n_perm, seed=12345):
    t0 = parse_ts(secuencias[0][0][0])
    ciclo_ts = [ts for s in secuencias for ts, _ in s]
    ciclo_t = np.array([(parse_ts(ts) - t0).total_seconds() / 60.0 for ts in ciclo_ts])
    es_primero = np.zeros(len(ciclo_t), dtype=bool)
    base = 0
    for s in secuencias:
        es_primero[base] = True
        base += len(s)

    items = []
    for ts, info in dict_cat.items():
        try:
            items.append(((parse_ts(ts) - t0).total_seconds() / 60.0, ts))
        except ValueError:
            continue
    items.sort()
    ev_t = np.array([t for t, _ in items])
    ev_ts = [ts for _, ts in items]
    cp_t = np.array([ciclo_t[cp["gidx"]] for cp in cps])
    eps = 1e-9

    d_ev_ciclo = dist_min(ciclo_t, ev_t)
    cerca_ev_ciclo = d_ev_ciclo <= tol + eps
    d_cp_ciclo = dist_min(ciclo_t, cp_t)
    cerca_cp_ciclo = d_cp_ciclo <= tol + eps

    resultados, n_coinc = [], 0
    for cp in cps:
        t = ciclo_t[cp["gidx"]]
        lo = np.searchsorted(ev_t, t - tol - eps, side="left")
        hi = np.searchsorted(ev_t, t + tol + eps, side="right")
        cats = defaultdict(int)
        for j in range(lo, hi):
            for c in dict_cat[ev_ts[j]]["categorias"]:
                cats[c] += 1
        coincide = len(cats) > 0
        if coincide:
            n_coinc += 1
        cp["coincide"] = coincide
        cp["categoria_path_analysis"] = max(cats.items(), key=lambda x: x[1])[0] if coincide else None
        cp["categorias_en_ventana"] = dict(cats)
        cp["dist_min_evento_min"] = float(d_ev_ciclo[cp["gidx"]]) if len(ev_t) else None
        resultados.append(cp)

    n_cp, n_ev = len(cps), len(ev_t)
    obs_prec = n_coinc / max(n_cp, 1)
    cand = np.where(~es_primero)[0]
    base_prec = float(cerca_ev_ciclo[cand].mean()) if len(cand) else float("nan")

    rng = np.random.RandomState(seed)
    p_prec = p95_prec = float("nan")
    if 0 < n_cp <= len(cand):
        fr = np.array([cerca_ev_ciclo[rng.choice(cand, n_cp, replace=False)].mean()
                       for _ in range(n_perm)])
        p_prec = (1.0 + float((fr >= obs_prec - 1e-12).sum())) / (1.0 + n_perm)
        p95_prec = float(np.percentile(fr, 95))

    d_cp_ev = dist_min(ev_t, cp_t)
    recordado = d_cp_ev <= tol + eps
    obs_rec = float(recordado.mean()) if n_ev else float("nan")
    base_rec = float(cerca_cp_ciclo.mean())
    p_rec = float("nan")
    if 0 < n_ev <= len(ciclo_t):
        fr = np.array([cerca_cp_ciclo[rng.choice(len(ciclo_t), n_ev, replace=False)].mean()
                       for _ in range(n_perm)])
        p_rec = (1.0 + float((fr >= obs_rec - 1e-12).sum())) / (1.0 + n_perm)

    pares = []
    for k in range(n_cp):
        lo = np.searchsorted(ev_t, cp_t[k] - tol - eps, side="left")
        hi = np.searchsorted(ev_t, cp_t[k] + tol + eps, side="right")
        for j in range(lo, hi):
            pares.append((abs(cp_t[k] - ev_t[j]), k, j))
    pares.sort()
    usados_c, usados_e, n11 = set(), set(), 0
    for _, k, j in pares:
        if k not in usados_c and j not in usados_e:
            usados_c.add(k)
            usados_e.add(j)
            n11 += 1

    por_cat = {}
    for j, ts in enumerate(ev_ts):
        for c in dict_cat[ts]["categorias"]:
            v = por_cat.setdefault(c, [0, 0])
            v[0] += 1
            if recordado[j]:
                v[1] += 1

    return {"cps": resultados, "n_cp": n_cp, "n_ev": n_ev, "n_coinc": n_coinc,
            "obs_prec": obs_prec, "base_prec": base_prec, "p_prec": p_prec, "p95_prec": p95_prec,
            "obs_rec": obs_rec, "base_rec": base_rec, "p_rec": p_rec,
            "n_1a1": n11, "por_cat": por_cat, "tol": tol, "n_perm": n_perm}


def interpretar(obs, base, p):
    if base != base or base <= 0 or p != p:
        return "sin base de comparacion"
    ratio = obs / base
    if p < 0.05 and ratio >= 1.5:
        return "por encima del azar ({0:.1f}x, p={1:.3f})".format(ratio, p)
    return "no distinguible del azar ({0:.1f}x, p={1:.3f})".format(ratio, p)


# ============================================================================
# FASE 7: RESIDUOS
# ============================================================================
def analizar_residuos(cps):
    return [cp for cp in cps if not cp["coincide"]]


# ============================================================================
# FASE 7b: SENSIBILIDAD AL SUAVIZADO, PATRON HORARIO, EPISODIOS DE SALTO FINAL
# ============================================================================
def coincidencia_rapida(cps, dict_cat, secuencias, tol):
    """Precision/cobertura frente a las reglas y sus lineas base analiticas (sin permutaciones)."""
    t0 = parse_ts(secuencias[0][0][0])
    ciclo_t = np.array([(parse_ts(ts) - t0).total_seconds() / 60.0 for s in secuencias for ts, _ in s])
    ev_t = np.array(sorted((parse_ts(ts) - t0).total_seconds() / 60.0 for ts in dict_cat))
    cp_t = np.array([ciclo_t[cp["gidx"]] for cp in cps])
    eps = 1e-9
    if len(cp_t) == 0 or len(ev_t) == 0:
        return {"prec": float("nan"), "base_prec": float("nan"), "rec": float("nan"), "base_rec": float("nan")}
    prec = float((dist_min(cp_t, ev_t) <= tol + eps).mean())
    base_prec = float((dist_min(ciclo_t, ev_t) <= tol + eps).mean())
    rec = float((dist_min(ev_t, cp_t) <= tol + eps).mean())
    base_rec = float((dist_min(ciclo_t, cp_t) <= tol + eps).mean())
    return {"prec": prec, "base_prec": base_prec, "rec": rec, "base_rec": base_rec}


def barrido_min_dwell(secuencias, fusionados, dict_cat, tol, valores, elegido):
    filas = []
    for d in sorted(set(list(valores) + [elegido])):
        sua = [np.array(suavizar_estados(a, d), dtype=int) for a in fusionados]
        cps = detectar_change_points(secuencias, sua)
        largos = [fin - ini for a in sua for ini, fin, _ in runs_de(a)]
        c = coincidencia_rapida(cps, dict_cat, secuencias, tol)
        filas.append({"min_dwell": d, "estados": len(set(int(x) for a in sua for x in a)),
                      "cambios": len(cps), "perm_med_ciclos": float(np.median(largos)) if largos else 0.0,
                      "prec": c["prec"], "base_prec": c["base_prec"], "rec": c["rec"],
                      "base_rec": c["base_rec"], "elegido": d == elegido})
    return filas


def analisis_horario(secuencias, suaves, stats_suave, cv_estable, n_perm, seed=12345):
    """Entradas al regimen variable desde uno estable, por hora del dia (hora del timestamp)."""
    arq = dict((e, arquetipo(s, cv_estable)) for e, s in stats_suave.items())
    n_tot, n_var, n_est = np.zeros(24), np.zeros(24), np.zeros(24)
    entradas = np.zeros(24)
    for seq, est in zip(secuencias, suaves):
        for (ts, _), e in zip(seq, est):
            h = parse_ts(ts).hour
            n_tot[h] += 1
            if arq[int(e)] == "variable":
                n_var[h] += 1
            elif arq[int(e)] == "estable":
                n_est[h] += 1
        rs = runs_de(est)
        for k in range(1, len(rs)):
            if arq[rs[k - 1][2]] == "estable" and arq[rs[k][2]] == "variable":
                entradas[parse_ts(seq[rs[k][0]][0]).hour] += 1
    out = {"n_tot": n_tot, "n_var": n_var, "n_est": n_est, "entradas": entradas,
           "n_entradas": int(entradas.sum()), "tiene_variable": bool(n_var.sum() > 0),
           "chi2": float("nan"), "p": float("nan"), "esperadas": np.zeros(24), "n_perm": n_perm}
    n = out["n_entradas"]
    if n == 0 or n_est.sum() == 0:
        return out
    probs = n_est / n_est.sum()
    esp = n * probs
    m = esp > 0
    chi2 = float(((entradas[m] - esp[m]) ** 2 / esp[m]).sum())
    rng = np.random.RandomState(seed)
    sims = rng.multinomial(n, probs, size=n_perm)
    chi_s = ((sims[:, m] - esp[m]) ** 2 / esp[m]).sum(axis=1)
    out.update({"chi2": chi2, "esperadas": esp,
                "p": (1.0 + float((chi_s >= chi2 - 1e-12).sum())) / (1.0 + n_perm)})
    return out


def episodios_salto_final(por_ciclo, hop_modal, secuencias, suaves, ranking, gap_minutes,
                          ts_excluidos, ventana, tol_ruta_min, n_perm, min_len, seed=12345):
    """
    Episodios de ciclos consecutivos cuyo ultimo salto con respuesta NO es el modal.
    Para cada uno: regimen antes/durante/despues, RTT (mediana) antes/durante/despues
    y cambios de ruta (categorias excluidas) a +-tol_ruta_min.
    """
    tss = sorted(ts for ts, h in por_ciclo.items() if h)
    if not tss or hop_modal is None:
        return {"lista": [], "resumen": None}
    tt = [parse_ts(t) for t in tss]
    ultimo = [max(por_ciclo[t].keys()) for t in tss]
    nm = [u != hop_modal for u in ultimo]

    reg = {}
    for seq, est in zip(secuencias, suaves):
        for (ts, _), e in zip(seq, est):
            reg[ts] = ranking[int(e)]
    ts_serie = sorted(reg.keys())

    def minutos(a, b):
        return (tt[b] - tt[a]).total_seconds() / 60.0

    def rtt_modal(i):
        h = por_ciclo[tss[i]]
        return h.get(hop_modal, h[max(h.keys())])

    def vecino(i, paso):
        """hasta 'ventana' ciclos modales contiguos (sin hueco) a partir de i en direccion paso."""
        vals, j, prev = [], i, None
        while 0 <= j < len(tss) and len(vals) < ventana:
            if nm[j]:
                break
            if prev is not None and abs(minutos(min(j, prev), max(j, prev))) > gap_minutes:
                break
            vals.append(rtt_modal(j))
            prev, j = j, j + paso
        return vals

    def reg_cerca(ts, paso):
        k = bisect.bisect_left(ts_serie, ts)
        k = k - 1 if paso < 0 else k
        if k < 0 or k >= len(ts_serie):
            return None
        return reg[ts_serie[k]]

    ex_ts = sorted(ts_excluidos.keys())
    ex_dt = [parse_ts(t) for t in ex_ts]

    lista, i, n = [], 0, len(tss)
    while i < n:
        if not nm[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and nm[j + 1] and minutos(j, j + 1) <= gap_minutes:
            j += 1
        ciclos = list(range(i, j + 1))
        dur = minutos(i, j)
        antes = vecino(i - 1, -1) if i > 0 else []
        desp = vecino(j + 1, +1) if j + 1 < n else []
        dur_v = [por_ciclo[tss[k]][max(por_ciclo[tss[k]].keys())] for k in ciclos]
        en_serie = [reg[tss[k]] for k in ciclos if tss[k] in reg]
        reg_dur = Counter(en_serie).most_common(1)[0][0] if en_serie else None
        lo = tt[i] - timedelta(minutes=tol_ruta_min)
        hi = tt[j] + timedelta(minutes=tol_ruta_min)
        cats = Counter()
        for t, dt in zip(ex_ts, ex_dt):
            if lo <= dt <= hi:
                for c in ts_excluidos[t]:
                    cats[c] += 1
        mb = float(np.median(antes)) if antes else None
        md = float(np.median(dur_v))
        ma = float(np.median(desp)) if desp else None
        r_ant, r_des = reg_cerca(tss[i], -1), reg_cerca(tss[j], +1)
        lista.append({"inicio": tss[i], "fin": tss[j], "n_ciclos": len(ciclos), "duracion_min": dur,
                      "hop_ultimo": Counter(ultimo[k] for k in ciclos).most_common(1)[0][0],
                      "reg_antes": r_ant, "reg_durante": reg_dur, "reg_despues": r_des,
                      "med_antes": mb, "med_durante": md, "med_despues": ma,
                      "delta_durante": (md - mb) if mb is not None else None,
                      "delta_despues": (ma - mb) if (mb is not None and ma is not None) else None,
                      "cambia_regimen": (r_ant is not None and r_des is not None and r_ant != r_des),
                      "ruta_cerca": "|".join("{0}:{1}".format(k, v) for k, v in sorted(cats.items())),
                      "hay_ruta": len(cats) > 0, "en_serie": len(en_serie) > 0,
                      "largo": len(ciclos) >= min_len})
        i = j + 1

    largos = [e for e in lista if e["largo"]]
    resumen = {"n_total": len(lista), "n_largos": len(largos), "min_len": min_len,
               "n_cambia": sum(1 for e in largos if e["cambia_regimen"]),
               "n_ruta": sum(1 for e in largos if e["hay_ruta"]),
               "n_ruta_total": len(ex_ts), "tol_ruta": tol_ruta_min, "p_ruta": float("nan"),
               "base_ruta": float("nan"),
               "n_salto_umbral": sum(1 for e in largos if e["delta_durante"] is not None
                                     and abs(e["delta_durante"]) > UMBRAL_PA_MS)}
    if largos and ex_ts:
        t0 = tt[0]
        ciclo_t = np.array([(x - t0).total_seconds() / 60.0 for x in tt])
        ex_t = np.array([(x - t0).total_seconds() / 60.0 for x in ex_dt])
        tol = float(tol_ruta_min)
        mat = []
        for e in largos:
            lo_i = np.searchsorted(ex_t, ciclo_t - tol - 1e-9, side="left")
            hi_i = np.searchsorted(ex_t, ciclo_t + e["duracion_min"] + tol + 1e-9, side="right")
            mat.append(hi_i > lo_i)
        mat = np.array(mat)
        resumen["base_ruta"] = float(mat.mean())
        rng = np.random.RandomState(seed)
        obs = resumen["n_ruta"] / float(len(largos))
        fr = np.zeros(n_perm)
        for s_ in range(n_perm):
            idx = rng.randint(0, len(ciclo_t), size=len(largos))
            fr[s_] = mat[np.arange(len(largos)), idx].mean()
        resumen["p_ruta"] = (1.0 + float((fr >= obs - 1e-12).sum())) / (1.0 + n_perm)
    return {"lista": lista, "resumen": resumen}


# ============================================================================
# INFORME (consola + resumen_ejecucion.txt identicos)
# ============================================================================
def tabla_estados(stats, ranking, cv_estable, titulo):
    L = [titulo,
         "  {0:<4}{1:>7}{2:>8}{3:>7}{4:>17}{5:>9}{6:>9}{7:>7}{8:>16}  {9}".format(
             "Reg", "Estado", "Ciclos", "%", "Media+-Desv(ms)", "Mediana", "Sig.rob", "Apar.",
             "Perm.med(c/min)", "Arquetipo")]
    for e in sorted(stats.keys(), key=lambda k: (ranking[k], k)):
        s = stats[e]
        L.append("  R{0:<3}{1:>7}{2:>8}{3:>7}{4:>17}{5:>9}{6:>9}{7:>7}{8:>16}  {9}".format(
            ranking[e], e, s["n"], "{0:.1f}".format(s["pct"]),
            "{0:.2f}+-{1:.2f}".format(s["media"], s["std"]), f2(s["mediana"]), f2(s["std_rob"]),
            s["apariciones"], "{0:.0f}/{1:.0f}".format(s["perm_med_ciclos"], s["perm_med_min"]),
            arquetipo(s, cv_estable)))
    return L


def construir_informe(ctx):
    a = ctx["args"]
    L = []
    L.append("=" * 74)
    L.append("INFORME HDP-HMM (v3) - caracterizacion de RTT end-to-end")
    L.append("=" * 74)
    if ctx["ids"][0]:
        L.append("Measurement {0} | Probe {1}".format(*ctx["ids"]))
    L.append("Fecha de ejecucion: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    L.append("")
    L.append("1. DATOS")
    s = ctx["serie"]
    L.append("  - {0} ciclos, {1} secuencia(s) (hueco > {2} min), cadencia mediana {3:.1f} min".format(
        len(s), len(ctx["secuencias"]), a.gap_minutes, ctx["cadencia"]))
    L.append("  - Rango: {0} a {1}".format(s[0][0], s[-1][0]))
    r = np.array([x[1] for x in s])
    L.append("  - RTT: min {0:.2f} | media {1:.2f} | mediana {2:.2f} | max {3:.2f} | desvio {4:.2f} ms".format(
        r.min(), r.mean(), np.median(r), r.max(), r.std()))
    top = ", ".join("hop {0}: {1} ({2:.1f}%)".format(h, n, 100.0 * n / max(sum(ctx["salto_cont"].values()), 1))
                    for h, n in ctx["salto_cont"].most_common(4))
    L.append("  - Ultimo salto con respuesta por ciclo: {0}".format(top))
    if ctx["n_no_modal"] > 0 and not a.usar_hop_modal:
        L.append("  [AVISO] {0} ciclos terminan en un salto distinto del habitual (hop {1}): "
                 "su RTT puede no ser end-to-end. Usar --usar-hop-modal para excluirlos.".format(
                     ctx["n_no_modal"], ctx["hop_modal"]))
    elif a.usar_hop_modal:
        L.append("  - --usar-hop-modal: se uso solo el hop {0} (ciclos excluidos: {1}).".format(
            ctx["hop_modal"], ctx["n_no_modal"]))
    ci = ctx["cat_info"]
    L.append("  - Ground truth: {0} lineas, {1} timestamps con evento tras filtrar".format(
        ci["n_lineas"], len(ctx["dict_cat"])))
    if ci["excluidas"]:
        L.append("    categorias de cambio de ruta excluidas (enfoque RTT): {0}".format(
            ", ".join("{0}={1}".format(k, v) for k, v in sorted(ci["excluidas"].items()))))
    L.append("    categorias: {0}".format(
        ", ".join("{0}={1}".format(k, v) for k, v in ci["por_cat"].most_common())))

    L.append("")
    L.append("2. MODELO")
    st_c, st_s = ctx["stats_crudo"], ctx["stats_suave"]
    L.append("  - K truncado = {0}; sF = {1:.2f}; hmmKappa = {2}; decodificacion = {3}".format(
        a.K, ctx["sF"], a.hmm_kappa, ctx["decod"]))
    L.append("  - Estados activos: {0} crudos -> {1} tras fusion -> {2} tras suavizado (--min-dwell {3})".format(
        len(st_c), len(ctx["stats_fus"]), len(st_s), a.min_dwell))
    cc = ctx["cambios"]
    L.append("  - Cambios de estado: argmax {0}{1} | crudos usados {2} | tras suavizado {3}".format(
        cc["argmax"], (" | viterbi {0}".format(cc["viterbi"]) if cc["viterbi"] is not None else ""),
        cc["crudo"], cc["suave"]))

    L.append("")
    L.append("3. REGIMENES (R1 = RTT medio mas bajo; orden por media cruda)")
    L += tabla_estados(st_s, ctx["ranking"], a.cv_estable, "  Tras fusion y suavizado (base de los change-points):")
    L.append("")
    L += tabla_estados(st_c, ctx["ranking"], a.cv_estable, "  Crudos (antes de fusionar y suavizar):")
    L.append("  Sig.rob = 1.4826*MAD. Arquetipos: estable = sig.rob/mediana < {0:.2f}; variable = resto; "
             "atipico = < 1% de los ciclos.".format(a.cv_estable))
    L.append("  Los estados fusionados comparten regimen (varias filas R<k> en la tabla de crudos).")
    L.append("")
    if a.no_fusionar:
        L.append("  Fusion desactivada (--no-fusionar). Pares que se habrian fusionado "
                 "(d robusta < {0}, cociente de desvios < {1}):".format(D_FUSIONABLE, RATIO_STD_VARIANZA))
        pe = pares_equivalentes(st_c, a.cv_estable)
        if not pe:
            L.append("    (ninguno)")
        for i, j, d, r in pe:
            L.append("    - estados {0} y {1}: d={2:.2f}, cociente de desvios {3:.2f}".format(i, j, d, r))
    else:
        L.append("  Fusion de estados equivalentes (d robusta < {0} y cociente de desvios < {1}, ninguno "
                 "atipico):".format(D_FUSIONABLE, RATIO_STD_VARIANZA))
        if not ctx["log_fusion"]:
            L.append("    (ningun par equivalente: nada que fusionar)")
        for m in ctx["log_fusion"]:
            L.append("    - estado {0} ({1} ciclos, mediana {2:.2f}, sig.rob {3:.2f}) absorbido por estado {4} "
                     "({5} ciclos, mediana {6:.2f}, sig.rob {7:.2f}): d={8:.2f}, cociente {9:.2f}".format(
                         m["absorbido"], m["n_abs"], m["med_abs"], m["sig_abs"], m["destino"], m["n_dest"],
                         m["med_dest"], m["sig_dest"], m["d"], m["ratio"]))

    L.append("")
    L.append("4. TRANSICIONES ENTRE REGIMENES (empiricas, tras suavizado)")
    trans = ctx["transiciones"]
    if trans:
        for (i, j), n, p in trans[:8]:
            L.append("  R{0} -> R{1}: {2} veces ({3:.0f}% de las salidas de R{0})".format(
                ctx["ranking"][i], ctx["ranking"][j], n, 100 * p))
        if len(trans) > 8:
            L.append("  ... ({0} transiciones mas en transiciones_*.csv)".format(len(trans) - 8))
    else:
        L.append("  (sin transiciones)")

    cps = ctx["cps"]
    L.append("")
    L.append("5. CHANGE-POINTS ({0})".format(len(cps)))
    tipos = Counter(cp["tipo"] for cp in cps)
    L.append("  Por tipo: {0}".format(", ".join("{0}={1}".format(k, v) for k, v in tipos.most_common())))
    L.append("    nivel/varianza = cambio de regimen real; equivalente = entre estados casi identicos "
             "(probable artefacto); atipico = involucra un estado raro.")
    pares = Counter((cp["regimen_anterior"], cp["regimen_nuevo"]) for cp in cps)
    L.append("  Pares mas frecuentes: {0}".format(
        ", ".join("R{0}->R{1}: {2}".format(a_, b_, n) for (a_, b_), n in pares.most_common(5))))
    n_sup = sum(1 for cp in cps if cp["supera_umbral_pa"])
    L.append("  Salto de MEDIANA (ventana de {0} ciclos) que supera el umbral de Path Analysis "
             "(>{1:g} ms y >{2:g}%): {3} de {4}".format(a.ventana_salto, UMBRAL_PA_MS, UMBRAL_PA_PCT,
                                                         n_sup, len(cps)))
    if cps:
        mags = np.array([abs(cp["salto_ventana_ms"]) for cp in cps])
        L.append("  |Salto de mediana|: mediana {0:.2f} ms, p90 {1:.2f} ms, max {2:.2f} ms".format(
            np.median(mags), np.percentile(mags, 90), mags.max()))

    co = ctx["corr"]
    L.append("")
    L.append("6. CORRELACION CON PATH ANALYSIS (tolerancia +-{0} min)".format(co["tol"]))
    L.append("  Change-points con un evento cerca: {0}/{1} ({2:.1f}%) | azar esperado {3:.1f}% (p95 {4:.1f}%) -> {5}".format(
        co["n_coinc"], co["n_cp"], 100 * co["obs_prec"], 100 * co["base_prec"], 100 * co["p95_prec"],
        interpretar(co["obs_prec"], co["base_prec"], co["p_prec"])))
    L.append("  Eventos con un change-point cerca: {0:.1f}% de {1} | azar esperado {2:.1f}% -> {3}".format(
        100 * co["obs_rec"], co["n_ev"], 100 * co["base_rec"],
        interpretar(co["obs_rec"], co["base_rec"], co["p_rec"])))
    L.append("  Emparejamiento uno a uno: {0} pares".format(co["n_1a1"]))
    if co["por_cat"]:
        L.append("  Cobertura por categoria (evento con un change-point cerca):")
        for c, (n, k) in sorted(co["por_cat"].items(), key=lambda x: -x[1][0]):
            L.append("    {0:<20} {1:>4} eventos, {2:>4.0f}% con change-point cerca".format(
                c, n, 100.0 * k / max(n, 1)))
    L.append("  Lectura: las reglas marcan saltos puntuales entre ciclos consecutivos; el HDP-HMM marca")
    L.append("  cambios de regimen sostenidos. Si la cobertura no supera al azar, miden fenomenos distintos.")

    res = ctx["residuos"]
    L.append("")
    L.append("7. RESIDUOS: change-points sin evento de Path Analysis ({0})".format(len(res)))
    if res:
        rt = Counter(cp["tipo"] for cp in res)
        L.append("  Por tipo: {0}".format(", ".join("{0}={1}".format(k, v) for k, v in rt.most_common())))
        reales = [cp for cp in res if cp["tipo"] in ("nivel", "varianza", "nivel+varianza")]
        sub = [cp for cp in reales if not cp["supera_umbral_pa"]]
        L.append("  Cambios de regimen reales sin evento: {0}; de ellos, por debajo del umbral de Path "
                 "Analysis: {1} (contribucion propia del HDP-HMM)".format(len(reales), len(sub)))
        L.append("  Equivalentes (probable artefacto): {0}; atipicos: {1}".format(
            rt.get("equivalente", 0), rt.get("atipico", 0)))

    # 8. barrido min-dwell
    L.append("")
    L.append("8. SENSIBILIDAD AL SUAVIZADO (--min-dwell; sobre estados ya fusionados)")
    L.append("  {0:>9}{1:>9}{2:>9}{3:>14}{4:>20}{5:>20}".format(
        "min-dwell", "estados", "cambios", "perm.med(c)", "CP con evento(azar)", "evento con CP(azar)"))
    for f_ in ctx["barrido"]:
        def pc(x):
            return "{0:.0f}%".format(100 * x) if x == x else "-"
        L.append("  {0:>9}{1:>9}{2:>9}{3:>14.1f}{4:>20}{5:>20}{6}".format(
            f_["min_dwell"], f_["estados"], f_["cambios"], f_["perm_med_ciclos"],
            "{0} ({1})".format(pc(f_["prec"]), pc(f_["base_prec"])),
            "{0} ({1})".format(pc(f_["rec"]), pc(f_["base_rec"])),
            "   <- elegido" if f_["elegido"] else ""))
    L.append("  Lectura: si el numero de estados y la coincidencia con las reglas se mantienen al variar "
             "min-dwell, el resultado no depende de ese parametro.")

    # 9. patron horario
    hz = ctx["horario"]
    L.append("")
    L.append("9. PATRON HORARIO (hora del timestamp del CSV, tipicamente UTC)")
    if not hz["tiene_variable"]:
        L.append("  No hay regimen 'variable': no se analiza el patron horario.")
    elif hz["n_entradas"] == 0:
        L.append("  Hay regimen variable pero ninguna entrada desde un regimen estable.")
    else:
        L.append("  Entradas a regimen variable desde uno estable: {0}".format(hz["n_entradas"]))
        L.append("  {0:>9}{1:>9}{2:>14}{3:>10}{4:>11}".format("hora", "ciclos", "% variable", "entradas", "esperadas"))
        for b in range(8):
            sl = slice(3 * b, 3 * b + 3)
            nt, nv = hz["n_tot"][sl].sum(), hz["n_var"][sl].sum()
            L.append("  {0:>9}{1:>9d}{2:>14}{3:>10d}{4:>11.1f}".format(
                "{0:02d}-{1:02d}h".format(3 * b, 3 * b + 3), int(nt),
                "{0:.0f}%".format(100.0 * nv / max(nt, 1)), int(hz["entradas"][sl].sum()),
                float(hz["esperadas"][sl].sum())))
        L.append("  Chi-cuadrado de las entradas por hora vs. esperado segun ciclos en regimen estable: "
                 "{0:.1f} (p={1:.3f}, {2} simulaciones) -> {3}".format(
                     hz["chi2"], hz["p"], hz["n_perm"],
                     "patron horario significativo" if hz["p"] < 0.05 else "no distinguible de un patron uniforme"))
        if hz["n_entradas"] < 20:
            L.append("  [AVISO] Pocas entradas ({0}): el test tiene poca potencia.".format(hz["n_entradas"]))

    # 10. episodios de salto final
    ep = ctx["episodios"]
    L.append("")
    L.append("10. EPISODIOS DE SALTO FINAL DISTINTO DEL MODAL (hop {0})".format(ctx["hop_modal"]))
    rs = ep["resumen"]
    if rs is None or rs["n_total"] == 0:
        L.append("  Ningun ciclo termina en un salto distinto del modal.")
    else:
        L.append("  Episodios (ciclos consecutivos): {0}; con >= {1} ciclos: {2}".format(
            rs["n_total"], rs["min_len"], rs["n_largos"]))
        lg = [e for e in ep["lista"] if e["largo"]]
        if lg:
            L.append("  {0:<20}{1:>7}{2:>6}{3:>8}{4:>8}{5:>8}{6:>10}{7:>10}  {8}".format(
                "inicio", "ciclos", "hop", "R antes", "R dur.", "R desp.", "RTT antes", "RTT dur.", "ruta cerca"))
            def fm(x):
                return "-" if x is None else "{0:.1f}".format(x)
            def fr(x):
                return "-" if x is None else "R{0}".format(x)
            for e in lg[:12]:
                L.append("  {0:<20}{1:>7}{2:>6}{3:>8}{4:>8}{5:>8}{6:>10}{7:>10}  {8}".format(
                    e["inicio"], e["n_ciclos"], e["hop_ultimo"], fr(e["reg_antes"]), fr(e["reg_durante"]),
                    fr(e["reg_despues"]), fm(e["med_antes"]), fm(e["med_durante"]), e["ruta_cerca"] or "-"))
            if len(lg) > 12:
                L.append("  ... ({0} episodios mas en episodios_salto_final_*.csv)".format(len(lg) - 12))
            L.append("  Con cambio de regimen entre antes y despues: {0}/{1}".format(rs["n_cambia"], rs["n_largos"]))
            L.append("  Con |RTT durante - RTT antes| > {0:g} ms: {1}/{2}".format(
                UMBRAL_PA_MS, rs["n_salto_umbral"], rs["n_largos"]))
            if rs["n_ruta_total"] == 0:
                L.append("  Sin eventos de cambio de ruta en el ground truth (o excluidos): no se puede "
                         "contrastar con rutas. Usar un CSV de categorias que los incluya.")
            else:
                L.append("  Con un cambio de ruta a +-{0:g} min: {1}/{2} ({3:.0f}%) | azar esperado {4:.0f}% (p={5:.3f})".format(
                    rs["tol_ruta"], rs["n_ruta"], rs["n_largos"], 100.0 * rs["n_ruta"] / max(rs["n_largos"], 1),
                    100 * rs["base_ruta"], rs["p_ruta"]))
        L.append("  Lectura: un episodio sin cambio de ruta ni cambio de regimen sugiere un destino que "
                 "deja de responder; con cambio de ruta y de regimen, el regimen sigue a la ruta.")

    L.append("")
    L.append("11. ARCHIVOS")
    for p in ctx["archivos"]:
        L.append("  - {0}".format(p))
    L.append("=" * 74)
    return L


# ============================================================================
# FASE 8: EXPORTACION
# ============================================================================
def transiciones_empiricas(cps):
    cont = Counter((cp["estado_anterior"], cp["estado_nuevo"]) for cp in cps)
    salidas = Counter()
    for (i, _), n in cont.items():
        salidas[i] += n
    out = [((i, j), n, n / max(salidas[i], 1)) for (i, j), n in cont.items()]
    out.sort(key=lambda x: -x[1])
    return out


def exportar(ctx, output_dir, suf):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rk = ctx["ranking"]
    archivos = []

    def ruta(nombre):
        base, ext = os.path.splitext(nombre)
        p = os.path.join(output_dir, base + suf + ext)
        archivos.append(p)
        return p

    # 1. estados por ciclo (columnas originales primero: compatible con caracterizar_regimenes.py)
    with abrir_csv_w(ruta("estados_hdphmm.csv")) as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "rtt_ms", "seq_id", "state_raw", "state", "regimen_raw", "regimen",
                    "state_fusion"])
        for sid, (seq, cr, su, fu) in enumerate(zip(ctx["secuencias"], ctx["crudos"], ctx["suaves"],
                                                    ctx["fusionados"])):
            for (ts, rtt), ec, es_, ef in zip(seq, cr, su, fu):
                w.writerow([ts, "{0:.3f}".format(rtt), sid, int(ec), int(es_),
                            "R{0}".format(rk[int(ec)]), "R{0}".format(rk[int(es_)]), int(ef)])

    # 2. resumen por estado (suavizado y crudo)
    with abrir_csv_w(ruta("estados_resumen.csv")) as f:
        w = csv.writer(f)
        w.writerow(["fuente", "regimen", "estado", "ciclos", "pct", "media_ms", "std_ms", "mediana_ms",
                    "apariciones", "permanencia_mediana_ciclos", "permanencia_mediana_min", "arquetipo",
                    "p_autotransicion_modelo", "std_robusto_ms"])
        P = ctx["P"]
        for fuente, st in (("suavizado", ctx["stats_suave"]), ("fusionado", ctx["stats_fus"]),
                           ("crudo", ctx["stats_crudo"])):
            for e in sorted(st.keys(), key=lambda k: (rk[k], k)):
                s = st[e]
                pa = ""
                if P is not None and e < P.shape[0]:
                    pa = "{0:.4f}".format(P[e, e])
                w.writerow([fuente, "R{0}".format(rk[e]), e, s["n"], "{0:.2f}".format(s["pct"]),
                            "{0:.3f}".format(s["media"]), "{0:.3f}".format(s["std"]),
                            "{0:.3f}".format(s["mediana"]), s["apariciones"],
                            "{0:.1f}".format(s["perm_med_ciclos"]), "{0:.1f}".format(s["perm_med_min"]),
                            arquetipo(s, ctx["args"].cv_estable), pa, "{0:.3f}".format(s["std_rob"])])

    # 3. transiciones
    with abrir_csv_w(ruta("transiciones.csv")) as f:
        w = csv.writer(f)
        w.writerow(["regimen_origen", "regimen_destino", "estado_origen", "estado_destino", "veces",
                    "prob_empirica", "prob_modelo"])
        P = ctx["P"]
        for (i, j), n, p in ctx["transiciones"]:
            pm = ""
            if P is not None and i < P.shape[0] and j < P.shape[1]:
                pm = "{0:.4f}".format(P[i, j])
            w.writerow(["R{0}".format(rk[i]), "R{0}".format(rk[j]), i, j, n, "{0:.4f}".format(p), pm])

    # 4. change-points con salto de regimen y correlacion
    cols = ["timestamp", "estado_anterior", "estado_nuevo", "regimen_anterior", "regimen_nuevo", "tipo",
            "salto_ventana_ms", "pct_ventana", "mediana_antes_ms", "mediana_despues_ms",
            "delta_mediana_estados_ms", "delta_std_rob_estados_ms", "d_cohen", "supera_umbral_pa",
            "categoria_path_analysis", "categorias_en_ventana", "dist_min_evento_min", "coincide"]
    with abrir_csv_w(ruta("change_points_correlacion.csv")) as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cp in ctx["cps"]:
            cats = "|".join("{0}:{1}".format(k, v) for k, v in sorted(cp["categorias_en_ventana"].items()))
            dm = cp["dist_min_evento_min"]
            w.writerow([cp["timestamp"], cp["estado_anterior"], cp["estado_nuevo"],
                        "R{0}".format(cp["regimen_anterior"]), "R{0}".format(cp["regimen_nuevo"]), cp["tipo"],
                        "{0:.3f}".format(cp["salto_ventana_ms"]), "{0:.2f}".format(cp["pct_ventana"]),
                        "{0:.3f}".format(cp["mediana_antes_ms"]), "{0:.3f}".format(cp["mediana_despues_ms"]),
                        "{0:.3f}".format(cp["delta_mediana_estados_ms"]),
                        "{0:.3f}".format(cp["delta_std_rob_estados_ms"]), "{0:.3f}".format(cp["d_cohen"]),
                        cp["supera_umbral_pa"], cp["categoria_path_analysis"] or "NINGUNA", cats,
                        "" if dm is None else "{0:.1f}".format(dm), cp["coincide"]])

    # 5. residuos
    with abrir_csv_w(ruta("residuos_contribucion_original.csv")) as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "estado_anterior", "estado_nuevo", "regimen_anterior", "regimen_nuevo", "tipo",
                    "salto_ventana_ms", "pct_ventana", "supera_umbral_pa",
                    "rtt_ciclo_anterior_ms", "rtt_ciclo_ms"])
        for cp in ctx["residuos"]:
            ra = cp["rtt_ciclo_anterior_ms"]
            w.writerow([cp["timestamp"], cp["estado_anterior"], cp["estado_nuevo"],
                        "R{0}".format(cp["regimen_anterior"]), "R{0}".format(cp["regimen_nuevo"]), cp["tipo"],
                        "{0:.3f}".format(cp["salto_ventana_ms"]), "{0:.2f}".format(cp["pct_ventana"]),
                        cp["supera_umbral_pa"], "" if ra is None else "{0:.3f}".format(ra),
                        "{0:.3f}".format(cp["rtt_ciclo_ms"])])

    # 5b. barrido de min-dwell
    with abrir_csv_w(ruta("barrido_min_dwell.csv")) as f:
        w = csv.writer(f)
        w.writerow(["min_dwell", "estados", "cambios", "permanencia_mediana_ciclos", "cp_con_evento",
                    "cp_con_evento_azar", "evento_con_cp", "evento_con_cp_azar", "elegido"])
        for r_ in ctx["barrido"]:
            w.writerow([r_["min_dwell"], r_["estados"], r_["cambios"], "{0:.1f}".format(r_["perm_med_ciclos"]),
                        "{0:.4f}".format(r_["prec"]), "{0:.4f}".format(r_["base_prec"]),
                        "{0:.4f}".format(r_["rec"]), "{0:.4f}".format(r_["base_rec"]), r_["elegido"]])

    # 5c. patron horario
    hz = ctx["horario"]
    with abrir_csv_w(ruta("por_hora.csv")) as f:
        w = csv.writer(f)
        w.writerow(["hora", "ciclos", "ciclos_variable", "pct_variable", "ciclos_estable",
                    "entradas_a_variable", "entradas_esperadas"])
        for h in range(24):
            w.writerow([h, int(hz["n_tot"][h]), int(hz["n_var"][h]),
                        "{0:.2f}".format(100.0 * hz["n_var"][h] / max(hz["n_tot"][h], 1)),
                        int(hz["n_est"][h]), int(hz["entradas"][h]), "{0:.2f}".format(float(hz["esperadas"][h]))])

    # 5d. episodios de salto final
    with abrir_csv_w(ruta("episodios_salto_final.csv")) as f:
        w = csv.writer(f)
        w.writerow(["inicio", "fin", "n_ciclos", "duracion_min", "hop_ultimo", "regimen_antes", "regimen_durante",
                    "regimen_despues", "rtt_mediana_antes", "rtt_mediana_durante", "rtt_mediana_despues",
                    "delta_durante_ms", "delta_despues_ms", "cambia_regimen", "ruta_cerca", "en_serie", "largo"])
        def v(x):
            return "" if x is None else "{0:.3f}".format(x)
        def rr(x):
            return "" if x is None else "R{0}".format(x)
        for e in ctx["episodios"]["lista"]:
            w.writerow([e["inicio"], e["fin"], e["n_ciclos"], "{0:.0f}".format(e["duracion_min"]), e["hop_ultimo"],
                        rr(e["reg_antes"]), rr(e["reg_durante"]), rr(e["reg_despues"]), v(e["med_antes"]),
                        v(e["med_durante"]), v(e["med_despues"]), v(e["delta_durante"]), v(e["delta_despues"]),
                        e["cambia_regimen"], e["ruta_cerca"], e["en_serie"], e["largo"]])

    # 6. eventos de segmento (formato original, compatible)
    with abrir_csv_w(ruta("segmentos_hdphmm.csv")) as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "categoria", "estado_anterior", "estado_nuevo"])
        for cp in ctx["cps"]:
            w.writerow([cp["timestamp"], "cambio_segmento_hdphmm", cp["estado_anterior"], cp["estado_nuevo"]])

    html_path = ruta("linea_tiempo_hdphmm.html")
    resumen_path = ruta("resumen_ejecucion.txt")
    ctx["archivos"] = archivos
    informe = construir_informe(ctx)
    texto = "\n".join(informe) + "\n"
    escribir_texto(resumen_path, texto)
    try:
        escribir_texto(html_path, generar_html(ctx, informe))
    except Exception as e:  # el HTML es accesorio: nunca debe impedir imprimir el informe
        print("[AVISO] No se pudo generar el HTML ({0}: {1}).".format(type(e).__name__, e))
    return informe


# ============================================================================
# LINEA DE TIEMPO HTML (SVG, sin dependencias)
# ============================================================================
def generar_html(ctx, informe):
    secuencias, suaves = ctx["secuencias"], ctx["suaves"]
    rk, stats = ctx["ranking"], ctx["stats_suave"]
    t0 = parse_ts(secuencias[0][0][0])
    tmax = parse_ts(secuencias[-1][-1][0])
    cad = max(ctx["cadencia"], 1.0)

    def mins(ts):
        return (parse_ts(ts) - t0).total_seconds() / 60.0

    total_min = max((tmax - t0).total_seconds() / 60.0 + cad, 1.0)
    W, H = 1400, 380
    ml, mr, mt, mb = 72, 20, 16, 86
    pw, ph = W - ml - mr, H - mt - mb
    rtts = np.array([x[1] for s in secuencias for x in s])
    ymax = float(min(rtts.max(), np.percentile(rtts, 99.5) * 1.15))
    ymax = max(ymax, 1.0)

    def X(m):
        return ml + pw * m / total_min

    def Y(v):
        return mt + ph * (1.0 - min(v, ymax) / ymax)

    def color(e):
        return PALETA[(rk[e] - 1) % len(PALETA)]

    p = []
    # bandas por regimen
    for seq, est in zip(secuencias, suaves):
        for ini, fin, val in runs_de(est):
            x0 = X(mins(seq[ini][0]))
            x1 = X(mins(seq[fin - 1][0]) + cad)
            p.append('<rect x="{0:.1f}" y="{1}" width="{2:.1f}" height="{3}" fill="{4}" opacity="0.22">'
                     '<title>R{5} (estado {6}): {7} a {8}, {9} ciclos</title></rect>'.format(
                         x0, mt, max(x1 - x0, 0.6), ph, color(val), rk[val], val,
                         seq[ini][0], seq[fin - 1][0], fin - ini))
    # episodios de salto final distinto del modal (banda roja tenue)
    for e in ctx["episodios"]["lista"]:
        if not e["largo"]:
            continue
        x0 = X(mins(e["inicio"]))
        x1 = X(mins(e["fin"]) + cad)
        p.append('<rect x="{0:.1f}" y="{1}" width="{2:.1f}" height="{3}" fill="#d32f2f" opacity="0.18">'
                 '<title>Episodio hop {4}: {5} a {6}, {7} ciclos, ruta cerca: {8}</title></rect>'.format(
                     x0, mt, max(x1 - x0, 2.0), ph, e["hop_ultimo"], e["inicio"], e["fin"], e["n_ciclos"],
                     e["ruta_cerca"] or "no"))
    # ejes y grilla
    for k in range(0, 6):
        v = ymax * k / 5.0
        y = Y(v)
        p.append('<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" class="grid"/>'.format(ml, y, W - mr))
        p.append('<text x="{0}" y="{1:.1f}" class="ax" text-anchor="end">{2:.1f}</text>'.format(ml - 6, y + 4, v))
    n_ticks = 8
    for k in range(n_ticks + 1):
        m = total_min * k / n_ticks
        x = X(m)
        dt = t0 + timedelta(seconds=(tmax - t0).total_seconds() * k / float(n_ticks))
        p.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" class="grid"/>'.format(x, mt, mt + ph))
        p.append('<text x="{0:.1f}" y="{1}" class="ax" text-anchor="middle">{2}</text>'.format(
            x, mt + ph + 16, dt.strftime("%m-%d")))
    # serie RTT
    for seq in secuencias:
        pts = " ".join("{0:.1f},{1:.1f}".format(X(mins(ts)), Y(r)) for ts, r in seq)
        p.append('<polyline points="{0}" class="rtt"/>'.format(pts))
        for ts, r in seq:
            if r > ymax:
                p.append('<circle cx="{0:.1f}" cy="{1}" r="2.6" fill="#d32f2f"><title>{2}: {3:.1f} ms '
                         '(fuera de escala)</title></circle>'.format(X(mins(ts)), mt + 2, ts, r))
    # change-points
    for cp in ctx["cps"]:
        x = X(mins(cp["timestamp"]))
        cls = "cp-ok" if cp["coincide"] else "cp-no"
        p.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" class="{3}"><title>{4}: R{5}->R{6}, {7}, '
                 'salto {8:+.2f} ms</title></line>'.format(
                     x, mt, mt + ph, cls, cp["timestamp"], cp["regimen_anterior"], cp["regimen_nuevo"],
                     cp["tipo"], cp["salto_ventana_ms"]))
    # filas inferiores: change-points y eventos
    ycp, yev = mt + ph + 30, mt + ph + 50
    p.append('<text x="{0}" y="{1}" class="ax" text-anchor="end">HDP-HMM</text>'.format(ml - 6, ycp + 4))
    p.append('<text x="{0}" y="{1}" class="ax" text-anchor="end">Reglas</text>'.format(ml - 6, yev + 4))
    for cp in ctx["cps"]:
        x = X(mins(cp["timestamp"]))
        p.append('<rect x="{0:.1f}" y="{1}" width="2" height="10" class="{2}"/>'.format(
            x - 1, ycp - 5, "cp-ok-b" if cp["coincide"] else "cp-no-b"))
    for ts, info in ctx["dict_cat"].items():
        try:
            x = X(mins(ts))
        except ValueError:
            continue
        cats = sorted(info["categorias"])
        c = COLOR_CAT.get(cats[0], "#888")
        p.append('<rect x="{0:.1f}" y="{1}" width="2" height="10" fill="{2}"><title>{3}: {4}</title></rect>'.format(
            x - 1, yev - 5, c, ts, ", ".join(cats)))
    svg = '<svg viewBox="0 0 {0} {1}" width="100%" role="img" aria-label="Linea de tiempo del RTT por regimen">{2}</svg>'.format(
        W, H, "".join(p))

    leg = []
    for e in sorted(stats.keys(), key=lambda k: rk[k]):
        s = stats[e]
        leg.append('<span class="lg"><i style="background:{0}"></i>R{1} (estado {2}): {3:.2f}+-{4:.2f} ms, '
                   '{5:.1f}% ({6})</span>'.format(color(e), rk[e], e, s["media"], s["std"], s["pct"],
                                                   arquetipo(s, ctx["args"].cv_estable)))
    leg_ev = []
    for c in sorted(ctx["cat_info"]["por_cat"].keys()):
        leg_ev.append('<span class="lg"><i style="background:{0}"></i>{1}</span>'.format(
            COLOR_CAT.get(c, "#888"), esc(c)))

    ids = "Measurement {0} / Probe {1}".format(*ctx["ids"]) if ctx["ids"][0] else ""
    html = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Linea de tiempo HDP-HMM</title>
<style>
:root{--bg:#fff;--fg:#222;--mut:#666;--grid:#e3e3e3;--line:#222;--card:#f6f6f6}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#aaa;--grid:#333;--line:#ddd;--card:#222}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:12px}
.card{background:var(--card);border-radius:8px;padding:12px;margin:12px 0;overflow-x:auto}
.grid{stroke:var(--grid);stroke-width:1}.ax{fill:var(--mut);font-size:11px}
.rtt{fill:none;stroke:var(--line);stroke-width:1}
.cp-ok{stroke:#2e7d32;stroke-width:1;opacity:.7}.cp-no{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 3;opacity:.55}
.cp-ok-b{fill:#2e7d32}.cp-no-b{fill:var(--mut)}
.lg{display:inline-block;margin:2px 14px 2px 0;white-space:nowrap}.lg i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-2px}
pre{margin:0;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre}
</style></head><body>
<h1>Linea de tiempo del RTT por regimen</h1>
<div class="sub">__IDS__ - bandas = regimen tras suavizado; lineas verticales = change-points (verde = hay evento de reglas cerca, gris punteado = sin evento); puntos rojos arriba = RTT fuera de escala; bandas rojas = episodios con ultimo salto distinto del modal.</div>
<div class="card">__SVG__</div>
<div class="card"><b>Regimenes</b><br>__LEG__<br><b>Eventos de reglas</b><br>__LEGEV__</div>
<div class="card"><pre>__INFORME__</pre></div>
</body></html>
"""
    return (html.replace("__IDS__", esc(ids)).replace("__SVG__", svg)
            .replace("__LEG__", "".join(leg)).replace("__LEGEV__", "".join(leg_ev))
            .replace("__INFORME__", esc("\n".join(informe))))


# ============================================================================
# PRINCIPAL
# ============================================================================
def parse_arguments():
    p = argparse.ArgumentParser(
        description="HDP-HMM con bnpy para caracterizacion de RTT end-to-end en RIPE Atlas (v3)")
    p.add_argument("--crudo", required=True, help="CSV crudo de traceroute")
    p.add_argument("--categorias", required=True, help="CSV de categorias (ground truth de las reglas)")
    p.add_argument("--output-dir", default="resultados_hdphmm_v3")
    p.add_argument("--dest-hop", type=int, default=None, help="Hop fijo de destino")
    p.add_argument("--usar-hop-modal", action="store_true",
                   help="Usar solo el hop final mas frecuente (excluye ciclos sin respuesta del destino)")
    p.add_argument("--gap-minutes", type=float, default=90.0)
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--nlap", type=int, default=100)
    p.add_argument("--trans-alpha", type=float, default=0.5)
    p.add_argument("--start-alpha", type=float, default=10.0)
    p.add_argument("--hmm-kappa", type=float, default=1000.0)
    p.add_argument("--sF", type=float, default=None, help="Escala de covarianza del prior (default: varianza de la serie)")
    p.add_argument("--min-dwell", type=int, default=3)
    p.add_argument("--seed", type=int, default=None, help="Semilla (algseed/dataorderseed de bnpy); sin flag, defaults de bnpy")
    p.add_argument("--bnpy-outdir", default="./bnpy-output")
    p.add_argument("--alg-name", default="moVB")
    p.add_argument("--decodificacion", choices=["viterbi", "argmax"], default="viterbi")
    p.add_argument("--tolerancia-min", type=float, default=15.0, help="Tolerancia de coincidencia con eventos (min)")
    p.add_argument("--ventana-salto", type=int, default=4, help="Ciclos antes/despues para medir el salto de media")
    p.add_argument("--incluir-cambios-ruta", action="store_true",
                   help="Conservar en el ground truth cambio_ip_asn/cambio_longitud/cambio_destino")
    p.add_argument("--cv-estable", type=float, default=0.10, help="Desvio/media bajo el cual un regimen es 'estable'")
    p.add_argument("--n-permutaciones", type=int, default=2000)
    p.add_argument("--no-fusionar", action="store_true",
                   help="Desactivar la fusion automatica de estados equivalentes")
    p.add_argument("--dwell-barrido", default="1,2,3,5,8",
                   help="Valores de min-dwell del barrido de sensibilidad (se agrega el elegido)")
    p.add_argument("--min-episodio", type=int, default=3,
                   help="Ciclos minimos para contar un episodio de salto final no modal (default 3)")
    p.add_argument("--ventana-ruta-min", type=float, default=60.0,
                   help="Tolerancia (min) para asociar un episodio con un cambio de ruta")
    return p.parse_args()


def main():
    args = parse_arguments()
    print("=" * 74)
    print("HDP-HMM con bnpy (v3) - caracterizacion de RTT end-to-end, RIPE Atlas")
    print("=" * 74)
    print("Python: {0}".format(sys.version.split("\n")[0]))
    print("NumPy: {0}".format(np.__version__))
    mid, pid = extraer_ids_de_archivo(args.crudo)
    if mid and pid:
        print("Measurement ID: {0} | Probe ID: {1}".format(mid, pid))
    else:
        print("[ADVERTENCIA] No se pudo extraer measurement_id/probe_id del nombre del archivo.")
    print("=" * 74)

    # FASE 1
    print("[FASE 1] Cargando CSV crudo: {0}".format(args.crudo))
    por_ciclo, n_lineas = leer_ciclos(args.crudo)
    salto_cont, hop_modal = analizar_salto_final(por_ciclo)
    n_no_modal = sum(n for h, n in salto_cont.items() if h != hop_modal)
    dest_hop = args.dest_hop
    if args.usar_hop_modal and dest_hop is None:
        dest_hop = hop_modal
    serie = construir_serie(por_ciclo, dest_hop)
    print("  - Lineas: {0} | ciclos extraidos: {1}".format(n_lineas, len(serie)))
    if n_no_modal and not args.usar_hop_modal and args.dest_hop is None:
        print("  - [AVISO] {0} ciclos terminan en un salto distinto del hop {1}.".format(n_no_modal, hop_modal))
    print("[FASE 1] Cargando categorias: {0}".format(args.categorias))
    excluir = () if args.incluir_cambios_ruta else CATEGORIAS_RUTA
    dict_cat, cat_info = cargar_categorias(args.categorias, excluir)
    print("  - Timestamps con eventos tras filtrar: {0}".format(len(dict_cat)))
    if not serie:
        print("[ERROR] No se pudo extraer ninguna muestra de RTT.")
        sys.exit(1)

    # FASE 2
    secuencias = [s for s in cortar_en_secuencias(serie, args.gap_minutes) if len(s) >= 2]
    if not secuencias:
        print("[ERROR] No quedo ninguna secuencia utilizable.")
        sys.exit(1)
    serie_u = [x for s in secuencias for x in s]
    cad = cadencia_mediana_min(secuencias)
    rtt_all = np.array([x[1] for x in serie_u])
    sF = args.sF if args.sF is not None else float(rtt_all.var())
    print("[FASE 2] {0} ciclos -> {1} secuencia(s) (hueco > {2} min); cadencia {3:.1f} min; varianza RTT {4:.2f}; sF = {5:.2f}".format(
        len(serie_u), len(secuencias), args.gap_minutes, cad, rtt_all.var(), sF))

    # FASE 3
    modelo = correr_hdphmm(secuencias, args.K, args.nlap, args.trans_alpha, args.start_alpha,
                           args.hmm_kappa, sF, args.seed, args.bnpy_outdir, args.alg_name,
                           args.decodificacion)
    if modelo is None:
        print("[ERROR] No se pudo entrenar el modelo.")
        sys.exit(1)
    crudos = modelo["viterbi"] if modelo["usados"] == "viterbi" else modelo["argmax"]

    # FASE 4
    stats_crudo = estadisticas_por_estado(secuencias, crudos, cad)
    if args.no_fusionar:
        fusionados = [np.array(a, dtype=int) for a in crudos]
        stats_fus = stats_crudo
        mapa = dict((e, e) for e in stats_crudo)
        log_fusion = []
    else:
        fusionados, stats_fus, mapa, log_fusion = fusionar_equivalentes(secuencias, crudos, cad, args.cv_estable)
    print("[FASE 4] Fusion de estados equivalentes: {0} crudos -> {1} ({2} fusion(es))".format(
        len(stats_crudo), len(stats_fus), len(log_fusion)))
    print("[FASE 4] Suavizado post-hoc (--min-dwell {0})...".format(args.min_dwell))
    suaves = [np.array(suavizar_estados(a, args.min_dwell), dtype=int) for a in fusionados]
    cambios = {"argmax": contar_cambios(modelo["argmax"]),
               "viterbi": contar_cambios(modelo["viterbi"]) if modelo["viterbi"] is not None else None,
               "crudo": contar_cambios(crudos), "fusion": contar_cambios(fusionados),
               "suave": contar_cambios(suaves)}
    print("  - Cambios: crudos {0} -> fusionados {1} -> suavizados {2}".format(
        cambios["crudo"], cambios["fusion"], cambios["suave"]))
    stats_suave = estadisticas_por_estado(secuencias, suaves, cad)
    ranking = ranking_regimenes(stats_fus)
    for e in stats_crudo:
        ranking[e] = ranking[mapa[e]]
    for e in stats_suave:
        if e not in ranking:
            ranking[e] = len(ranking) + 1

    # FASE 5
    print("[FASE 5] Detectando change-points y midiendo el salto de regimen...")
    cps = detectar_change_points(secuencias, suaves)
    cps = enriquecer_change_points(cps, secuencias, stats_suave, ranking, args.ventana_salto, args.cv_estable)
    print("  - Change-points: {0}".format(len(cps)))

    # FASE 6
    print("[FASE 6] Correlacionando con las reglas (tolerancia +-{0:g} min, {1} permutaciones)...".format(
        args.tolerancia_min, args.n_permutaciones))
    corr = correlacionar(cps, dict_cat, secuencias, args.tolerancia_min, args.n_permutaciones)

    print("[FASE 6b] Sensibilidad al suavizado, patron horario y episodios de salto final...")
    try:
        valores = [int(x) for x in args.dwell_barrido.split(",") if x.strip()]
    except ValueError:
        valores = [1, 2, 3, 5, 8]
    barrido = barrido_min_dwell(secuencias, fusionados, dict_cat, args.tolerancia_min, valores, args.min_dwell)
    horario = analisis_horario(secuencias, suaves, stats_suave, args.cv_estable, args.n_permutaciones)
    episodios = episodios_salto_final(por_ciclo, hop_modal, secuencias, suaves, ranking, args.gap_minutes,
                                      cat_info.get("ts_excluidos", {}), 4, args.ventana_ruta_min,
                                      args.n_permutaciones, args.min_episodio)
    print("  - Episodios de salto final no modal: {0}".format(len(episodios["lista"])))

    # FASE 7
    residuos = analizar_residuos(corr["cps"])
    print("[FASE 7] Residuos: {0}".format(len(residuos)))

    # FASE 8
    print("[FASE 8] Exportando a: {0}".format(args.output_dir))
    ctx = {"args": args, "ids": (mid, pid), "serie": serie_u, "secuencias": secuencias, "cadencia": cad,
           "salto_cont": salto_cont, "hop_modal": hop_modal, "n_no_modal": n_no_modal,
           "dict_cat": dict_cat, "cat_info": cat_info, "sF": sF, "decod": modelo["usados"],
           "cambios": cambios, "stats_crudo": stats_crudo, "stats_suave": stats_suave,
           "ranking": ranking, "fusionados": fusionados, "stats_fus": stats_fus, "log_fusion": log_fusion,
           "barrido": barrido, "horario": horario, "episodios": episodios,
           "transiciones": transiciones_empiricas(cps), "cps": corr["cps"], "corr": corr,
           "residuos": residuos, "crudos": crudos, "suaves": suaves, "P": modelo["P"], "archivos": []}
    suf = "_measurement{0}_probe{1}".format(mid, pid) if (mid and pid) else ""
    informe = exportar(ctx, args.output_dir, suf)

    print("")
    for linea in informe:
        print(linea)
    print("EJECUCION COMPLETADA EXITOSAMENTE")


if __name__ == "__main__":
    main()
