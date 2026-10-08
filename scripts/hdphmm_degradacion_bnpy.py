# -*- coding: utf-8 -*-
"""
=============================================================================
Deteccion de DEGRADACION SOSTENIDA de RTT end-to-end (RIPE Atlas) con HDP-HMM
=============================================================================
Basado en: Mouchet, Vaton, Chonavel, Aben y den Hertog, "Large-Scale
           Characterization and Segmentation of Internet Path Delays with
           Infinite HMMs" (IEEE Access, 2020; arXiv:1910.12714).
           (bnpy es de Hughes & Sudderth; NO es el paper.)

Entorno: Python 2.7 (o 3) + bnpy + numpy. Solo numpy + biblioteca estandar.

Objetivo: dado el CSV crudo de traceroute de un par origen-destino, segmentar la
serie de RTT end-to-end en regimenes (HDP-HMM, Viterbi, suavizado) y reportar los
EPISODIOS DE DEGRADACION SOSTENIDA: periodos en que el RTT se mantiene por encima
de una linea base. No compara con Path Analysis ni usa ground truth.

Definicion de degradacion (el paper NO fija umbrales; es decision declarada):
  - Base: regimen dominante (mas ciclos) o el de mediana mas baja (--base).
  - Candidatos: corridas de ciclos consecutivos en regimenes con mediana por encima de la base.
  - Episodio: candidato que dura >= persistencia-min y cuya PROPIA mediana de RTT supera la
    base en >= max(factor-sigma x sigma_robusto_base, piso-ms).
  - Cada episodio se etiqueta segun el % de sus ciclos con ultimo salto distinto del modal:
    "camino" (>= no-modal-camino, probable cambio de ruta), "RTT" (<= no-modal-rtt, mismo camino)
    o "mixto" (entre ambos).
  - El informe incluye sensibilidad al piso (ms) y a la persistencia (min).
  - Regimenes con mediana a menos de piso-ms entre si se fusionan antes del suavizado.
  - Estadisticos robustos: sigma = 1.4826 x MAD; medianas (los atipicos no los inflan).

Salidas (CSV, columnas estables para un visor): estados_hdphmm, segmentos_hdphmm,
episodios_degradacion, regimenes, sensibilidad_umbral y resumen_ejecucion.txt.

USO:
    python hdphmm_degradacion_bnpy.py \\
        --crudo historial_traceroute_measurement126502326_probe64883.csv \\
        --output-dir resultados_hdphmm_deg \\
        --gap-minutes 90 --K 15 --nlap 100 --sF 10 \\
        --piso-ms 2 --factor-sigma 3 --persistencia-min 30
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import argparse
import csv
import re
import io
from collections import defaultdict, Counter
from datetime import datetime

try:
    import numpy as np
except ImportError:
    sys.stderr.write("Este script necesita numpy instalado.\n")
    sys.exit(1)

STD_ROB_MIN = 0.01  # piso del desvio robusto (ms)


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


# ============================================================================
# FASE 1-2: CARGA Y SECUENCIAS
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
# FASE 3: HDP-HMM CON BNPY + VITERBI
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
# SUAVIZADO Y ESTADISTICAS ROBUSTAS
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


def construir_serie_modal(por_ciclo, hop_modal):
    """Solo ciclos cuyo ultimo salto con respuesta es el modal (RTT end-to-end por el camino habitual)."""
    serie = []
    for ts in sorted(por_ciclo.keys()):
        hops = por_ciclo[ts]
        if hops and max(hops.keys()) == hop_modal:
            serie.append((ts, hops[hop_modal]))
    return serie


# ============================================================================
# FASE 4: FUSION POR PISO Y REGIMENES
# ============================================================================
def fusionar_por_piso(secuencias, estados, cadencia_min, piso_ms):
    """
    Fusion aglomerativa: une de a un par (el de menor diferencia de medianas) mientras
    exista algun par con |dif. de medianas| < piso_ms; el menos poblado se absorbe en el
    mas poblado y se recalcula. Devuelve (estados, stats, mapa {estado: representante}, log).
    """
    est = [np.array(a, dtype=int) for a in estados]
    stats = estadisticas_por_estado(secuencias, est, cadencia_min)
    mapa = dict((e, e) for e in stats)
    log = []
    for _ in range(len(stats)):
        ids = sorted(stats.keys())
        mejor = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = abs(stats[ids[i]]["mediana"] - stats[ids[j]]["mediana"])
                if d < piso_ms and (mejor is None or d < mejor[0]):
                    mejor = (d, ids[i], ids[j])
        if mejor is None:
            break
        d, i, j = mejor
        dest, absorbido = (i, j) if stats[i]["n"] >= stats[j]["n"] else (j, i)
        log.append({"absorbido": absorbido, "destino": dest, "dif_ms": d,
                    "n_abs": stats[absorbido]["n"], "n_dest": stats[dest]["n"],
                    "med_abs": stats[absorbido]["mediana"], "med_dest": stats[dest]["mediana"]})
        for k in mapa:
            if mapa[k] == absorbido:
                mapa[k] = dest
        est = [np.array([dest if int(x) == absorbido else int(x) for x in a], dtype=int) for a in est]
        stats = estadisticas_por_estado(secuencias, est, cadencia_min)
    return est, stats, mapa, log


def ranking_por_mediana(stats):
    """{estado: rank 1..n}, R1 = mediana mas baja."""
    orden = sorted(stats.keys(), key=lambda e: (stats[e]["mediana"], e))
    return dict((e, i + 1) for i, e in enumerate(orden))


def elegir_base(stats, modo):
    """Estado de referencia: 'dominante' (mas ciclos) o 'minimo' (mediana mas baja con >= 1% de ciclos)."""
    if modo == "minimo":
        cand = [e for e in stats if stats[e]["pct"] >= 1.0] or list(stats.keys())
        return min(cand, key=lambda e: (stats[e]["mediana"], e))
    return max(stats.keys(), key=lambda e: (stats[e]["n"], -e))


# ============================================================================
# FASE 5: EPISODIOS DE DEGRADACION SOSTENIDA
# ============================================================================
def detectar_episodios(secuencias, suaves, stats, ranking, base_med, umbral_ms, persist_min,
                       cad, no_modal_ts, lim_rtt, lim_camino):
    """
    Candidato = corrida de ciclos consecutivos en regimenes con mediana por encima de la base.
    Episodio = candidato de al menos persist_min minutos cuya PROPIA mediana de RTT supera la
    base en >= umbral_ms. tipo segun el % de ciclos con ultimo salto no modal: "camino" si
    >= lim_camino, "RTT" si <= lim_rtt, "mixto" en el rango intermedio.
    """
    cand = dict((e, s_["mediana"] > base_med) for e, s_ in stats.items())
    eps = []
    for sid, (seq, est) in enumerate(zip(secuencias, suaves)):
        r = np.array([x[1] for x in seq])
        flags = [cand[int(e)] for e in est]
        n = len(flags)
        i = 0
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            dur = (parse_ts(seq[j][0]) - parse_ts(seq[i][0])).total_seconds() / 60.0 + cad
            tramo = r[i:j + 1]
            med = float(np.median(tramo))
            if dur >= persist_min and (med - base_med) >= umbral_ms:
                regs = []
                for e in est[i:j + 1]:
                    rg = "R{0}".format(ranking[int(e)])
                    if not regs or regs[-1] != rg:
                        regs.append(rg)
                pico = max(stats[int(e)]["mediana"] for e in set(est[i:j + 1])) - base_med
                antes = r[max(0, i - 4):i]
                despues = r[i:min(j + 1, i + 4)]
                nm = sum(1 for k in range(i, j + 1) if seq[k][0] in no_modal_ts)
                pct_nm = 100.0 * nm / (j - i + 1)
                eps.append({"seq_id": sid, "i": i, "j": j, "inicio": seq[i][0], "fin": seq[j][0],
                            "n_ciclos": j - i + 1, "duracion_min": dur, "mediana_ms": med,
                            "magnitud_ms": med - base_med,
                            "pct": 100.0 * (med - base_med) / max(base_med, 1e-9),
                            "pico_ms": pico, "regimenes": ">".join(regs),
                            "recuperado": j < n - 1, "inicio_censurado": i == 0,
                            "salto_inicio_ms": (float(np.median(despues)) - float(np.median(antes)))
                            if len(antes) else None,
                            "pct_no_modal": pct_nm,
                            "tipo": ("camino" if pct_nm >= lim_camino
                                     else ("RTT" if pct_nm <= lim_rtt else "mixto"))})
            i = j + 1
    for k, e in enumerate(eps):
        e["id"] = k + 1
    return eps


def _conteos(eps, total):
    return {"n": len(eps),
            "n_rtt": sum(1 for e in eps if e["tipo"] == "RTT"),
            "n_mixto": sum(1 for e in eps if e["tipo"] == "mixto"),
            "n_camino": sum(1 for e in eps if e["tipo"] == "camino"),
            "minutos": sum(e["duracion_min"] for e in eps),
            "pct_tiempo": 100.0 * sum(e["n_ciclos"] for e in eps) / max(total, 1.0),
            "mag_mediana": float(np.median([e["magnitud_ms"] for e in eps])) if eps else float("nan")}


def sensibilidad_umbral(secuencias, suaves, stats, ranking, base_med, sig_base, factor, pisos,
                        elegido, persist_min, cad, no_modal_ts, lim_rtt, lim_camino):
    total = float(sum(len(s_) for s_ in secuencias))
    filas = []
    for piso in sorted(set(list(pisos) + [elegido])):
        umbral = max(factor * sig_base, piso)
        eps = detectar_episodios(secuencias, suaves, stats, ranking, base_med, umbral, persist_min,
                                 cad, no_modal_ts, lim_rtt, lim_camino)
        f_ = _conteos(eps, total)
        f_.update({"piso": piso, "umbral": umbral, "elegido": piso == elegido})
        filas.append(f_)
    return filas


def sensibilidad_persistencia(secuencias, suaves, stats, ranking, base_med, umbral, persistencias,
                              elegida, cad, no_modal_ts, lim_rtt, lim_camino):
    total = float(sum(len(s_) for s_ in secuencias))
    filas = []
    for pm in sorted(set(list(persistencias) + [elegida])):
        eps = detectar_episodios(secuencias, suaves, stats, ranking, base_med, umbral, pm,
                                 cad, no_modal_ts, lim_rtt, lim_camino)
        f_ = _conteos(eps, total)
        f_.update({"persistencia": pm, "elegido": pm == elegida})
        filas.append(f_)
    return filas


# ============================================================================
# INFORME (consola + resumen_ejecucion.txt identicos)
# ============================================================================
def movimientos(antes, despues):
    """Counter {(estado_antes, estado_despues): ciclos} de los ciclos que cambiaron de estado."""
    c = Counter()
    for a_, b_ in zip(antes, despues):
        for x, y in zip(a_, b_):
            if int(x) != int(y):
                c[(int(x), int(y))] += 1
    return c


def tabla_etapa(stats, titulo, extra_titulo, extra):
    """Tabla de estados de una etapa. extra(e) devuelve el texto de la ultima columna."""
    L = [titulo,
         "  {0:>7}{1:>8}{2:>7}{3:>10}{4:>9}{5:>7}{6:>13}  {7}".format(
             "Estado", "Ciclos", "%", "Mediana", "Sig.rob", "Apar.", "Perm.med(c)", extra_titulo)]
    for e in sorted(stats.keys(), key=lambda k: (stats[k]["mediana"], k)):
        x = stats[e]
        L.append("  {0:>7}{1:>8}{2:>7}{3:>10}{4:>9}{5:>7}{6:>13}  {7}".format(
            e, x["n"], "{0:.1f}".format(x["pct"]), f2(x["mediana"]), f2(x["std_rob"]),
            x["apariciones"], "{0:.0f}".format(x["perm_med_ciclos"]), extra(e)))
    return L


def construir_informe(ctx):
    a = ctx["args"]
    st = ctx["stats_suave"]
    rk = ctx["ranking"]
    L = []
    L.append("=" * 74)
    L.append("DEGRADACION SOSTENIDA DE RTT (HDP-HMM, bnpy)")
    L.append("=" * 74)
    if ctx["ids"][0]:
        L.append("Measurement {0} | Probe {1}".format(*ctx["ids"]))
    L.append("Fecha de ejecucion: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    L.append("")
    L.append("1. DATOS")
    s = ctx["serie"]
    r = np.array([x[1] for x in s])
    L.append("  - {0} ciclos, {1} secuencia(s) (hueco > {2:g} min), cadencia mediana {3:.1f} min".format(
        len(s), len(ctx["secuencias"]), a.gap_minutes, ctx["cadencia"]))
    L.append("  - Rango: {0} a {1}".format(s[0][0], s[-1][0]))
    L.append("  - RTT: min {0:.2f} | mediana {1:.2f} | media {2:.2f} | max {3:.2f} ms".format(
        r.min(), np.median(r), r.mean(), r.max()))
    tot = max(sum(ctx["salto_cont"].values()), 1)
    L.append("  - Ultimo salto con respuesta por ciclo: {0}".format(
        ", ".join("hop {0}: {1} ({2:.1f}%)".format(h, n, 100.0 * n / tot)
                  for h, n in ctx["salto_cont"].most_common(4))))
    if ctx["n_no_modal"] > 0 and not a.usar_hop_modal:
        L.append("  [AVISO] {0} ciclos terminan en un salto distinto del habitual (hop {1}): su RTT puede "
                 "reflejar otro camino. Cada episodio informa el % de esos ciclos y se etiqueta camino/RTT; "
                 "--usar-hop-modal los excluye de la serie.".format(
                     ctx["n_no_modal"], ctx["hop_modal"]))
    elif a.usar_hop_modal:
        L.append("  - --usar-hop-modal: solo ciclos cuyo ultimo salto es el hop {0} (excluidos: {1}); "
                 "los episodios de camino quedan fuera por construccion.".format(
                     ctx["hop_modal"], ctx["n_no_modal"]))

    L.append("")
    L.append("2. MODELO")
    L.append("  - K truncado = {0}; sF = {1:.2f}; hmmKappa = {2}; decodificacion = {3}".format(
        a.K, ctx["sF"], a.hmm_kappa, ctx["decod"]))
    L.append("  - Estados activos: {0} crudos -> {1} tras fusion (piso {2:g} ms) -> {3} tras suavizado "
             "(--min-dwell {4})".format(ctx["n_crudos"], len(ctx["stats_fus"]), a.piso_ms, len(st), a.min_dwell))
    L.append("  - Cambios de estado: crudos {0} -> suavizados {1}".format(
        ctx["cambios_crudo"], ctx["cambios_suave"]))
    for m in ctx["log_fusion"]:
        L.append("    fusion: estado {0} ({1} ciclos, mediana {2:.2f}) -> estado {3} ({4} ciclos, "
                 "mediana {5:.2f}); diferencia {6:.2f} ms".format(
                     m["absorbido"], m["n_abs"], m["med_abs"], m["destino"], m["n_dest"],
                     m["med_dest"], m["dif_ms"]))

    L.append("")
    L.append("3. ESTADOS: DE CRUDOS A REGIMENES (ordenados por mediana; sig.rob = 1.4826*MAD)")
    total = float(sum(len(q) for q in ctx["secuencias"]))
    sc, sf = ctx["stats_crudo"], ctx["stats_fus"]
    mapa = ctx["mapa_fusion"]
    L += tabla_etapa(sc, "  a) Estados crudos de Viterbi ({0} estados, {1:.0f} ciclos):".format(len(sc), total),
                     "Fusion",
                     lambda e: ("-> estado {0}".format(mapa[e]) if mapa[e] != e else "se mantiene"))
    n_fus = sum(sc[e]["n"] for e in sc if mapa[e] != e)
    L.append("  Ciclos reasignados por la fusion: {0} ({1:.1f}%)".format(n_fus, 100.0 * n_fus / max(total, 1)))
    L.append("")
    mov = ctx["mov_suavizado"]
    destinos = defaultdict(list)
    for (x, y), n in mov.items():
        destinos[x].append((n, y))

    def txt_suave(e):
        if e not in st:
            partes = ", ".join("{0} a estado {1}".format(n, y) for n, y in sorted(destinos[e], reverse=True))
            return "DESAPARECE (ciclos: {0})".format(partes if partes else "ninguno")
        d = st[e]["n"] - sf[e]["n"]
        return "{0} ciclos ({1:+d})".format(st[e]["n"], d)

    L += tabla_etapa(sf, "  b) Tras la fusion ({0} estados):".format(len(sf)), "Tras suavizado", txt_suave)
    n_mov = sum(mov.values())
    L.append("  Ciclos reasignados por el suavizado: {0} ({1:.1f}%)".format(n_mov, 100.0 * n_mov / max(total, 1)))
    for (x, y), n in sorted(mov.items(), key=lambda kv: -kv[1])[:6]:
        L.append("    estado {0} -> estado {1}: {2} ciclos".format(x, y, n))
    L.append("")
    L.append("  c) Tras el suavizado: regimenes (R1 = mediana mas baja)")
    L.append("  {0:<5}{1:>7}{2:>8}{3:>7}{4:>10}{5:>9}{6:>11}{7:>7}{8:>16}  {9}".format(
        "Reg", "Estado", "Ciclos", "%", "Mediana", "Sig.rob", "Dif.base", "Apar.", "Perm.med(c/min)", "Rol"))
    for e in sorted(st.keys(), key=lambda k: rk[k]):
        x = st[e]
        if e == ctx["base_estado"]:
            rol = "BASE"
        elif e in ctx["estados_sobre_base"]:
            rol = "sobre base"
        else:
            rol = "-"
        L.append("  R{0:<4}{1:>7}{2:>8}{3:>7}{4:>10}{5:>9}{6:>11}{7:>7}{8:>16}  {9}".format(
            rk[e], e, x["n"], "{0:.1f}".format(x["pct"]), f2(x["mediana"]), f2(x["std_rob"]),
            "{0:+.2f}".format(x["mediana"] - ctx["base_med"]), x["apariciones"],
            "{0:.0f}/{1:.0f}".format(x["perm_med_ciclos"], x["perm_med_min"]), rol))

    L.append("")
    L.append("4. CRITERIO DE DEGRADACION")
    L.append("  - Base ({0}): R{1}, mediana {2:.2f} ms, sig.rob {3:.2f} ms".format(
        a.base, rk[ctx["base_estado"]], ctx["base_med"], ctx["sig_base"]))
    L.append("  - Umbral = max({0:g} x sig.rob, {1:g} ms) = {2:.2f} ms sobre la base".format(
        a.factor_sigma, a.piso_ms, ctx["umbral"]))
    L.append("  - Episodio = corrida de ciclos consecutivos en regimenes por encima de la base, de al menos "
             "{0:g} min y con mediana PROPIA de RTT >= base + umbral.".format(a.persistencia_min))
    L.append("  - Tipo segun % de ciclos con ultimo salto no modal: camino >= {0:g}%, RTT <= {1:g}%, "
             "mixto en el medio.".format(a.no_modal_camino, a.no_modal_rtt))
    L.append("  - Estos parametros no vienen del paper (no fija umbrales): son decision declarada del analisis.")

    eps = ctx["episodios"]
    L.append("")
    L.append("5. EPISODIOS DE DEGRADACION SOSTENIDA ({0})".format(len(eps)))
    if not eps:
        L.append("  Ningun episodio supera el umbral y la persistencia minima.")
    else:
        tot_min = sum(e["duracion_min"] for e in eps)
        tot_ciclos = float(sum(len(q) for q in ctx["secuencias"]))
        L.append("  Tiempo degradado acumulado: {0:.1f} h ({1:.1f}% de los ciclos)".format(
            tot_min / 60.0, 100.0 * sum(e["n_ciclos"] for e in eps) / max(tot_ciclos, 1)))
        for t_ in ("RTT", "mixto", "camino"):
            sel = [e for e in eps if e["tipo"] == t_]
            L.append("    - {0:<7}: {1} episodios, {2:.1f} h".format(
                t_, len(sel), sum(e["duracion_min"] for e in sel) / 60.0))
        L.append("  {0:>3} {1:<20}{2:>8}{3:>9}{4:>9}{5:>8}{6:>10}  {7:<8}{8:<10}{9}".format(
            "#", "inicio", "dur(h)", "mag(ms)", "mag(%)", "pico", "no modal", "tipo", "regimenes", "recuperado"))
        for e in eps[:20]:
            L.append("  {0:>3} {1:<20}{2:>8.1f}{3:>9.2f}{4:>9.1f}{5:>8.2f}{6:>10}  {7:<8}{8:<10}{9}".format(
                e["id"], e["inicio"], e["duracion_min"] / 60.0, e["magnitud_ms"], e["pct"], e["pico_ms"],
                "{0:.0f}%".format(e["pct_no_modal"]), e["tipo"], e["regimenes"],
                "si" if e["recuperado"] else "no (censurado)"))
        if len(eps) > 20:
            L.append("  ... ({0} episodios mas en episodios_degradacion_*.csv)".format(len(eps) - 20))
        L.append("  mag = mediana del RTT durante el episodio - base (es la que se compara con el umbral); "
                 "pico = mediana del regimen mas alto - base; no modal = % de ciclos del episodio "
                 "cuyo ultimo salto no es el habitual.")

    L.append("")
    L.append("6. SENSIBILIDAD AL PISO (estados fijos; solo cambia el criterio de deteccion por episodio)")
    L.append("  {0:>9}{1:>11}{2:>11}{3:>7}{4:>7}{5:>8}{6:>9}{7:>10}{8:>14}".format(
        "piso(ms)", "umbral(ms)", "episodios", "RTT", "mixto", "camino", "horas", "% tiempo", "mag.med(ms)"))
    for f_ in ctx["sensibilidad"]:
        L.append("  {0:>9g}{1:>11.2f}{2:>11}{3:>7}{4:>7}{5:>8}{6:>9.1f}{7:>10.1f}{8:>14}{9}".format(
            f_["piso"], f_["umbral"], f_["n"], f_["n_rtt"], f_["n_mixto"], f_["n_camino"],
            f_["minutos"] / 60.0, f_["pct_tiempo"],
            "-" if f_["mag_mediana"] != f_["mag_mediana"] else "{0:.2f}".format(f_["mag_mediana"]),
            "   <- elegido" if f_["elegido"] else ""))

    L.append("")
    L.append("7. SENSIBILIDAD A LA PERSISTENCIA (piso fijo en {0:g} ms; umbral {1:.2f} ms)".format(
        a.piso_ms, ctx["umbral"]))
    L.append("  {0:>12}{1:>11}{2:>7}{3:>7}{4:>8}{5:>9}{6:>10}".format(
        "persist.(min)", "episodios", "RTT", "mixto", "camino", "horas", "% tiempo"))
    for f_ in ctx["sens_persistencia"]:
        L.append("  {0:>12g}{1:>11}{2:>7}{3:>7}{4:>8}{5:>9.1f}{6:>10.1f}{7}".format(
            f_["persistencia"], f_["n"], f_["n_rtt"], f_["n_mixto"], f_["n_camino"], f_["minutos"] / 60.0,
            f_["pct_tiempo"], "   <- elegida" if f_["elegido"] else ""))
    L.append("  Lectura: un episodio de tipo RTT que desaparece al subir la persistencia es una excursion breve, "
             "no una degradacion sostenida.")

    L.append("")
    L.append("8. ARCHIVOS")
    for p in ctx["archivos"]:
        L.append("  - {0}".format(p))
    L.append("=" * 74)
    return L


# ============================================================================
# EXPORTACION
# ============================================================================
def exportar(ctx, output_dir, suf):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rk = ctx["ranking"]
    base_med = ctx["base_med"]
    archivos = []

    def ruta(nombre):
        base, ext = os.path.splitext(nombre)
        p = os.path.join(output_dir, base + suf + ext)
        archivos.append(p)
        return p

    # episodio por (seq_id, indice de ciclo)
    ep_de = {}
    for e in ctx["episodios"]:
        for k in range(e["i"], e["j"] + 1):
            ep_de[(e["seq_id"], k)] = e["id"]

    # 1. estados por ciclo
    with abrir_csv_w(ruta("estados_hdphmm.csv")) as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "rtt_ms", "seq_id", "state_raw", "state", "regimen", "degradado", "episodio_id"])
        for sid, (seq, cr, su) in enumerate(zip(ctx["secuencias"], ctx["crudos"], ctx["suaves"])):
            for k, ((ts, rtt), ec, es_) in enumerate(zip(seq, cr, su)):
                eid = ep_de.get((sid, k), "")
                w.writerow([ts, "{0:.3f}".format(rtt), sid, int(ec), int(es_), "R{0}".format(rk[int(es_)]),
                            1 if eid != "" else 0, eid])

    # 2. segmentos (corridas de un mismo regimen)
    with abrir_csv_w(ruta("segmentos_hdphmm.csv")) as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "inicio", "fin", "n_ciclos", "duracion_min", "estado", "regimen", "mediana_ms",
                    "media_ms", "std_rob_ms", "dif_base_ms", "sobre_base", "episodio_id"])
        for sid, (seq, su) in enumerate(zip(ctx["secuencias"], ctx["suaves"])):
            r = np.array([x[1] for x in seq])
            for ini, fin, val in runs_de(su):
                tr = r[ini:fin]
                med = float(np.median(tr))
                mad = float(np.median(np.abs(tr - med)))
                dur = (parse_ts(seq[fin - 1][0]) - parse_ts(seq[ini][0])).total_seconds() / 60.0 + ctx["cadencia"]
                w.writerow([sid, seq[ini][0], seq[fin - 1][0], fin - ini, "{0:.1f}".format(dur), val,
                            "R{0}".format(rk[val]), "{0:.3f}".format(med), "{0:.3f}".format(float(tr.mean())),
                            "{0:.3f}".format(max(1.4826 * mad, STD_ROB_MIN)), "{0:.3f}".format(med - base_med),
                            1 if val in ctx["estados_sobre_base"] else 0, ep_de.get((sid, ini), "")])

    # 3. episodios
    with abrir_csv_w(ruta("episodios_degradacion.csv")) as f:
        w = csv.writer(f)
        w.writerow(["episodio_id", "seq_id", "inicio", "fin", "n_ciclos", "duracion_min", "mediana_ms",
                    "base_ms", "magnitud_ms", "magnitud_pct", "pico_ms", "regimenes", "salto_inicio_ms",
                    "pct_ciclos_no_modal", "tipo", "recuperado", "inicio_censurado"])
        for e in ctx["episodios"]:
            sj = e["salto_inicio_ms"]
            w.writerow([e["id"], e["seq_id"], e["inicio"], e["fin"], e["n_ciclos"], "{0:.1f}".format(e["duracion_min"]),
                        "{0:.3f}".format(e["mediana_ms"]), "{0:.3f}".format(base_med),
                        "{0:.3f}".format(e["magnitud_ms"]), "{0:.2f}".format(e["pct"]), "{0:.3f}".format(e["pico_ms"]),
                        e["regimenes"], "" if sj is None else "{0:.3f}".format(sj),
                        "{0:.1f}".format(e["pct_no_modal"]), e["tipo"], e["recuperado"], e["inicio_censurado"]])

    # 4. regimenes
    with abrir_csv_w(ruta("regimenes.csv")) as f:
        w = csv.writer(f)
        w.writerow(["regimen", "estado", "ciclos", "pct", "mediana_ms", "std_rob_ms", "dif_base_ms",
                    "apariciones", "permanencia_mediana_ciclos", "permanencia_mediana_min", "rol"])
        for e in sorted(ctx["stats_suave"].keys(), key=lambda k: rk[k]):
            x = ctx["stats_suave"][e]
            rol = "base" if e == ctx["base_estado"] else ("sobre_base" if e in ctx["estados_sobre_base"] else "")
            w.writerow(["R{0}".format(rk[e]), e, x["n"], "{0:.2f}".format(x["pct"]), "{0:.3f}".format(x["mediana"]),
                        "{0:.3f}".format(x["std_rob"]), "{0:.3f}".format(x["mediana"] - base_med),
                        x["apariciones"], "{0:.1f}".format(x["perm_med_ciclos"]),
                        "{0:.1f}".format(x["perm_med_min"]), rol])

    # 4b. estados por etapa (crudo, fusionado, suavizado)
    with abrir_csv_w(ruta("estados_etapas.csv")) as f:
        w = csv.writer(f)
        w.writerow(["etapa", "estado", "ciclos", "pct", "mediana_ms", "std_rob_ms", "apariciones",
                    "permanencia_mediana_ciclos", "destino_fusion", "ciclos_tras_suavizado"])
        for etapa, st_ in (("crudo", ctx["stats_crudo"]), ("fusionado", ctx["stats_fus"]),
                           ("suavizado", ctx["stats_suave"])):
            for e in sorted(st_.keys(), key=lambda k: (st_[k]["mediana"], k)):
                x = st_[e]
                dest = ctx["mapa_fusion"][e] if etapa == "crudo" else ""
                tras = ""
                if etapa == "fusionado":
                    tras = ctx["stats_suave"][e]["n"] if e in ctx["stats_suave"] else 0
                w.writerow([etapa, e, x["n"], "{0:.2f}".format(x["pct"]), "{0:.3f}".format(x["mediana"]),
                            "{0:.3f}".format(x["std_rob"]), x["apariciones"],
                            "{0:.1f}".format(x["perm_med_ciclos"]), dest, tras])

    # 5. sensibilidad
    with abrir_csv_w(ruta("sensibilidad_umbral.csv")) as f:
        w = csv.writer(f)
        w.writerow(["piso_ms", "umbral_ms", "episodios", "episodios_rtt", "episodios_mixto", "episodios_camino",
                    "horas", "pct_tiempo", "magnitud_mediana_ms", "elegido"])
        for x in ctx["sensibilidad"]:
            w.writerow(["{0:g}".format(x["piso"]), "{0:.3f}".format(x["umbral"]), x["n"], x["n_rtt"],
                        x["n_mixto"], x["n_camino"],
                        "{0:.2f}".format(x["minutos"] / 60.0), "{0:.2f}".format(x["pct_tiempo"]),
                        "" if x["mag_mediana"] != x["mag_mediana"] else "{0:.3f}".format(x["mag_mediana"]),
                        x["elegido"]])

    with abrir_csv_w(ruta("sensibilidad_persistencia.csv")) as f:
        w = csv.writer(f)
        w.writerow(["persistencia_min", "episodios", "episodios_rtt", "episodios_mixto", "episodios_camino",
                    "horas", "pct_tiempo", "elegido"])
        for x in ctx["sens_persistencia"]:
            w.writerow(["{0:g}".format(x["persistencia"]), x["n"], x["n_rtt"], x["n_mixto"], x["n_camino"],
                        "{0:.2f}".format(x["minutos"] / 60.0), "{0:.2f}".format(x["pct_tiempo"]), x["elegido"]])

    resumen_path = ruta("resumen_ejecucion.txt")
    ctx["archivos"] = archivos
    informe = construir_informe(ctx)
    escribir_texto(resumen_path, "\n".join(informe) + "\n")
    return informe


# ============================================================================
# PRINCIPAL
# ============================================================================
def parse_arguments():
    p = argparse.ArgumentParser(
        description="Deteccion de degradacion sostenida de RTT end-to-end (RIPE Atlas) con HDP-HMM (bnpy)")
    p.add_argument("--crudo", required=True, help="CSV crudo de traceroute (extract_traceroute_v2.py)")
    p.add_argument("--output-dir", default="resultados_hdphmm_deg")
    p.add_argument("--dest-hop", type=int, default=None, help="Hop fijo de destino")
    p.add_argument("--usar-hop-modal", action="store_true",
                   help="Usar solo los ciclos cuyo ultimo salto con respuesta es el hop final mas frecuente "
                        "(excluye los demas de la serie; ver --dest-hop para fijar un hop)")
    p.add_argument("--gap-minutes", type=float, default=90.0)
    p.add_argument("--K", type=int, default=15, help="Truncamiento del HDP (maximo de estados; el modelo decide cuantos usa)")
    p.add_argument("--nlap", type=int, default=100)
    p.add_argument("--trans-alpha", type=float, default=0.5)
    p.add_argument("--start-alpha", type=float, default=10.0)
    p.add_argument("--hmm-kappa", type=float, default=1000.0)
    p.add_argument("--sF", type=float, default=None, help="Escala de covarianza del prior (default: varianza de la serie)")
    p.add_argument("--min-dwell", type=int, default=3, help="Ciclos minimos de permanencia en un estado (suavizado)")
    p.add_argument("--seed", type=int, default=None, help="Semilla (algseed/dataorderseed de bnpy)")
    p.add_argument("--bnpy-outdir", default="./bnpy-output")
    p.add_argument("--alg-name", default="moVB")
    p.add_argument("--piso-ms", type=float, default=2.0,
                   help="Diferencia minima de medianas (ms): fusiona regimenes mas cercanos y es el piso del umbral")
    p.add_argument("--factor-sigma", type=float, default=3.0,
                   help="El umbral de degradacion es max(factor x sigma robusto de la base, piso)")
    p.add_argument("--persistencia-min", type=float, default=30.0,
                   help="Duracion minima (min) de un episodio de degradacion")
    p.add_argument("--no-modal-camino", type=float, default=80.0,
                   help="%% de ciclos con ultimo salto no modal desde el cual un episodio es de 'camino'")
    p.add_argument("--no-modal-rtt", type=float, default=20.0,
                   help="%% de ciclos con ultimo salto no modal hasta el cual un episodio es de 'RTT' (mismo camino)")
    p.add_argument("--persistencias-sensibilidad", default="30,60,120,180",
                   help="Persistencias (min) de la tabla de sensibilidad (se agrega la elegida)")
    p.add_argument("--base", choices=["dominante", "minimo"], default="dominante",
                   help="Regimen de referencia: el de mas ciclos o el de mediana mas baja")
    p.add_argument("--pisos-sensibilidad", default="1,2,5,10",
                   help="Pisos (ms) de la tabla de sensibilidad (se agrega el elegido)")
    return p.parse_args()


def main():
    args = parse_arguments()
    print("=" * 74)
    print("Degradacion sostenida de RTT - HDP-HMM con bnpy")
    print("=" * 74)
    print("Python: {0}".format(sys.version.split("\n")[0]))
    print("NumPy: {0}".format(np.__version__))
    mid, pid = extraer_ids_de_archivo(args.crudo)
    if mid and pid:
        print("Measurement ID: {0} | Probe ID: {1}".format(mid, pid))
    print("=" * 74)

    # FASE 1
    print("[FASE 1] Cargando CSV crudo: {0}".format(args.crudo))
    por_ciclo, n_lineas = leer_ciclos(args.crudo)
    salto_cont, hop_modal = analizar_salto_final(por_ciclo)
    n_no_modal = sum(n for h, n in salto_cont.items() if h != hop_modal)
    no_modal_ts = set(ts for ts, h in por_ciclo.items() if h and max(h.keys()) != hop_modal)
    dest_hop = args.dest_hop
    if args.usar_hop_modal and args.dest_hop is None:
        serie = construir_serie_modal(por_ciclo, hop_modal)
    else:
        serie = construir_serie(por_ciclo, dest_hop)
    print("  - Lineas: {0} | ciclos extraidos: {1}".format(n_lineas, len(serie)))
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
    print("[FASE 2] {0} ciclos -> {1} secuencia(s); cadencia {2:.1f} min; sF = {3:.2f}".format(
        len(serie_u), len(secuencias), cad, sF))

    # FASE 3
    modelo = correr_hdphmm(secuencias, args.K, args.nlap, args.trans_alpha, args.start_alpha,
                           args.hmm_kappa, sF, args.seed, args.bnpy_outdir, args.alg_name, "viterbi")
    if modelo is None:
        print("[ERROR] No se pudo entrenar el modelo.")
        sys.exit(1)
    crudos = modelo["viterbi"] if modelo["usados"] == "viterbi" else modelo["argmax"]

    # FASE 4
    stats_crudo = estadisticas_por_estado(secuencias, crudos, cad)
    fusionados, stats_fus, mapa, log_fusion = fusionar_por_piso(secuencias, crudos, cad, args.piso_ms)
    print("[FASE 4] Fusion (piso {0:g} ms): {1} estados -> {2}".format(
        args.piso_ms, len(stats_crudo), len(stats_fus)))
    suaves = [np.array(suavizar_estados(a, args.min_dwell), dtype=int) for a in fusionados]
    stats_suave = estadisticas_por_estado(secuencias, suaves, cad)
    ranking = ranking_por_mediana(stats_suave)
    for e in stats_crudo:
        if e not in ranking:
            ranking[e] = ranking.get(mapa.get(e, e), len(ranking) + 1)
    print("  - Suavizado (--min-dwell {0}): {1} regimenes".format(args.min_dwell, len(stats_suave)))

    # FASE 5
    base_estado = elegir_base(stats_suave, args.base)
    base_med = stats_suave[base_estado]["mediana"]
    sig_base = stats_suave[base_estado]["std_rob"]
    umbral = max(args.factor_sigma * sig_base, args.piso_ms)
    sobre_base = set(e for e, s in stats_suave.items() if s["mediana"] > base_med)
    episodios = detectar_episodios(secuencias, suaves, stats_suave, ranking, base_med, umbral,
                                   args.persistencia_min, cad, no_modal_ts, args.no_modal_rtt, args.no_modal_camino)
    print("[FASE 5] Base R{0} = {1:.2f} ms; umbral {2:.2f} ms; episodios: {3}".format(
        ranking[base_estado], base_med, umbral, len(episodios)))
    try:
        pisos = [float(x) for x in args.pisos_sensibilidad.split(",") if x.strip()]
    except ValueError:
        pisos = [1.0, 2.0, 5.0, 10.0]
    sens = sensibilidad_umbral(secuencias, suaves, stats_suave, ranking, base_med, sig_base,
                               args.factor_sigma, pisos, args.piso_ms, args.persistencia_min, cad, no_modal_ts,
                               args.no_modal_rtt, args.no_modal_camino)
    try:
        persists = [float(x) for x in args.persistencias_sensibilidad.split(",") if x.strip()]
    except ValueError:
        persists = [30.0, 60.0, 120.0, 180.0]
    sens_p = sensibilidad_persistencia(secuencias, suaves, stats_suave, ranking, base_med, umbral, persists,
                                       args.persistencia_min, cad, no_modal_ts, args.no_modal_rtt,
                                       args.no_modal_camino)

    # EXPORTACION
    print("[FASE 6] Exportando a: {0}".format(args.output_dir))
    ctx = {"args": args, "ids": (mid, pid), "serie": serie_u, "secuencias": secuencias, "cadencia": cad,
           "salto_cont": salto_cont, "hop_modal": hop_modal, "n_no_modal": n_no_modal, "sF": sF,
           "decod": modelo["usados"], "n_crudos": len(stats_crudo), "stats_fus": stats_fus,
           "stats_crudo": stats_crudo, "mapa_fusion": mapa,
           "mov_suavizado": movimientos(fusionados, suaves),
           "log_fusion": log_fusion, "stats_suave": stats_suave, "ranking": ranking,
           "cambios_crudo": contar_cambios(crudos), "cambios_suave": contar_cambios(suaves),
           "base_estado": base_estado, "base_med": base_med, "sig_base": sig_base, "umbral": umbral,
           "estados_sobre_base": sobre_base, "episodios": episodios, "sensibilidad": sens,
           "sens_persistencia": sens_p,
           "crudos": crudos, "suaves": suaves, "archivos": []}
    suf = "_measurement{0}_probe{1}".format(mid, pid) if (mid and pid) else ""
    informe = exportar(ctx, args.output_dir, suf)

    print("")
    for linea in informe:
        print(linea)
    print("EJECUCION COMPLETADA EXITOSAMENTE")


if __name__ == "__main__":
    main()
