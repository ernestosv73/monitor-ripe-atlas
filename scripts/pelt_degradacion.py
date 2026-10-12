#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
PELT / VENTANA DESLIZANTE: EPISODIOS DE DEGRADACION SOSTENIDA DE RTT (RIPE Atlas)
=============================================================================
Metodologia ALTERNATIVA al HDP-HMM (hdphmm_degradacion_bnpy.py). Misma entrada
(CSV crudo del extractor), mismo criterio de episodio y mismas etiquetas
camino/RTT, pero la segmentacion se hace con puntos de cambio.

SERIE: RTT del DESTINO. Se detecta la IP de destino (la IP del ultimo salto que
responde en la mayoria de los ciclos, o --dest-ip) y se toma, en cada ciclo, el
RTT del salto donde responde esa IP (sea el salto 9 o el 10). Los ciclos donde el
destino no responde se excluyen y se informan. El indice de salto del destino
permite etiquetar los episodios: camino (cambia el nº de saltos) o RTT.

MODOS (--modo paper | global | ambos):
  paper  : ventana deslizante con evaluacion EXHAUSTIVA de un unico corte por
           ventana (Baguswasito & Hilman, IEEE Access 2026, Algoritmo 1). Acepta el
           corte si  SSE(sin corte) - [SSE(izq) + SSE(der)] > beta_mult x varianza de la
           ventana (el paper usa beta = 3 sigma^2). Luego fusiona cortes cercanos
           (--tolerancia) y aplica una separacion minima (--separacion-min).
  global : PELT exacto sobre toda la serie (costo SSE de la media, penalizacion
           beta, tamano minimo de segmento = persistencia). beta por BIC robusto
           (2 ln N sigma^2, sigma por MAD de las diferencias) o derivado del umbral
           de degradacion (beta = m x umbral^2 / 2).

CADENCIA: los parametros estadisticos se fijan en MUESTRAS (--ventana-muestras
12, --separacion-min 6, --tolerancia 1) y el script detecta la cadencia (mediana
de las diferencias entre ciclos) para documentar su equivalente en tiempo. La
persistencia, el hueco maximo y la tolerancia se derivan en ciclos de la cadencia.

EPISODIO (decision declarada, igual en ambos modos): segmento (o corrida de
segmentos consecutivos) cuya mediana propia supera la base en >= umbral, con
duracion >= persistencia efectiva. Base = mediana global de la serie; umbral =
max(factor-sigma x sigma_robusto, piso-ms); sigma_robusto = 1.4826 x MAD de los
residuos respecto de la mediana de cada segmento.

OJO: el SSE se aplica sobre la serie tras una mediana movil de --mediana-movil
muestras (default 3), que elimina picos de un solo ciclo (no son sostenidos).
Con 1 se desactiva (fiel al paper, que tambien busca anomalias puntuales).

EVALUACION OPCIONAL (cobertura con tolerancia de +-tolerancia ciclos):
  --eventos-pa    eventos_measurement<id>_probe<id>.csv del extractor (timestamp,categoria)
  --segmentos-hdphmm  segmentos_hdphmm_*.csv de hdphmm_degradacion_bnpy.py

USO:
    python pelt_degradacion.py \\
        --crudo historial_traceroute_measurement9213364_probe24204.csv \\
        --output-dir resultados_pelt --modo ambos \\
        --ventana-muestras 12 --separacion-min 6 \\
        --piso-ms 2 --factor-sigma 3 --persistencia-min 30
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import re
import io
import csv
import math
import argparse
from collections import defaultdict, Counter
from datetime import datetime

try:
    import numpy as np
except ImportError:
    sys.stderr.write("Este script necesita numpy instalado.\n")
    sys.exit(1)

STD_ROB_MIN = 0.01   # piso del desvio robusto (ms)
EPOCH = datetime(1970, 1, 1)


# ============================================================================
# UTILIDADES
# ============================================================================
def extraer_ids_de_archivo(filename):
    m = re.search(r"measurement(\d+)_probe(\d+)", os.path.basename(filename))
    return (m.group(1), m.group(2)) if m else (None, None)


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def epoch_s(s):
    return (parse_ts(s) - EPOCH).total_seconds()


def abrir_csv_w(path):
    if sys.version_info[0] >= 3:
        return open(path, "w", newline="")
    return open(path, "wb")


def escribir_texto(path, texto):
    if isinstance(texto, bytes):
        with open(path, "wb") as f:
            f.write(texto)
    else:
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(texto)


def f2(x):
    return "{0:.2f}".format(x)


def horas_txt(minutos):
    return "{0:.1f} h".format(minutos / 60.0)


# ============================================================================
# CARGA: SERIE DEL DESTINO
# ============================================================================
def leer_historial(path):
    """CSV crudo -> {ts: {hop: (ip, rtt)}}; ignora filas NULL (sin ip o sin rtt)."""
    por_ciclo = defaultdict(dict)
    n_lineas = 0
    with open(path, "r") as f:
        rd = csv.DictReader(f)
        if "ip" not in (rd.fieldnames or []):
            sys.stderr.write("El CSV crudo necesita la columna 'ip' (extract_traceroute_v2.py).\n")
            sys.exit(1)
        for row in rd:
            n_lineas += 1
            ts = row["timestamp"].strip()
            ip = (row.get("ip") or "").strip()
            try:
                hop = int(row["hop"])
                rtt = float(row["rtt_ms"])
            except (ValueError, TypeError, KeyError):
                continue
            if not ip or rtt != rtt:
                continue
            por_ciclo[ts][hop] = (ip, rtt)
    return por_ciclo, n_lineas


# ============================================================================
# CAMBIOS DE CAMINO (AS-path colapsado + longitud)
# ============================================================================
ASN_VACIOS = ("", "unknown", "na", "none", "nan", "null")


def firma_ciclo(hops):
    """hops: {hop: (ip, asn)} -> (aspath colapsado, longitud). Longitud = ultimo salto con IP."""
    aspath, prev = [], None
    for hp in sorted(hops):
        asn = hops[hp][1]
        if asn.strip().lower() in ASN_VACIOS:
            continue
        if asn != prev:
            aspath.append(asn)
            prev = asn
    return tuple(aspath), (max(hops) if hops else 0)


def leer_firmas(path):
    """CSV crudo -> {ts: (aspath, longitud)} (usa solo saltos con IP)."""
    ciclos = defaultdict(dict)
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            ip = (row.get("ip") or "").strip()
            if not ip:
                continue
            try:
                hop = int(row["hop"])
            except (ValueError, TypeError, KeyError):
                continue
            ciclos[row["timestamp"].strip()][hop] = (ip, (row.get("asn") or "").strip())
    return dict((ts, firma_ciclo(h)) for ts, h in ciclos.items())


def analizar_caminos(firmas, rtt_por_ts, persist):
    """Tramos consecutivos con la misma firma (aspath, longitud) y cambios entre tramos SOSTENIDOS.
    Un tramo es sostenido si dura >= persist ciclos; los demas son transitorios (parpadeos)."""
    tramos = []
    for ts in sorted(firmas.keys()):
        f = firmas[ts]
        if tramos and tramos[-1]["firma"] == f:
            tramos[-1]["fin"] = ts
            tramos[-1]["n"] += 1
        else:
            tramos.append({"firma": f, "ini": ts, "fin": ts, "n": 1, "rtts": []})
        if ts in rtt_por_ts:
            tramos[-1]["rtts"].append(rtt_por_ts[ts])
    for t in tramos:
        t["sost"] = t["n"] >= persist
        t["rtt_med"] = float(np.median(t["rtts"])) if t["rtts"] else float("nan")
    cambios, prev = [], None
    for t in tramos:
        if not t["sost"]:
            continue
        if prev is not None and prev["firma"] != t["firma"]:
            tipo = "AS-path" if prev["firma"][0] != t["firma"][0] else "longitud intra-AS"
            cambios.append({"ts": t["ini"], "tipo": tipo, "antes": prev, "despues": t,
                            "d_rtt": t["rtt_med"] - prev["rtt_med"]})
        prev = t
    return tramos, cambios


def texto_caminos(tramos, cambios, persist, umbral_rtt):
    n = len(tramos)
    sost = [t for t in tramos if t["sost"]]
    trans = [t for t in tramos if not t["sost"]]
    nas = sum(1 for c in cambios if c["tipo"] == "AS-path")
    nlg = len(cambios) - nas
    L = ["-" * 74, "CAMBIOS DE CAMINO (AS-path colapsado + longitud)", "-" * 74,
         "  Definicion: firma de un ciclo = (AS-path con ASN consecutivos fusionados, longitud).",
         "  Cambio de IP dentro del mismo ASN (ECMP) NO es cambio de camino.",
         "  Tramo = ciclos consecutivos con la misma firma. Sostenido si dura >= {0} ciclos; el resto es transitorio.".format(persist),
         "  Tramos: {0} ({1} sostenidos, {2} transitorios con {3} ciclos en total)".format(
             n, len(sost), len(trans), sum(t["n"] for t in trans)),
         "  Transiciones entre tramos consecutivos : {0}".format(max(n - 1, 0)),
         "  Cambios sostenidos (sostenido -> sostenido distinto): {0}".format(len(cambios)),
         "    - cambio de AS-path        : {0}".format(nas),
         "    - longitud intra-AS        : {0}".format(nlg),
         "  Distintas firmas sostenidas: {0}".format(len(set(t["firma"] for t in sost)))]
    if cambios:
        L.append("")
        L.append("  {0:<20} {1:<18} {2:>9} {3:>9} {4:>9}  {5}".format(
            "inicio", "tipo", "long.", "ciclos", "d_RTT ms", "RTT"))
        for c in cambios:
            a_, d_ = c["antes"], c["despues"]
            dr = c["d_rtt"]
            marca = "-" if dr != dr else ("RTT cambia" if abs(dr) >= umbral_rtt else "RTT sin cambio")
            L.append("  {0:<20} {1:<18} {2:>9} {3:>9} {4:>9}  {5}".format(
                c["ts"], c["tipo"], "{0}->{1}".format(a_["firma"][1], d_["firma"][1]), d_["n"],
                "n/d" if dr != dr else "{0:+.2f}".format(dr), marca))
    L.append("")
    return "\n".join(L)


def exportar_caminos(tramos, cambios, outdir, suf):
    with abrir_csv_w(os.path.join(outdir, "tramos_camino{0}.csv".format(suf))) as f:
        w = csv.writer(f)
        w.writerow(["inicio", "fin", "ciclos", "longitud", "aspath", "sostenido", "rtt_mediana_ms"])
        for t in tramos:
            w.writerow([t["ini"], t["fin"], t["n"], t["firma"][1], ">".join(t["firma"][0]),
                        int(t["sost"]), "" if t["rtt_med"] != t["rtt_med"] else "{0:.3f}".format(t["rtt_med"])])
    with abrir_csv_w(os.path.join(outdir, "cambios_camino{0}.csv".format(suf))) as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "tipo", "long_antes", "long_despues", "ciclos_tramo", "delta_rtt_ms"])
        for c in cambios:
            w.writerow([c["ts"], c["tipo"], c["antes"]["firma"][1], c["despues"]["firma"][1],
                        c["despues"]["n"], "" if c["d_rtt"] != c["d_rtt"] else "{0:.3f}".format(c["d_rtt"])])


def leer_asn_por_hop(path):
    """CSV crudo -> {ts: {hop: asn}} (solo filas con IP)."""
    m = defaultdict(dict)
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            if not (row.get("ip") or "").strip():
                continue
            try:
                m[row["timestamp"].strip()][int(row["hop"])] = (row.get("asn") or "").strip()
            except (ValueError, TypeError, KeyError):
                continue
    return m


def asn_destino_modal(por_ciclo, asn_map):
    c = Counter()
    for ts, h in por_ciclo.items():
        if h:
            a_ = asn_map.get(ts, {}).get(max(h), "")
            if a_.strip().lower() not in ASN_VACIOS:
                c[a_] += 1
    return c.most_common(1)[0][0] if c else None


def serie_destino_asn(por_ciclo, asn_map, asn_dest):
    """El destino de cada ciclo es su ULTIMO salto con respuesta, sea cual sea su IP (el destino es un
    nombre de dominio y la IP del ultimo salto puede variar por balanceo). Se aceptan los ciclos cuyo
    ultimo salto pertenece a asn_dest; los demas se excluyen (el destino no respondio)."""
    serie, excl = [], 0
    for ts in sorted(por_ciclo.keys()):
        hops = por_ciclo[ts]
        if not hops:
            excl += 1
            continue
        hp = max(hops)
        if asn_map.get(ts, {}).get(hp, "") != asn_dest:
            excl += 1
            continue
        serie.append((ts, hops[hp][1], hp))
    return serie, excl


def ip_destino_modal(por_ciclo):
    c = Counter(h[max(h)][0] for h in por_ciclo.values() if h)
    return c.most_common(1)[0][0] if c else None


def serie_destino(por_ciclo, dest_ip):
    """[(ts, rtt, hop_destino)] de los ciclos donde responde dest_ip; y n de ciclos excluidos."""
    serie, excl = [], 0
    for ts in sorted(por_ciclo.keys()):
        hops = por_ciclo[ts]
        cand = [hp for hp, (ip, _) in hops.items() if ip == dest_ip]
        if not cand:
            excl += 1
            continue
        hp = max(cand)
        serie.append((ts, hops[hp][1], hp))
    return serie, excl


def cadencia_min(serie):
    d = [(epoch_s(serie[i][0]) - epoch_s(serie[i - 1][0])) / 60.0 for i in range(1, len(serie))]
    return float(np.median(d)) if d else 0.0


def cortar_secuencias(serie, gap_min):
    secs, act = [], [serie[0]]
    for i in range(1, len(serie)):
        if (epoch_s(serie[i][0]) - epoch_s(serie[i - 1][0])) / 60.0 > gap_min:
            secs.append(act)
            act = [serie[i]]
        else:
            act.append(serie[i])
    secs.append(act)
    out = []
    for s in secs:
        out.append({"ts": [x[0] for x in s],
                    "t": np.array([epoch_s(x[0]) for x in s]),
                    "y": np.array([x[1] for x in s], dtype=float),
                    "hop": np.array([x[2] for x in s], dtype=int)})
    return out


def mediana_movil(y, k):
    if k <= 1 or len(y) < k:
        return y.copy()
    p = k // 2
    yp = np.concatenate([np.repeat(y[0], p), y, np.repeat(y[-1], p)])
    return np.median(np.vstack([yp[i:i + len(y)] for i in range(k)]), axis=0)


def sigma_ruido_robusto(seqs):
    """sigma del ruido a partir de las diferencias consecutivas (MAD), robusto a picos y cambios de nivel."""
    d = np.concatenate([np.diff(s["y"]) for s in seqs if len(s["y"]) > 1] + [np.zeros(0)])
    if d.size == 0:
        return STD_ROB_MIN
    mad = float(np.median(np.abs(d - np.median(d))))
    return max(1.4826 * mad / math.sqrt(2.0), STD_ROB_MIN)


# ============================================================================
# DETECCION DE PUNTOS DE CAMBIO
# ============================================================================
def _acum(y):
    c = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    return c, c2


def detectar_paper(y, n, beta_mult, segmin, sep, tol):
    """
    Algoritmo 1 del paper: por ventana [start, start+n), compara 'sin corte' contra todos los
    cortes unicos (tamano minimo segmin a cada lado); acepta si la reduccion de SSE supera
    beta_mult x varianza de la ventana. Despues: fusion de cortes a +-tol muestras (se queda el
    de mayor estadistico) y separacion minima sep entre cortes (se queda el de mayor estadistico).
    Devuelve [(indice_primera_muestra_del_nuevo_tramo, estadistico)] ordenado.
    """
    L = len(y)
    if L < n or n < 2 * segmin:
        return []
    c, c2 = _acum(y)
    cand = []
    for start in range(0, L - n + 1):
        end = start + n
        tot = (c2[end] - c2[start]) - (c[end] - c[start]) ** 2 / n
        tot = max(tot, 0.0)
        var = tot / n
        if var <= 1e-12:
            continue
        ks = np.arange(start + segmin, end - segmin + 1)
        if ks.size == 0:
            continue
        izq = (c2[ks] - c2[start]) - (c[ks] - c[start]) ** 2 / (ks - start)
        der = (c2[end] - c2[ks]) - (c[end] - c[ks]) ** 2 / (end - ks)
        gan = tot - (izq + der)
        j = int(np.argmax(gan))
        if gan[j] > beta_mult * var:
            cand.append((int(ks[j]), float(gan[j] / var)))
    if not cand:
        return []
    cand.sort()
    # 1) fusion por tolerancia (encadenada)
    grupos, g = [], [cand[0]]
    for k, st in cand[1:]:
        if k - g[-1][0] <= tol:
            g.append((k, st))
        else:
            grupos.append(g)
            g = [(k, st)]
    grupos.append(g)
    reps = [max(g_, key=lambda x: x[1]) for g_ in grupos]
    # 2) separacion minima
    final = []
    for k, st in reps:
        if final and k - final[-1][0] < sep:
            if st > final[-1][1]:
                final[-1] = (k, st)
        else:
            final.append((k, st))
    return final


def pelt_exacto(y, beta, min_size):
    """Particion optima penalizada exacta (SSE de la media), O(N^2) vectorizado con numpy."""
    n = len(y)
    if n < 2 * min_size:
        return []
    c, c2 = _acum(y)
    INF = float("inf")
    F = np.full(n + 1, INF)
    F[0] = -beta
    last = np.zeros(n + 1, dtype=int)
    for t in range(min_size, n + 1):
        s = np.arange(0, t - min_size + 1)
        seg = (c2[t] - c2[s]) - (c[t] - c[s]) ** 2 / (t - s)
        tot = F[:t - min_size + 1] + seg + beta
        j = int(np.argmin(tot))
        F[t] = tot[j]
        last[t] = j
    cps, t = [], n
    while t > 0:
        s = int(last[t])
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


# ============================================================================
# SEGMENTOS Y EPISODIOS
# ============================================================================
def segmentar(seqs, cps_por_seq):
    """Lista de segmentos con su mediana (sobre el RTT original) y residuos."""
    segs, res = [], []
    for sid, (s, cps) in enumerate(zip(seqs, cps_por_seq)):
        L = len(s["y"])
        b = [0] + sorted(cps) + [L]
        for a, e in zip(b[:-1], b[1:]):
            tr = s["y"][a:e]
            med = float(np.median(tr))
            segs.append({"seq": sid, "a": a, "b": e, "n": e - a, "med": med})
            res.append(tr - med)
    return segs, (np.concatenate(res) if res else np.zeros(0))


def criterio(seqs, segs, res, factor_sigma, piso_ms):
    y_all = np.concatenate([s["y"] for s in seqs])
    base = float(np.median(y_all))
    if res.size:
        mad = float(np.median(np.abs(res - np.median(res))))
        sig = max(1.4826 * mad, STD_ROB_MIN)
    else:
        sig = STD_ROB_MIN
    return base, sig, max(factor_sigma * sig, piso_ms)


def detectar_episodios(seqs, segs, base, umbral, persist_ciclos, cad_min, modal_hop, lim_camino, lim_rtt):
    por_seq = defaultdict(list)
    for sg in segs:
        por_seq[sg["seq"]].append(sg)
    eps = []
    for sid, s in enumerate(seqs):
        L = len(s["y"])
        lst = por_seq[sid]
        i = 0
        while i < len(lst):
            if lst[i]["med"] - base < umbral:
                i += 1
                continue
            j = i
            while j + 1 < len(lst) and lst[j + 1]["med"] - base >= umbral:
                j += 1
            a, b = lst[i]["a"], lst[j]["b"]
            dur = (s["t"][b - 1] - s["t"][a]) / 60.0 + cad_min
            if (b - a) >= persist_ciclos:   # en ciclos: robusto al jitter de segundos de los timestamps
                tr = s["y"][a:b]
                med = float(np.median(tr))
                nm = int(np.sum(s["hop"][a:b] != modal_hop))
                pct = 100.0 * nm / (b - a)
                eps.append({"seq_id": sid, "a": a, "b": b, "inicio": s["ts"][a], "fin": s["ts"][b - 1],
                            "n_ciclos": b - a, "duracion_min": dur, "mediana_ms": med,
                            "magnitud_ms": med - base,
                            "pico_ms": max(x["med"] for x in lst[i:j + 1]) - base,
                            "n_segmentos": j - i + 1, "recuperado": b < L, "inicio_censurado": a == 0,
                            "pct_no_modal": pct,
                            "tipo": ("camino" if pct >= lim_camino else ("RTT" if pct <= lim_rtt else "mixto"))})
            i = j + 1
    for k, e in enumerate(eps):
        e["id"] = k + 1
    return eps


def conteos(eps, total_ciclos):
    return {"n": len(eps),
            "n_rtt": sum(1 for e in eps if e["tipo"] == "RTT"),
            "n_mixto": sum(1 for e in eps if e["tipo"] == "mixto"),
            "n_camino": sum(1 for e in eps if e["tipo"] == "camino"),
            "minutos": sum(e["duracion_min"] for e in eps),
            "pct_tiempo": 100.0 * sum(e["n_ciclos"] for e in eps) / max(total_ciclos, 1.0)}


# ============================================================================
# EJECUCION DE UN MODO
# ============================================================================
def correr(modo, P, seqs, filt, sig_ruido, cad_min, modal_hop, **ov):
    """P: dict de parametros; ov: sobrescribe (ventana, beta_mult, beta_factor, beta_modo)."""
    p = dict(P)
    p.update(ov)
    cps_por_seq, stats_por_seq, beta_info = [], [], None
    if modo == "paper":
        for yf in filt:
            r = detectar_paper(yf, p["ventana"], p["beta_mult"], p["segmento_minimo"],
                               p["separacion_min"], p["tolerancia"])
            cps_por_seq.append([k for k, _ in r])
            stats_por_seq.append(dict(r))
    else:
        N = sum(len(y) for y in filt)
        if p["beta_modo"] == "derivado":
            umbral0 = max(p["factor_sigma"] * sig_ruido, p["piso_ms"])
            beta = p["persist_ciclos"] * umbral0 ** 2 / 2.0
        else:
            beta = p["beta_factor"] * 2.0 * math.log(max(N, 2)) * sig_ruido ** 2
        beta_info = beta
        for yf in filt:
            cps_por_seq.append(pelt_exacto(yf, beta, p["min_size"]))
            stats_por_seq.append({})
    segs, res = segmentar(seqs, cps_por_seq)
    base, sig, umbral = criterio(seqs, segs, res, p["factor_sigma"], p["piso_ms"])
    eps = detectar_episodios(seqs, segs, base, umbral, p["persist_ciclos"], cad_min, modal_hop,
                             p["no_modal_camino"], p["no_modal_rtt"])
    total = float(sum(len(s["y"]) for s in seqs))
    return {"cps": cps_por_seq, "stats": stats_por_seq, "segs": segs, "base": base, "sig": sig,
            "umbral": umbral, "eps": eps, "beta": beta_info, "conteos": conteos(eps, total),
            "n_cps": sum(len(c) for c in cps_por_seq)}


# ============================================================================
# EVALUACION POR COBERTURA
# ============================================================================
def tiempos_cortes(seqs, cps_por_seq):
    return sorted(s["t"][k] for s, cps in zip(seqs, cps_por_seq) for k in cps)


def cobertura(a, b, tol_s):
    """Fraccion de elementos de a con algun elemento de b a <= tol_s (ambas listas ordenadas)."""
    if not a:
        return 0, 0
    b = np.array(b)
    ok = 0
    for x in a:
        if b.size and np.min(np.abs(b - x)) <= tol_s:
            ok += 1
    return ok, len(a)


def leer_eventos_pa(path):
    ts = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(row["timestamp"].strip())
            except KeyError:
                continue
    return sorted(set(ts))


def pa_persistentes(seqs, ts_pa, m, umbral):
    """Eventos de PA que estan en la serie y cuyo nivel cambia >= umbral y se mantiene m ciclos."""
    idx = {}
    for sid, s in enumerate(seqs):
        for k, t in enumerate(s["ts"]):
            idx[t] = (sid, k)
    pers, en_serie = [], 0
    for t in ts_pa:
        if t not in idx:
            continue
        en_serie += 1
        sid, k = idx[t]
        y = seqs[sid]["y"]
        if k < 1 or k + m > len(y):
            continue
        antes = float(np.median(y[max(0, k - m):k]))
        desp = float(np.median(y[k:k + m]))
        if abs(desp - antes) >= umbral:
            pers.append(epoch_s(t))
    return pers, en_serie


def leer_cortes_hdphmm(path):
    """Instantes de inicio de cada segmento del HDP-HMM, salvo el primero de cada secuencia."""
    primeros, out = set(), []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            sid = row.get("seq_id")
            if sid not in primeros:
                primeros.add(sid)
                continue
            out.append(epoch_s(row["inicio"].strip()))
    return sorted(out)


# ============================================================================
# INFORME
# ============================================================================
def tabla_sens(filas, etiqueta):
    L = ["  {0:>16} {1:>8} {2:>10} {3:>5} {4:>6} {5:>7} {6:>8} {7:>9}".format(
        etiqueta, "cortes", "episodios", "RTT", "mixto", "camino", "horas", "% tiempo")]
    for f in filas:
        c = f["conteos"]
        L.append("  {0:>16} {1:>8} {2:>10} {3:>5} {4:>6} {5:>7} {6:>8.1f} {7:>9.1f}{8}".format(
            f["etq"], f["n_cps"], c["n"], c["n_rtt"], c["n_mixto"], c["n_camino"],
            c["minutos"] / 60.0, c["pct_tiempo"], "   <- elegido" if f.get("elegido") else ""))
    return L


def construir_informe(modo, ctx):
    a, r = ctx["args"], ctx["res"]
    P = ctx["P"]
    L = []
    L.append("=" * 74)
    L.append("DEGRADACION SOSTENIDA DE RTT - PUNTOS DE CAMBIO ({0})".format(
        "ventana deslizante exhaustiva (paper)" if modo == "paper" else "PELT exacto global"))
    L.append("=" * 74)
    L.append("Measurement {0} | Probe {1}".format(ctx["mid"] or "?", ctx["pid"] or "?"))
    L.append("Fecha de ejecucion: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    L.append("")
    L.append("1. DATOS Y CADENCIA")
    L.append("  - Destino: {0}{1}; ciclos usados {2}, excluidos (destino sin respuesta) {3}".format(
        ctx["dest_ip"], ctx["dest_desc"],
        ctx["n_serie"], ctx["n_excl"]))
    L.append("  - Salto del destino por ciclo: {0}; salto modal = {1}".format(
        ", ".join("hop {0}: {1} ({2:.1f}%)".format(h, n, 100.0 * n / ctx["n_serie"])
                  for h, n in ctx["cont_hop"].most_common(5)), ctx["modal_hop"]))
    L.append("  - Rango: {0} a {1}; {2} secuencia(s) (hueco > {3} ciclos = {4:.0f} min)".format(
        ctx["ini"], ctx["fin"], len(ctx["seqs"]), a.gap_ciclos, a.gap_ciclos * ctx["cad"]))
    L.append("  - RTT: min {0} | mediana {1} | max {2} ms".format(
        f2(ctx["rtt_min"]), f2(ctx["rtt_med"]), f2(ctx["rtt_max"])))
    L.append("  - CADENCIA detectada: {0:.1f} min ({1:.0f} s). Los parametros estadisticos se fijan en "
             "MUESTRAS; su equivalente en tiempo con esta cadencia:".format(ctx["cad"], ctx["cad"] * 60))
    L.append("      ventana          = {0} muestras = {1}".format(a.ventana_muestras, horas_txt(a.ventana_muestras * ctx["cad"])))
    L.append("      separacion minima = {0} muestras = {1}".format(a.separacion_min, horas_txt(a.separacion_min * ctx["cad"])))
    L.append("      tolerancia        = +-{0} muestra(s) = +-{1:.0f} min".format(a.tolerancia, a.tolerancia * ctx["cad"]))
    L.append("      persistencia      = {0:g} min pedidos -> {1:.0f} min efectivos ({2} ciclos)".format(
        a.persistencia_min, P["persist_efectiva_min"], P["persist_ciclos"]))
    for w in ctx["avisos"]:
        L.append("  [AVISO] " + w)
    L.append("")
    L.append("2. METODO Y PARAMETROS")
    if modo == "paper":
        L.append("  - Ventana deslizante (paso 1 muestra), un unico corte por ventana, evaluacion exhaustiva.")
        L.append("  - Se acepta el corte si ganancia de SSE > {0:g} x varianza de la ventana "
                 "(paper: beta = 3 sigma^2); segmento minimo {1} muestra(s).".format(a.beta_mult, a.segmento_minimo))
        L.append("  - Fusion de cortes a +-{0} muestra(s) y separacion minima de {1} muestras.".format(a.tolerancia, a.separacion_min))
    else:
        L.append("  - PELT exacto (O(N^2) vectorizado), costo SSE de la media, tamano minimo de segmento {0} ciclos.".format(P["min_size"]))
        L.append("  - beta = {0:.3f} ms^2 ({1}; sigma ruido robusto = {2:.3f} ms; beta/sigma^2 = {3:.1f})".format(
            r["beta"], "derivado del umbral" if a.beta_global == "derivado" else
            "BIC robusto x {0:g}".format(a.beta_factor), ctx["sig_ruido"], r["beta"] / ctx["sig_ruido"] ** 2))
    L.append("  - Serie filtrada con mediana movil de {0} muestra(s) antes de buscar cortes{1}.".format(
        a.mediana_movil, " (desactivada)" if a.mediana_movil <= 1 else " (elimina picos de un ciclo)"))
    L.append("")
    L.append("3. SEGMENTOS ({0} cortes -> {1} segmentos; {2:.1f} cortes por cada 100 ciclos)".format(
        r["n_cps"], len(r["segs"]), 100.0 * r["n_cps"] / max(ctx["n_serie"], 1)))
    largos = sorted(r["segs"], key=lambda x: -x["n"])[:8]
    L.append("  Segmentos mas largos (por ciclos):")
    L.append("   {0:>4} {1:>8} {2:>9} {3:>10}  {4}".format("Seq", "Ciclos", "Mediana", "Dif.base", "Inicio"))
    for sg in largos:
        s = ctx["seqs"][sg["seq"]]
        L.append("   {0:>4} {1:>8} {2:>9} {3:>+10.2f}  {4}".format(
            sg["seq"], sg["n"], f2(sg["med"]), sg["med"] - r["base"], s["ts"][sg["a"]]))
    L.append("")
    L.append("4. CRITERIO DE DEGRADACION")
    L.append("  - Base = mediana global de la serie = {0:.2f} ms; sigma robusto (residuos por segmento) = {1:.2f} ms".format(
        r["base"], r["sig"]))
    L.append("  - Umbral = max({0:g} x sigma, {1:g} ms) = {2:.2f} ms sobre la base".format(a.factor_sigma, a.piso_ms, r["umbral"]))
    L.append("  - Episodio = segmento (o corrida de segmentos consecutivos) con mediana >= base + umbral y al menos {0} ciclos ({1:.0f} min).".format(
        P["persist_ciclos"], P["persist_efectiva_min"]))
    L.append("  - Tipo segun % de ciclos con el destino en un salto distinto del modal ({0}): camino >= {1:g}%, RTT <= {2:g}%, mixto en medio.".format(
        ctx["modal_hop"], a.no_modal_camino, a.no_modal_rtt))
    L.append("  - Difiere del HDP-HMM en que aqui cada segmento debe superar el umbral por si mismo (no se agrupan tramos casi iguales a la base).")
    L.append("")
    eps = r["eps"]
    L.append("5. EPISODIOS DE DEGRADACION SOSTENIDA ({0})".format(len(eps)))
    if not eps:
        L.append("  Ningun episodio supera el umbral y la persistencia minima.")
    else:
        L.append("  {0:>3} {1:<19} {2:<19} {3:>7} {4:>9} {5:>9} {6:>7} {7:>8}  {8}".format(
            "id", "inicio", "fin", "ciclos", "dur(min)", "mag(ms)", "tipo", "%no-mod", "nota"))
        for e in eps:
            nota = []
            if e["inicio_censurado"]:
                nota.append("inicio censurado")
            if not e["recuperado"]:
                nota.append("sin recuperar")
            L.append("  {0:>3} {1:<19} {2:<19} {3:>7} {4:>9.0f} {5:>9.2f} {6:>7} {7:>8.1f}  {8}".format(
                e["id"], e["inicio"], e["fin"], e["n_ciclos"], e["duracion_min"], e["magnitud_ms"],
                e["tipo"], e["pct_no_modal"], ", ".join(nota)))
    L.append("")
    L.append("6. SENSIBILIDAD")
    if modo == "paper":
        L.append("  a) A la ventana (muestras; beta_mult {0:g}):".format(a.beta_mult))
        L += tabla_sens(ctx["sens_a"], "ventana(horas)")
        L.append("  b) A beta_mult (ventana {0} muestras):".format(a.ventana_muestras))
        L += tabla_sens(ctx["sens_b"], "beta_mult")
    else:
        L.append("  A beta (BIC robusto x factor; la fila 'derivado' usa beta = m x umbral^2 / 2):")
        L += tabla_sens(ctx["sens_a"], "beta")
    L.append("  Lectura: un episodio que desaparece al cambiar la penalizacion o la ventana depende del ajuste, no de los datos.")
    L.append("")
    n = 7
    if ctx["eval"]:
        L.append("{0}. EVALUACION POR COBERTURA (tolerancia +-{1} ciclo(s) = +-{2:.0f} min)".format(
            n, a.tolerancia, a.tolerancia * ctx["cad"]))
        L += ctx["eval"]
        L.append("")
        n += 1
    L.append("{0}. ARCHIVOS".format(n))
    for pth in ctx["archivos"]:
        L.append("  - " + pth)
    return "\n".join(L) + "\n"


# ============================================================================
# EXPORTACION
# ============================================================================
def exportar(modo, ctx, outdir, suf):
    r = ctx["res"]
    seqs = ctx["seqs"]
    rutas = []

    def ruta(nombre):
        p = os.path.join(outdir, "{0}_{1}{2}.csv".format(nombre, modo, suf))
        rutas.append(p)
        return p

    ep_de = {}
    for e in r["eps"]:
        for k in range(e["a"], e["b"]):
            ep_de[(e["seq_id"], k)] = e["id"]
    seg_de = {}
    for i, sg in enumerate(r["segs"]):
        for k in range(sg["a"], sg["b"]):
            seg_de[(sg["seq"], k)] = (i + 1, sg["med"])

    with abrir_csv_w(ruta("ciclos_pelt")) as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "rtt_ms", "rtt_filtrado_ms", "hop_destino", "seq_id", "segmento",
                    "mediana_segmento_ms", "dif_base_ms", "degradado", "episodio_id"])
        for sid, (s, yf) in enumerate(zip(seqs, ctx["filt"])):
            for k in range(len(s["y"])):
                sg_id, med = seg_de[(sid, k)]
                eid = ep_de.get((sid, k), "")
                w.writerow([s["ts"][k], "{0:.3f}".format(s["y"][k]), "{0:.3f}".format(yf[k]), int(s["hop"][k]),
                            sid, sg_id, "{0:.3f}".format(med), "{0:.3f}".format(med - r["base"]),
                            1 if eid != "" else 0, eid])
    with abrir_csv_w(ruta("segmentos_pelt")) as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "inicio", "fin", "n_ciclos", "duracion_min", "mediana_ms", "dif_base_ms",
                    "sobre_umbral", "episodio_id"])
        for sg in r["segs"]:
            s = seqs[sg["seq"]]
            dur = (s["t"][sg["b"] - 1] - s["t"][sg["a"]]) / 60.0 + ctx["cad"]
            w.writerow([sg["seq"], s["ts"][sg["a"]], s["ts"][sg["b"] - 1], sg["n"], "{0:.1f}".format(dur),
                        "{0:.3f}".format(sg["med"]), "{0:.3f}".format(sg["med"] - r["base"]),
                        1 if sg["med"] - r["base"] >= r["umbral"] else 0,
                        ep_de.get((sg["seq"], sg["a"]), "")])
    with abrir_csv_w(ruta("puntos_cambio_pelt")) as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "timestamp", "estadistico"])
        for sid, (s, cps, st) in enumerate(zip(seqs, r["cps"], r["stats"])):
            for k in cps:
                w.writerow([sid, s["ts"][k], "{0:.3f}".format(st[k]) if k in st else ""])
    with abrir_csv_w(ruta("episodios_pelt")) as f:
        w = csv.writer(f)
        w.writerow(["id", "seq_id", "inicio", "fin", "n_ciclos", "duracion_min", "mediana_ms", "magnitud_ms",
                    "pico_ms", "n_segmentos", "recuperado", "inicio_censurado", "pct_no_modal", "tipo"])
        for e in r["eps"]:
            w.writerow([e["id"], e["seq_id"], e["inicio"], e["fin"], e["n_ciclos"], "{0:.1f}".format(e["duracion_min"]),
                        "{0:.3f}".format(e["mediana_ms"]), "{0:.3f}".format(e["magnitud_ms"]),
                        "{0:.3f}".format(e["pico_ms"]), e["n_segmentos"], int(e["recuperado"]),
                        int(e["inicio_censurado"]), "{0:.1f}".format(e["pct_no_modal"]), e["tipo"]])
    with abrir_csv_w(ruta("sensibilidad_pelt")) as f:
        w = csv.writer(f)
        w.writerow(["variacion", "valor", "cortes", "episodios", "rtt", "mixto", "camino", "horas", "pct_tiempo", "elegido"])
        for var, filas in ctx["sens_all"]:
            for fl in filas:
                c = fl["conteos"]
                w.writerow([var, fl["etq"], fl["n_cps"], c["n"], c["n_rtt"], c["n_mixto"], c["n_camino"],
                            "{0:.2f}".format(c["minutos"] / 60.0), "{0:.2f}".format(c["pct_tiempo"]),
                            1 if fl.get("elegido") else 0])
    p_res = os.path.join(outdir, "resumen_pelt_{0}{1}.txt".format(modo, suf))
    rutas.append(p_res)
    ctx["archivos"] = rutas
    texto = construir_informe(modo, ctx)
    escribir_texto(p_res, texto)
    return texto


# ============================================================================
# MAIN
# ============================================================================
def parse_lista(s, tipo=float):
    return [tipo(x) for x in s.split(",") if x.strip()]


def parse_arguments():
    p = argparse.ArgumentParser(description="Episodios de degradacion de RTT con puntos de cambio (PELT / ventana exhaustiva)")
    p.add_argument("--crudo", required=True, help="CSV crudo del extractor (timestamp,hop,ip,asn,rtt_ms)")
    p.add_argument("--output-dir", default="resultados_pelt")
    p.add_argument("--dest-ip", default=None, help="IP de destino (default: IP modal del ultimo salto)")
    p.add_argument("--destino-por", choices=["asn", "ip"], default="asn",
                   help="asn: destino = ultimo salto de cada ciclo si pertenece al ASN modal (dominio con IP variable); "
                        "ip: solo ciclos donde responde la IP modal")
    p.add_argument("--modo", choices=["paper", "global", "ambos"], default="ambos")
    p.add_argument("--ventana-muestras", type=int, default=12, help="Ventana (muestras) del modo paper")
    p.add_argument("--separacion-min", type=int, default=6, help="Separacion minima entre cortes (muestras)")
    p.add_argument("--tolerancia", type=int, default=1, help="Fusion de cortes y tolerancia de evaluacion (muestras/ciclos)")
    p.add_argument("--beta-mult", type=float, default=3.0, help="Modo paper: beta = beta_mult x varianza de la ventana")
    p.add_argument("--segmento-minimo", type=int, default=1, help="Modo paper: muestras minimas a cada lado del corte (paper: 1)")
    p.add_argument("--beta-global", choices=["bic", "derivado"], default="bic", help="Modo global: origen de beta")
    p.add_argument("--beta-factor", type=float, default=1.0, help="Modo global: multiplicador del beta BIC")
    p.add_argument("--mediana-movil", type=int, default=3, help="Muestras de la mediana movil previa (1 = sin filtro)")
    p.add_argument("--gap-ciclos", type=float, default=6.0, help="Hueco (en ciclos) que corta la serie en secuencias")
    p.add_argument("--piso-ms", type=float, default=2.0)
    p.add_argument("--factor-sigma", type=float, default=3.0)
    p.add_argument("--persistencia-min", type=float, default=30.0, help="Persistencia pedida (min); efectiva >= 2 ciclos")
    p.add_argument("--no-modal-camino", type=float, default=80.0)
    p.add_argument("--no-modal-rtt", type=float, default=20.0)
    p.add_argument("--ventanas-sensibilidad", default="6,12,18,30")
    p.add_argument("--beta-mult-sensibilidad", default="2,3,4,6,9")
    p.add_argument("--beta-factores-sensibilidad", default="0.25,0.5,1,2,4")
    p.add_argument("--persistencia-camino-ciclos", type=int, default=4,
                   help="Ciclos minimos para que un tramo de camino sea sostenido (menos = transitorio)")
    p.add_argument("--eventos-pa", default=None, help="eventos_*.csv del extractor (timestamp,categoria)")
    p.add_argument("--segmentos-hdphmm", default=None, help="segmentos_hdphmm_*.csv del HDP-HMM")
    return p.parse_args()


def main():
    a = parse_arguments()
    print("=" * 74)
    print("Degradacion de RTT - puntos de cambio (PELT / ventana exhaustiva)")
    print("=" * 74)
    if not os.path.isfile(a.crudo):
        sys.stderr.write("No existe: {0}\n".format(a.crudo))
        sys.exit(1)
    mid, pid = extraer_ids_de_archivo(a.crudo)
    suf = "_measurement{0}_probe{1}".format(mid, pid) if mid else ""
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)

    por_ciclo, n_lin = leer_historial(a.crudo)
    print("[FASE 1] {0} lineas, {1} ciclos con alguna respuesta".format(n_lin, len(por_ciclo)))
    if a.dest_ip:
        dest_ip = a.dest_ip
        dest_desc = " (--dest-ip)"
        serie, n_excl = serie_destino(por_ciclo, dest_ip)
    elif a.destino_por == "ip":
        dest_ip = ip_destino_modal(por_ciclo)
        dest_desc = " (IP modal del ultimo salto)"
        serie, n_excl = serie_destino(por_ciclo, dest_ip)
    else:
        asn_map = leer_asn_por_hop(a.crudo)
        asn_d = asn_destino_modal(por_ciclo, asn_map)
        if asn_d is None:
            sys.stderr.write("No se pudo determinar el ASN del ultimo salto; use --destino-por ip.\n")
            sys.exit(1)
        serie, n_excl = serie_destino_asn(por_ciclo, asn_map, asn_d)
        ips_ult = Counter(por_ciclo[ts_][hp_][0] for ts_, _, hp_ in serie)
        if len(ips_ult) == 1:
            dest_ip = "ultimo salto de AS{0} (IP fija {1})".format(asn_d, list(ips_ult)[0])
        else:
            ip_top, n_top = ips_ult.most_common(1)[0]
            dest_ip = "ultimo salto de AS{0} ({1} IP distintas; la mas frecuente {2}: {3:.1f}%)".format(
                asn_d, len(ips_ult), ip_top, 100.0 * n_top / max(len(serie), 1))
        dest_desc = " (ASN modal del ultimo salto; se acepta cualquier IP del ASN)"
    if len(serie) < 2 * a.ventana_muestras:
        sys.stderr.write("Serie demasiado corta ({0} ciclos) para una ventana de {1}.\n".format(len(serie), a.ventana_muestras))
        sys.exit(1)
    cad = cadencia_min(serie)
    seqs = cortar_secuencias(serie, a.gap_ciclos * cad)
    print("[FASE 2] destino {0}: {1} ciclos ({2} excluidos); cadencia {3:.1f} min; {4} secuencia(s)".format(
        dest_ip, len(serie), n_excl, cad, len(seqs)))

    avisos = []
    persist_efectiva = max(a.persistencia_min, 2.0 * cad)
    if persist_efectiva > a.persistencia_min + 1e-9:
        avisos.append("persistencia pedida {0:g} min < 2 ciclos: se usan {1:.0f} min efectivos.".format(
            a.persistencia_min, persist_efectiva))
    persist_ciclos = int(math.ceil(persist_efectiva / cad - 1e-9))
    if a.ventana_muestras < 6:
        avisos.append("ventana de {0} muestras: muy poca potencia estadistica.".format(a.ventana_muestras))
    if a.ventana_muestras * cad > 24 * 60:
        avisos.append("la ventana equivale a {0} (> 24 h): la deteccion no es de tipo 'casi tiempo real'.".format(
            horas_txt(a.ventana_muestras * cad)))
    if a.separacion_min >= a.ventana_muestras:
        avisos.append("la separacion minima es >= a la ventana: se perderan cortes cercanos.")
    n_cortas = sum(1 for s in seqs if len(s["y"]) < a.ventana_muestras)
    if n_cortas:
        avisos.append("{0} secuencia(s) mas cortas que la ventana (modo paper no las analiza).".format(n_cortas))

    cont_hop = Counter(h for s in seqs for h in s["hop"].tolist())
    modal_hop = cont_hop.most_common(1)[0][0]
    y_all = np.concatenate([s["y"] for s in seqs])
    filt = [mediana_movil(s["y"], a.mediana_movil) for s in seqs]
    sig_ruido = sigma_ruido_robusto(seqs)

    P = {"ventana": a.ventana_muestras, "beta_mult": a.beta_mult, "segmento_minimo": a.segmento_minimo,
         "separacion_min": a.separacion_min, "tolerancia": a.tolerancia, "beta_modo": a.beta_global,
         "beta_factor": a.beta_factor, "factor_sigma": a.factor_sigma, "piso_ms": a.piso_ms,
         "persist_efectiva_min": persist_efectiva, "persist_ciclos": persist_ciclos,
         "min_size": max(2, persist_ciclos), "no_modal_camino": a.no_modal_camino, "no_modal_rtt": a.no_modal_rtt}

    ts_pa = cortes_hdp = None
    if a.eventos_pa:
        if os.path.isfile(a.eventos_pa):
            ts_pa = leer_eventos_pa(a.eventos_pa)
        else:
            avisos.append("no existe --eventos-pa ({0}): se omite esa evaluacion.".format(a.eventos_pa))
    if a.segmentos_hdphmm:
        if os.path.isfile(a.segmentos_hdphmm):
            cortes_hdp = leer_cortes_hdphmm(a.segmentos_hdphmm)
        else:
            avisos.append("no existe --segmentos-hdphmm ({0}): se omite la comparacion con el HDP-HMM "
                          "(ese CSV lo genera hdphmm_degradacion_bnpy.py).".format(a.segmentos_hdphmm))
    for w in avisos:
        print("[AVISO] " + w)
    tol_s = a.tolerancia * cad * 60.0 * 1.1

    firmas = leer_firmas(a.crudo)
    rtt_ts = dict((ts, r) for ts, r, _ in serie)
    tramos, cambios = analizar_caminos(firmas, rtt_ts, max(1, a.persistencia_camino_ciclos))
    txt_cam = texto_caminos(tramos, cambios, max(1, a.persistencia_camino_ciclos), a.piso_ms)
    exportar_caminos(tramos, cambios, a.output_dir, suf)
    escribir_texto(os.path.join(a.output_dir, "informe_caminos{0}.txt".format(suf)), txt_cam)
    print(txt_cam)

    modos = ["paper", "global"] if a.modo == "ambos" else [a.modo]
    for modo in modos:
        print("[MODO {0}] buscando puntos de cambio...".format(modo))
        res = correr(modo, P, seqs, filt, sig_ruido, cad, modal_hop)
        sens_all = []

        def fila(etq, r_, elegido=False):
            return {"etq": etq, "n_cps": r_["n_cps"], "conteos": r_["conteos"], "elegido": elegido}
        if modo == "paper":
            vs = sorted(set(parse_lista(a.ventanas_sensibilidad, int) + [a.ventana_muestras]))
            sa = []
            for v in vs:
                rr = res if v == a.ventana_muestras else correr(modo, P, seqs, filt, sig_ruido, cad, modal_hop, ventana=v)
                sa.append(fila("{0} ({1:.1f})".format(v, v * cad / 60.0), rr, v == a.ventana_muestras))
            bs = sorted(set(parse_lista(a.beta_mult_sensibilidad) + [a.beta_mult]))
            sb = []
            for b in bs:
                rr = res if b == a.beta_mult else correr(modo, P, seqs, filt, sig_ruido, cad, modal_hop, beta_mult=b)
                sb.append(fila("{0:g}".format(b), rr, b == a.beta_mult))
            sens_all = [("ventana_muestras", sa), ("beta_mult", sb)]
        else:
            fs = sorted(set(parse_lista(a.beta_factores_sensibilidad) + [a.beta_factor]))
            sa = []
            for fct in fs:
                rr = correr(modo, P, seqs, filt, sig_ruido, cad, modal_hop, beta_modo="bic", beta_factor=fct)
                sa.append(fila("BIC x{0:g}".format(fct), rr, a.beta_global == "bic" and fct == a.beta_factor))
            rr = correr(modo, P, seqs, filt, sig_ruido, cad, modal_hop, beta_modo="derivado")
            sa.append(fila("derivado", rr, a.beta_global == "derivado"))
            sens_all = [("beta_global", sa)]
            sb = []

        ev = []
        t_cortes = tiempos_cortes(seqs, res["cps"])
        if ts_pa is not None:
            pers, en_serie = pa_persistentes(seqs, ts_pa, max(3, persist_ciclos), res["umbral"])
            t_pa = [epoch_s(t) for t in ts_pa]
            ok_p, n_p = cobertura(t_cortes, t_pa, tol_s)
            ok_r, n_r = cobertura(t_pa, t_cortes, tol_s)
            ok_k, n_k = cobertura(pers, t_cortes, tol_s)
            ev.append("  Contra Path Analysis ({0} eventos, {1} en la serie, {2} persistentes: el nivel cambia >= {3:.2f} ms durante {4} ciclos):".format(
                len(ts_pa), en_serie, len(pers), res["umbral"], max(3, persist_ciclos)))
            ev.append("    cortes con un evento de PA a +-tol : {0}/{1} ({2:.1f}%)".format(ok_p, n_p, 100.0 * ok_p / max(n_p, 1)))
            ev.append("    eventos de PA cubiertos por un corte : {0}/{1} ({2:.1f}%)".format(ok_r, n_r, 100.0 * ok_r / max(n_r, 1)))
            ev.append("    eventos PERSISTENTES de PA cubiertos : {0}/{1} ({2:.1f}%)".format(ok_k, n_k, 100.0 * ok_k / max(n_k, 1)))
            ev.append("    (PA marca cambios puntuales de un ciclo; solo los persistentes son comparables con un corte sostenido.)")
        if cortes_hdp is not None:
            ok_p, n_p = cobertura(t_cortes, cortes_hdp, tol_s)
            ok_r, n_r = cobertura(cortes_hdp, t_cortes, tol_s)
            ev.append("  Contra los limites de segmento del HDP-HMM ({0} limites):".format(len(cortes_hdp)))
            ev.append("    cortes que coinciden con un limite del HDP-HMM: {0}/{1} ({2:.1f}%)".format(ok_p, n_p, 100.0 * ok_p / max(n_p, 1)))
            ev.append("    limites del HDP-HMM cubiertos por un corte    : {0}/{1} ({2:.1f}%)".format(ok_r, n_r, 100.0 * ok_r / max(n_r, 1)))

        ctx = {"args": a, "res": res, "P": P, "mid": mid, "pid": pid, "dest_ip": dest_ip, "dest_desc": dest_desc, "n_serie": len(serie),
               "n_excl": n_excl, "cont_hop": cont_hop, "modal_hop": modal_hop, "seqs": seqs, "filt": filt,
               "cad": cad, "ini": serie[0][0], "fin": serie[-1][0], "rtt_min": float(y_all.min()),
               "rtt_med": float(np.median(y_all)), "rtt_max": float(y_all.max()), "avisos": avisos,
               "sig_ruido": sig_ruido, "sens_a": sa, "sens_b": sb, "sens_all": sens_all, "eval": ev}
        texto = exportar(modo, ctx, a.output_dir, suf)
        print()
        print(texto)


if __name__ == "__main__":
    main()
