#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_bootstrap.py - Historial sintetico por REMUESTREO POR BLOQUES de la sonda real + INYECCION CONTROLADA de escalones.

Idea: el ruido, el jitter, los spikes, los ciclos sin destino, los huecos y la estructura de ruta son los de la propia
sonda (no se modelan ni se calibran). Solo se inyectan escalones de latencia con delta, duracion y posicion CONOCIDOS.
La verdad de campo es exactamente lo inyectado.

Pasos
  1. Bloque principal (verificar_caminos): ruta fija (longitud + AS-path). Serie = RTT del ultimo salto de sus ciclos.
  2. Pool limpio: segmentacion independiente del HDP-HMM (mediana movil + umbral, la misma del calibrador). Se conservan los
     tramos en regimen base (nivel < --umbral-degradacion sobre la base). Los niveles sub-umbral (p. ej. +1.3 ms) quedan
     dentro del pool: son parte del ruido real de la sonda.
  3. Remuestreo por bloques solapados de largo fijo (moving block bootstrap, Kunsch 1989; solapados y largo fijo segun
     Lahiri 1999). Un bloque son ciclos CONSECUTIVOS COMPLETOS (todas las filas de cada ciclo, con su diferencia de tiempo
     al ciclo anterior); cada bloque cae dentro de un unico tramo base. No tiene relacion con los "bloques" de
     verificar_caminos (esos son tramos de ruta fija: aqui son la FUENTE del pool, no el largo de remuestreo).
  4. Inyeccion controlada: diseno factorial deltas x duraciones x repeticiones, en orden aleatorio, separados por tramos base.
     El delta se suma al RTT de los saltos >= salto de localizacion que respondieron (como en el generador anterior).

Salidas (mismo formato que el generador anterior; el evaluador y paso1/paso2 no cambian):
  historial_traceroute_measurement<M>_probe<P>.csv   mismas columnas que el historial real
  verdad_ciclos_measurement<M>_probe<P>.csv          idx,timestamp,hueco,regimen,delta_ms,salto_localizacion,nivel_ultimo_ms,
                                                     spike_ms,sin_destino,modo_fallo,n_saltos_emitidos,rtt_ultimo_ms
  verdad_episodios_measurement<M>_probe<P>.csv       un episodio por escalon inyectado
  costuras_measurement<M>_probe<P>.csv               uniones entre bloques y salto de nivel en cada una
  informe_bootstrap_measurement<M>_probe<P>.txt      pool, bloques, diseno, costuras y limitaciones

Limitaciones (se imprimen en el informe)
  * El ruido es el mismo en base y en degradacion (el delta es aditivo).
  * Los ciclos del pool se reutilizan (la serie sintetica es mas larga que el pool): las semillas no son independientes.
  * Resultados condicionales a la sonda de origen.
  * Las costuras pueden crear saltos de nivel sub-umbral que no son un escalon inyectado (se reportan).

Uso
  python3 generar_bootstrap.py --historial historial_traceroute_measurement123_probe456.csv \\
      --bloque-principal vc_real/bloque_principal_measurement123_probe456.json \\
      --ciclos-excluidos vc_real/ciclos_excluidos_measurement123_probe456.csv \\
      --deltas 2,3,5,10,20 --duraciones-h 1,6,48 --repeticiones 2 --largo-bloque-h 24 --seed 1 --output-dir boot_s1
Requiere solo numpy (script autonomo).
"""
from __future__ import print_function, division
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np

from collections import defaultdict

FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# utilidades (copiadas de calibrar_perfil_desde_bloque.py para que este script sea autonomo)
# ---------------------------------------------------------------------------
ASN_VACIOS = ("", "unknown", "none", "nan", "null", "?", "*")


def a_seg(ts):
    return (datetime.strptime(ts.strip()[:19], FMT) - datetime(1970, 1, 1)).total_seconds()


def asn_ok(a):
    return (a or "").strip().lower() not in ASN_VACIOS


def aspath(h):
    p = []
    for k in sorted(h):
        a = h[k][1].strip()
        if not asn_ok(a):
            continue
        if not p or p[-1] != a:
            p.append(a)
    return p


def leer_ciclos(path):
    """{ts: {hop: (ip, asn, rtt|None)}} con solo los saltos que respondieron (ip no vacia)."""
    ciclos = defaultdict(dict)
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            ip = (r.get("ip") or "").strip()
            ts = (r.get("timestamp") or "").strip()
            if not ts:
                ciclos.setdefault(ts, {})
                continue
            ciclos.setdefault(ts, {})
            if not ip:
                continue
            try:
                hop = int(r["hop"])
            except (ValueError, TypeError, KeyError):
                continue
            try:
                rtt = float(r.get("rtt_ms"))
            except (ValueError, TypeError):
                rtt = None
            ciclos[ts][hop] = (ip, (r.get("asn") or "").strip(), rtt)
    ciclos.pop("", None)
    return ciclos


def leer_excluidos(path):
    ex = {}
    if path and os.path.exists(path):
        with open(path, "r", newline="") as f:
            for r in csv.DictReader(f):
                ts = (r.get("ciclo") or "").strip()
                if ts:
                    ex[ts] = r
    return ex


def segmentar_niveles(y, ventana, umbral, dwell):
    """Segmentacion por mediana movil + umbral. Devuelve lista de (i0, i1) sobre y (indices inclusivos)."""
    n = len(y)
    if n < 2 * dwell:
        return [(0, n - 1)]
    w = max(3, ventana | 1)
    pad = w // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    m = np.array([np.median(yp[i:i + w]) for i in range(n)])
    segs = []
    ini = 0
    nivel = float(np.median(m[:dwell]))
    t = dwell
    while t < n:
        d = m[t] - nivel
        if abs(d) >= umbral:
            fin = min(n, t + dwell)
            tramo = m[t:fin]
            if len(tramo) >= dwell and np.all(np.sign(tramo - nivel) == np.sign(d)) and \
                    np.all(np.abs(tramo - nivel) >= umbral):
                # el nivel nuevo empieza en el primer ciclo consecutivo que ya se aparto del nivel anterior
                j = t
                while j > ini and np.sign(m[j - 1] - nivel) == np.sign(d) and abs(m[j - 1] - nivel) >= umbral:
                    j -= 1
                segs.append((ini, j - 1))
                ini = j
                nivel = float(np.median(m[ini:min(n, ini + dwell)]))
                t = ini + dwell
                continue
        nivel = float(np.median(m[ini:t + 1]))
        t += 1
    segs.append((ini, n - 1))
    segs = [(a, b) for a, b in segs if b >= a]
    return segs




def leer_filas_crudas(path):
    """(cabecera, {ts: [fila, ...]}) conservando todas las columnas y filas (tambien saltos mudos)."""
    with open(path, "r", newline="") as f:
        rd = csv.reader(f)
        cab = next(rd)
        i_ts = cab.index("timestamp")
        por_ts = {}
        for fila in rd:
            if len(fila) > i_ts and fila[i_ts].strip():
                por_ts.setdefault(fila[i_ts].strip(), []).append(fila)
    return cab, por_ts


def fmt(x):
    return "{:.3f}".format(x)


def acf_agrupada(segmentos, K):
    """Autocorrelacion de lags 1..K sobre una lista de segmentos (sin cruzar el borde de ningun segmento)."""
    todo = np.concatenate(segmentos)
    m = float(todo.mean())
    c0 = float(np.mean((todo - m) ** 2))
    out = np.full(K, np.nan)
    if c0 <= 0:
        return out
    for k in range(1, K + 1):
        num, cnt = 0.0, 0
        for s in segmentos:
            if len(s) > k:
                d = s - m
                num += float(np.sum(d[:-k] * d[k:]))
                cnt += len(s) - k
        if cnt > 0:
            out[k - 1] = num / cnt / c0
    return out


def tramos_contiguos(idx):
    """Tramos de indices consecutivos de un arreglo ordenado: lista de (inicio_pos, fin_pos) sobre `idx`."""
    out = []
    if len(idx) == 0:
        return out
    i0 = 0
    for k in range(1, len(idx)):
        if idx[k] - idx[k - 1] != 1:
            out.append((i0, k - 1))
            i0 = k
    out.append((i0, len(idx) - 1))
    return out


def main():
    ap = argparse.ArgumentParser(description="Historial sintetico por remuestreo por bloques + inyeccion controlada")
    ap.add_argument("--historial", required=True, help="historial_traceroute_*.csv real (crudo)")
    ap.add_argument("--bloque-principal", required=True, help="bloque_principal_*.json de verificar_caminos")
    ap.add_argument("--ciclos-excluidos", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", default="2026-01-01 00:00:00", help="Inicio de la serie sintetica")
    ap.add_argument("--dias", type=float, default=None,
                    help="Duracion minima de la serie; por defecto la fija el diseno. Si es mayor, se agrega base al final")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--measurement-id", default=None, help="Default: '9' + measurement real")
    ap.add_argument("--probe-id", default=None, help="Default: el de la sonda real")
    # modo y pool
    ap.add_argument("--modo", choices=["crudo", "residuos"], default="crudo",
                    help="crudo: bloques de RTT con su nivel (conserva el vaiven sub-umbral; el pool se limita a una banda de "
                         "niveles de ancho < umbral). residuos: bloques de residuos + nivel base constante (sin vaiven; "
                         "sin costuras de nivel; el pool es todo el regimen base)")
    ap.add_argument("--ventana", type=int, default=9)
    ap.add_argument("--umbral-ms", type=float, default=1.5, help="Cambio de nivel para la segmentacion independiente")
    ap.add_argument("--min-dwell", type=int, default=5)
    ap.add_argument("--umbral-degradacion", type=float, default=2.0,
                    help="Nivel >= base + umbral = degradacion real: queda fuera del pool (default 2 ms)")
    ap.add_argument("--k-spike", type=float, default=6.0)
    ap.add_argument("--spike-min-ms", type=float, default=3.0)
    ap.add_argument("--largo-bloque-h", default=None,
                    help="Largo del bloque en horas, o 'auto' (solo modo residuos: el menor largo de --candidatos-auto cuya "
                         "autocorrelacion se parece a la real). Default: 24 en crudo, auto en residuos")
    ap.add_argument("--candidatos-auto", default="3,6,12,24,48,96,192", help="Largos candidatos de 'auto', en ciclos")
    ap.add_argument("--lags-h", type=float, default=2.0, help="Horizonte de autocorrelacion que se compara (default 2 h)")
    ap.add_argument("--tol-acf", type=float, default=0.05, help="Diferencia maxima de autocorrelacion aceptada (default 0.05)")
    ap.add_argument("--tol-tasa", type=float, default=0.25,
                    help="Diferencia relativa maxima de la tasa de spikes y de ciclos sin destino respecto del pool (default 0.25)")
    ap.add_argument("--max-intentos", type=int, default=30, help="Re-sorteos del remuestreo si las tasas no entran en tolerancia")
    ap.add_argument("--sin-spikes", action="store_true",
                    help="Recorta (winsoriza) el RTT del ultimo salto de los spikes: variante para aislar su efecto")
    # diseno
    ap.add_argument("--deltas", default="2,3,5,10,20", help="Deltas inyectados en ms (default 2,3,5,10,20)")
    ap.add_argument("--duraciones-h", default="1,6,48", help="Duraciones de los escalones en horas (default 1,6,48)")
    ap.add_argument("--repeticiones", type=int, default=2, help="Repeticiones por combinacion delta x duracion (default 2)")
    ap.add_argument("--separacion-h", type=float, default=12.0,
                    help="Base minima entre escalones, en horas; la real es separacion x U(1, 1.5) (default 12)")
    ap.add_argument("--margen-h", type=float, default=12.0, help="Base inicial y final (default 12)")
    ap.add_argument("--umbral-severa", type=float, default=5.0, help="Solo para rotular moderada/severa (default 5 ms)")
    a = ap.parse_args()

    modo = a.modo
    if a.largo_bloque_h is None:
        a.largo_bloque_h = "24" if modo == "crudo" else "auto"
    auto = str(a.largo_bloque_h).lower() == "auto"
    if auto and modo == "crudo":
        sys.exit("[ERROR] --largo-bloque-h auto solo existe en modo residuos: en crudo el bloque debe ser largo para "
                 "conservar el nivel y se fija a mano.")

    with open(a.bloque_principal) as f:
        bp = json.load(f)
    el = bp["elegido"]
    interval = float(bp["cadencia_s"])
    n_long = int(el["longitud"])
    ruta_ref = el["ruta"].split(">") if el["ruta"] else []
    mid = a.measurement_id or ("9" + str(bp.get("measurement")))
    pid = a.probe_id or str(bp.get("probe"))
    rng = np.random.RandomState(a.seed)

    # ---------------------------------------------------------------- lectura y bloque
    cab, crudas = leer_filas_crudas(a.historial)
    i_ts, i_hop, i_rtt = cab.index("timestamp"), cab.index("hop"), cab.index("rtt_ms")
    ciclos = leer_ciclos(a.historial)
    excl = leer_excluidos(a.ciclos_excluidos)
    t0, t1 = a_seg(el["inicio"]), a_seg(el["fin"])
    orden = sorted(ts for ts in ciclos if t0 <= a_seg(ts) <= t1)
    n_b = len(orden)
    tiempos = np.array([a_seg(ts) for ts in orden])
    propio = np.zeros(n_b, dtype=bool)
    for i, ts in enumerate(orden):
        h = ciclos[ts]
        if not h or ts in excl:
            continue
        if max(h) != n_long or aspath(h) != ruta_ref or h[max(h)][2] is None:
            continue
        propio[i] = True
    if propio.sum() < 200:
        sys.exit("[ERROR] solo {} ciclos propios en el bloque principal.".format(int(propio.sum())))

    # ---------------------------------------------------------------- segmentacion independiente
    idx_p = np.where(propio)[0]
    y = np.array([ciclos[orden[i]][max(ciclos[orden[i]])][2] for i in idx_p])
    segs = segmentar_niveles(y, a.ventana, a.umbral_ms, a.min_dwell)
    nivel_p = np.zeros(len(y))
    for (i0, i1) in segs:
        nivel_p[i0:i1 + 1] = float(np.median(y[i0:i1 + 1]))
    base0 = float(np.percentile(nivel_p, 10))
    grupo = nivel_p[(nivel_p - base0) < a.umbral_degradacion]
    base_lvl = float(np.median(grupo)) if len(grupo) else base0
    res = y - nivel_p
    mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
    thr_spike = max(a.spike_min_ms, a.k_spike * mad)

    nivel = np.zeros(n_b)
    resid = np.zeros(n_b)
    k = 0
    ult = nivel_p[0]
    for i in range(n_b):
        if propio[i]:
            ult = nivel_p[k]
            resid[i] = res[k]
            k += 1
        nivel[i] = ult
    es_base = (nivel - base_lvl) < a.umbral_degradacion
    spike_flag = propio & (resid > thr_spike)
    rc = np.clip(resid, -thr_spike, thr_spike)   # ruido sin spikes: para la autocorrelacion

    dts = np.diff(tiempos)
    dts_ok = dts[(dts > 0.5 * interval) & (dts < 1.5 * interval)]
    if len(dts_ok) == 0:
        dts_ok = np.array([interval])
    dt_med = float(np.median(dts_ok))

    # ---------------------------------------------------------------- pool segun el modo
    banda_txt = ""
    if modo == "crudo":
        # banda de niveles de ancho < umbral-degradacion que contenga mas ciclos base: asi dos ciclos cualesquiera del pool
        # (dentro de un bloque o en una costura) difieren menos de 2 ms y la verdad de nivel no tiene eventos que no se inyectaron
        W = a.umbral_degradacion - 0.01
        lv = np.sort(nivel[es_base])
        j = 0
        mejor = (0, lv[0] if len(lv) else 0.0)
        for i in range(len(lv)):
            while lv[i] - lv[j] >= W:
                j += 1
            if i - j + 1 > mejor[0]:
                mejor = (i - j + 1, lv[j])
        lo = float(mejor[1])
        pool_mask = es_base & (nivel >= lo - 1e-9) & (nivel < lo + W)
        banda_txt = "banda de niveles [{:.2f}, {:.2f}) ms (ancho {:.2f} < {:g}): conserva {:.0%} de los ciclos base".format(
            lo, lo + W, W, a.umbral_degradacion, pool_mask.sum() / float(max(es_base.sum(), 1)))
    else:
        pool_mask = es_base.copy()
    pi = np.where(pool_mask)[0]
    runs_pos = tramos_contiguos(pi)          # tramos contiguos del pool, en posiciones de `pi`
    largos_run = [b - a_ + 1 for a_, b in runs_pos]
    if len(pi) < 50:
        sys.exit("[ERROR] el pool tiene solo {} ciclos.".format(len(pi)))
    n_pool = len(pi)

    # ---------------------------------------------------------------- diseno (en ciclos)
    deltas = [float(x) for x in a.deltas.split(",") if x.strip()]
    durs = [float(x) for x in a.duraciones_h.split(",") if x.strip()]
    combos = [(d, h) for d in deltas for h in durs for _ in range(a.repeticiones)]
    rng.shuffle(combos)
    ciclos_h = 3600.0 / dt_med
    margen = int(round(a.margen_h * ciclos_h))
    sep = a.separacion_h * ciclos_h
    cursor = margen
    eps = []
    for (d, h) in combos:
        ini = int(round(cursor + sep * rng.uniform(1.0, 1.5)))
        dur = max(1, int(round(h * ciclos_h)))
        eps.append({"d": d, "h": h, "ini": ini, "fin": ini + dur - 1, "k": int(rng.randint(max(2, n_long - 3), n_long + 1))})
        cursor = ini + dur
    N = cursor + margen
    if a.dias is not None:
        N = max(N, int(round(a.dias * 86400.0 / dt_med)))

    # ---------------------------------------------------------------- referencias del pool (para elegir largo y chequear)
    K = max(3, int(round(a.lags_h * 3600.0 / dt_med)))
    ref_segs = [rc[pi[a_]:pi[b] + 1] for a_, b in runs_pos if b - a_ + 1 > K]
    if not ref_segs:
        ref_segs = [rc[pi]]
    acf_ref = acf_agrupada(ref_segs, K)
    tasa_spike_ref = float(spike_flag[pi].mean())
    tasa_fallo_ref = float((~propio[pi]).mean())
    pares = np.diff(pi) == 1
    gap_ref = float((np.diff(tiempos[pi])[pares] > 1.5 * interval).mean()) if pares.any() else 0.0

    # ---------------------------------------------------------------- largo del bloque
    L_pedido_ciclos = None
    tabla_auto = []
    if auto:
        cands = sorted(set(int(x) for x in a.candidatos_auto.split(",") if x.strip()))
        cands = [c for c in cands if c <= max(2, n_pool // 4)] or [max(2, n_pool // 4)]
        M = min(N, 6 * n_pool)
        elegido_auto = None
        for c in cands:
            acfs = []
            for _ in range(3):
                xs, n_x = [], 0
                while n_x < M:
                    p = int(rng.randint(0, n_pool - c + 1))
                    xs.append(rc[pi[p:p + c]])
                    n_x += c
                acfs.append(acf_agrupada([np.concatenate(xs)[:M]], K))
            dif = float(np.nanmax(np.abs(np.mean(acfs, axis=0) - acf_ref)))
            tabla_auto.append((c, dif))
            if elegido_auto is None and dif <= a.tol_acf:
                elegido_auto = c
        if elegido_auto is None:
            elegido_auto = min(tabla_auto, key=lambda z: z[1])[0]
            aviso_auto = "ningun candidato alcanza la tolerancia de autocorrelacion ({:g}): se usa el de menor diferencia".format(a.tol_acf)
        else:
            aviso_auto = None
        L_pedido_ciclos = int(elegido_auto)
    else:
        aviso_auto = None
        L_pedido_ciclos = max(1, int(round(float(a.largo_bloque_h) * 3600.0 / interval)))
    L = L_pedido_ciclos
    aviso_L = None
    if modo == "crudo" and max(largos_run) < L:
        aviso_L = ("el tramo del pool mas largo ({} ciclos) es menor que el bloque pedido ({} ciclos): se usa {}"
                   .format(max(largos_run), L, max(largos_run)))
        L = max(largos_run)
    if L > n_pool:
        L = n_pool

    if modo == "crudo":
        ok_start = (pi[L - 1:] - pi[:len(pi) - L + 1]) == (L - 1)
        starts = np.where(ok_start)[0]
    else:
        starts = np.arange(0, n_pool - L + 1)
    if len(starts) == 0:
        sys.exit("[ERROR] ningun bloque de {} ciclos cabe en el pool.".format(L))
    n_runs_ok = sum(1 for x in largos_run if x >= L)

    # ---------------------------------------------------------------- serie base remuestreada (con control de tasas)
    def construir():
        src, dtn, cost = [], [], []
        while len(src) < N:
            p = int(starts[rng.randint(len(starts))])
            blq = pi[p:p + L]
            if len(blq) > 1:
                d = np.where(np.diff(blq) == 1, tiempos[blq[1:]] - tiempos[blq[:-1]],
                             dts_ok[rng.randint(len(dts_ok), size=len(blq) - 1)])
            else:
                d = np.array([])
            if src:
                cost.append((len(src), float(nivel[blq[0]] - nivel[src[-1]]) if modo == "crudo" else 0.0))
            dtn.append(float(dts_ok[rng.randint(len(dts_ok))]))
            dtn.extend(d.tolist())
            src.extend(blq.tolist())
        return np.array(src[:N]), np.array(dtn[:N]), cost

    def desvio_tasas(src_):
        ds = float(spike_flag[src_].mean())
        df = float((~propio[src_]).mean())
        rel = lambda x, r: abs(x - r) / r if r > 1e-9 else (0.0 if x < 1e-3 else 1.0)  # noqa: E731
        return max(rel(ds, tasa_spike_ref), rel(df, tasa_fallo_ref))

    mejor_ser, mejor_dev, intentos, paso = None, None, 0, False
    for intentos in range(1, a.max_intentos + 1):
        ser = construir()
        dev = desvio_tasas(ser[0])
        if mejor_dev is None or dev < mejor_dev:
            mejor_ser, mejor_dev = ser, dev
        if dev <= a.tol_tasa:
            paso = True
            break
    src, dt_nuevo, costuras = mejor_ser
    dt_nuevo = dt_nuevo.copy()
    dt_nuevo[0] = 0.0

    t_ini = datetime.strptime(a.start, FMT)
    t_seg = np.cumsum(dt_nuevo)
    ts_new = [(t_ini + timedelta(seconds=int(round(x)))).strftime(FMT) for x in t_seg]
    for i in range(1, N):
        if ts_new[i] <= ts_new[i - 1]:
            ts_new[i] = (datetime.strptime(ts_new[i - 1], FMT) + timedelta(seconds=1)).strftime(FMT)

    delta_c = np.zeros(N)
    kloc_c = np.zeros(N, dtype=int)
    clase_c = [""] * N
    for e in eps:
        for j in range(e["ini"], min(e["fin"], N - 1) + 1):
            delta_c[j] = e["d"]
            kloc_c[j] = e["k"]
            clase_c[j] = "severa" if e["d"] >= a.umbral_severa else "moderada"

    # ---------------------------------------------------------------- escritura
    os.makedirs(a.output_dir, exist_ok=True)
    sfx = "measurement{}_probe{}".format(mid, pid)
    f_h = open(os.path.join(a.output_dir, "historial_traceroute_{}.csv".format(sfx)), "w", newline="")
    f_c = open(os.path.join(a.output_dir, "verdad_ciclos_{}.csv".format(sfx)), "w", newline="")
    wh = csv.writer(f_h, lineterminator="\n")
    wc = csv.writer(f_c, lineterminator="\n")
    wh.writerow(cab)
    wc.writerow(["idx", "timestamp", "hueco", "regimen", "delta_ms", "salto_localizacion", "nivel_ultimo_ms", "spike_ms",
                 "sin_destino", "modo_fallo", "n_saltos_emitidos", "rtt_ultimo_ms"])
    hop_nivel_min = max(2, n_long - 3)   # saltos a los que se traslada el nivel en modo residuos
    n_spk = 0
    for j in range(N):
        ix = int(src[j])
        filas = crudas[orden[ix]]
        d = float(delta_c[j])
        kk = int(kloc_c[j])
        corr = (base_lvl - float(nivel[ix])) if modo == "residuos" else 0.0
        nivel_verdad = (base_lvl if modo == "residuos" else float(nivel[ix])) + d
        spike = float(resid[ix]) if spike_flag[ix] else 0.0
        rtt_ult = None
        max_hop = 0
        for fila in filas:
            nf = list(fila)
            nf[i_ts] = ts_new[j]
            try:
                hop = int(nf[i_hop])
            except ValueError:
                hop = None
            rt_txt = nf[i_rtt].strip() if len(nf) > i_rtt else ""
            if hop is not None and rt_txt != "":
                try:
                    rt = float(rt_txt)
                except ValueError:
                    rt = None
                if rt is not None:
                    if a.sin_spikes and spike > 0 and hop == n_long:
                        rt = rt - spike
                    if corr != 0.0 and hop >= hop_nivel_min:
                        rt = max(rt + corr, 0.01)
                    if d > 0 and hop >= kk:
                        rt = rt + d
                    nf[i_rtt] = fmt(rt)
                    if hop == n_long:
                        rtt_ult = rt
            if hop is not None and (len(nf) > 2 and nf[2].strip() != ""):
                max_hop = max(max_hop, hop)
            wh.writerow(nf)
        if a.sin_spikes:
            spike = 0.0
        n_spk += 1 if spike > 0 else 0
        sin_dest = 0 if propio[ix] else 1
        wc.writerow([j, ts_new[j], 0, clase_c[j] or "base", fmt(d), kk if d > 0 else "", fmt(nivel_verdad),
                     fmt(spike) if spike > 0 else "0", sin_dest, "", max_hop,
                     "" if (sin_dest or rtt_ult is None) else fmt(rtt_ult)])
    f_h.close()
    f_c.close()

    with open(os.path.join(a.output_dir, "verdad_episodios_{}.csv".format(sfx)), "w", newline="") as fe:
        we = csv.writer(fe, lineterminator="\n")
        we.writerow(["tipo", "inicio", "fin", "n_ciclos", "magnitud_ms", "detalle"])
        for n_e, e in enumerate(sorted(eps, key=lambda z: z["ini"]), 1):
            fin = min(e["fin"], N - 1)
            we.writerow(["degradacion_" + ("severa" if e["d"] >= a.umbral_severa else "moderada"), ts_new[e["ini"]],
                         ts_new[fin], fin - e["ini"] + 1, fmt(e["d"]),
                         "salto_localizacion={};duracion_h={:g};diseno={}".format(e["k"], e["h"], n_e)])
    with open(os.path.join(a.output_dir, "costuras_{}.csv".format(sfx)), "w", newline="") as fcs:
        w = csv.writer(fcs, lineterminator="\n")
        w.writerow(["idx", "timestamp", "salto_nivel_ms"])
        for pos, dl in costuras:
            w.writerow([pos, ts_new[pos], fmt(dl)])

    # ---------------------------------------------------------------- chequeo de realismo: pool real vs serie sintetica
    prop_s = propio[src]
    res_s = resid[src][prop_s]
    res_p = resid[pi][propio[pi]]
    acf_syn = acf_agrupada([rc[src]], K)
    dif_acf = np.abs(acf_syn - acf_ref)
    tasa_spike_s = float(spike_flag[src].mean())
    tasa_fallo_s = float((~prop_s).mean())
    gap_s = float((dt_nuevo[1:] > 1.5 * interval).mean())

    def rob(x):
        return float(np.median(np.abs(x - np.median(x)))) * 1.4826

    filas_r = []   # (metrica, pool, sintetico, estado)
    lags_mostrar = sorted(set([l for l in (1, 2, 3, 6, 12, K) if l <= K]))
    for l in lags_mostrar:
        filas_r.append(("autocorrelacion lag {}".format(l), acf_ref[l - 1], acf_syn[l - 1],
                        "OK" if dif_acf[l - 1] <= a.tol_acf else "REVISAR"))
    filas_r.append(("sigma robusta del residuo (ms)", rob(res_p), rob(res_s), "OK" if abs(rob(res_p) - rob(res_s)) <= 0.15 * max(rob(res_p), 1e-9) else "REVISAR"))
    for q in (10, 50, 90, 99):
        vp, vs = float(np.percentile(res_p, q)), float(np.percentile(res_s, q))
        filas_r.append(("percentil {} del residuo (ms)".format(q), vp, vs,
                        "OK" if abs(vp - vs) <= max(0.25 * abs(vp), 0.15) else "REVISAR"))

    def est_tasa(x, r):
        return "OK" if (abs(x - r) <= a.tol_tasa * r or abs(x - r) < 0.002) else "REVISAR"

    filas_r.append(("tasa de spikes", tasa_spike_ref, tasa_spike_s, est_tasa(tasa_spike_s, tasa_spike_ref)))
    filas_r.append(("tasa de ciclos sin destino", tasa_fallo_ref, tasa_fallo_s, est_tasa(tasa_fallo_s, tasa_fallo_ref)))
    filas_r.append(("tasa de huecos (dt > 1.5 x cadencia)", gap_ref, gap_s, est_tasa(gap_s, gap_ref)))
    with open(os.path.join(a.output_dir, "realismo_{}.csv".format(sfx)), "w", newline="") as fr:
        w = csv.writer(fr, lineterminator="\n")
        w.writerow(["metrica", "pool_real", "sintetico", "estado"])
        for m_, vp, vs, st in filas_r:
            w.writerow([m_, "{:.4f}".format(vp), "{:.4f}".format(vs), st])

    # referencia opcional: Politis-White (si hay libreria)
    ref_pw = None
    try:
        from tsbootstrap.block.pwsd import optimal_block_length  # type: ignore
        ref_pw = float(np.ravel(optimal_block_length(np.concatenate(ref_segs).reshape(-1, 1)))[0])
    except Exception:
        try:
            from arch.bootstrap import optimal_block_length as _obl  # type: ignore
            ref_pw = float(_obl(np.concatenate(ref_segs)).iloc[0, 0])
        except Exception:
            ref_pw = None

    # ---------------------------------------------------------------- informe
    saltos = np.array([abs(x[1]) for x in costuras]) if costuras else np.array([0.0])
    L_inf = []
    P = L_inf.append
    P("=" * 74)
    P("HISTORIAL SINTETICO POR REMUESTREO POR BLOQUES - measurement {} | probe {} | semilla {}".format(mid, pid, a.seed))
    P("=" * 74)
    P("Modo: {}".format("CRUDO (bloques de RTT con su nivel)" if modo == "crudo" else
                         "RESIDUOS (bloques de residuos + nivel base constante {:.2f} ms)".format(base_lvl)))
    P("Fuente: bloque {} de la sonda real ({} saltos, {}), {} ciclos, {} propios".format(
        el["bloque_id"], n_long, el["ruta"], n_b, int(propio.sum())))
    P("Pool: {} tramos, {} ciclos ({:.0%} del bloque); nivel base {:.2f} ms; tramo mas largo {} ciclos".format(
        len(runs_pos), n_pool, n_pool / float(n_b), base_lvl, max(largos_run)))
    if banda_txt:
        P("Pool crudo: " + banda_txt)
    P("Bloque de remuestreo: {} ciclos ({:.1f} h), solapado y de largo fijo{}; posiciones de inicio posibles: {}".format(
        L, L * dt_med / 3600.0, " (elegido con 'auto')" if auto else "", len(starts)))
    if modo == "crudo":
        P("Tramos del pool que admiten el bloque: {} de {}".format(n_runs_ok, len(runs_pos)))
        if len(starts) > 0:
            # reparto de posiciones de inicio por tramo
            cont = [0] * len(runs_pos)
            for s_ in starts:
                for q, (a_, b_) in enumerate(runs_pos):
                    if a_ <= s_ <= b_:
                        cont[q] += 1
                        break
            P("Posiciones de inicio en el tramo mas usado: {:.0%}".format(max(cont) / float(len(starts))))
    if aviso_L:
        P("[AVISO] " + aviso_L)
    if auto:
        P("Seleccion automatica del largo (menor largo con diferencia maxima de autocorrelacion <= {:g} hasta lag {}):".format(a.tol_acf, K))
        for c, dif in tabla_auto:
            P("   {:>4} ciclos ({:>5.1f} h): diferencia maxima {:.3f}{}".format(c, c * dt_med / 3600.0, dif,
                                                                            "   <-- elegido" if c == L_pedido_ciclos else ""))
        if aviso_auto:
            P("[AVISO] " + aviso_auto)
    P("Referencia Politis-White (varianza de la media; no es el criterio usado): {}".format(
        "{:.0f} ciclos".format(ref_pw) if ref_pw is not None else "no disponible (requiere tsbootstrap o arch)"))
    P("Serie sintetica: {} ciclos, {:.1f} dias | ciclos del pool: {} (cada uno se usa en promedio {:.1f} veces)".format(
        N, N * dt_med / 86400.0, n_pool, N / float(max(n_pool, 1))))
    P("Control de tasas: {} sorteo(s); desvio maximo de las tasas de spikes y de fallos {:.0%} (tolerancia {:.0%}){}".format(
        intentos, mejor_dev, a.tol_tasa, "" if paso else "  [AVISO] no se alcanzo la tolerancia; se uso el mejor sorteo"))
    P("Spikes (residuo > {:.2f} ms): {} ciclos{}".format(thr_spike, n_spk, " (recortados: --sin-spikes)" if a.sin_spikes else ""))
    P("")
    P("--- Diseno controlado ---")
    P("Deltas {} ms x duraciones {} h x {} repeticiones = {} escalones | separacion >= {:g} h | margen {:g} h".format(
        deltas, durs, a.repeticiones, len(eps), a.separacion_h, a.margen_h))
    P("Salto de localizacion: uniforme en [{}, {}]".format(max(2, n_long - 3), n_long))
    corto = [h for h in durs if h * ciclos_h < 10]
    if corto:
        P("[AVISO] duraciones {} h = menos de 10 ciclos: el evaluador (min-dwell 5) las tratara como mesetas cortas".format(corto))
    P("")
    P("--- Uniones entre bloques ---")
    if modo == "crudo":
        P("{} uniones | salto de nivel |.|: mediana {:.2f} ms, max {:.2f} ms (la banda del pool lo acota por debajo de {:g} ms) | "
          ">= 1.5 ms: {}".format(len(costuras), float(np.median(saltos)), float(np.max(saltos)), a.umbral_degradacion,
                                 int((saltos >= 1.5).sum())))
        P("Estos saltos son parte de la verdad de nivel y siempre quedan bajo el umbral: el evaluador los informa como "
          "cambios sub-umbral y no como eventos.")
    else:
        P("{} uniones; el nivel es constante, asi que ninguna crea un salto de nivel. Costo: no hay vaiven sub-umbral "
          "(la sonda real tiene unos {}). Comparar con el modo crudo.".format(
              len(costuras), "cambios de nivel entre 1.5 y 2 ms entre sus tramos base"))
    P("")
    P("--- Chequeo de realismo (pool real vs serie sintetica) ---")
    P("{:<42s} {:>10s} {:>10s}  {}".format("metrica", "pool real", "sintetico", "estado"))
    for m_, vp, vs, st in filas_r:
        P("{:<42s} {:>10.4f} {:>10.4f}  {}".format(m_, vp, vs, st))
    P("Autocorrelacion: diferencia maxima hasta lag {} = {:.3f} (tolerancia {:g}); calculada sobre el residuo sin spikes.".format(
        K, float(np.nanmax(dif_acf)), a.tol_acf))
    P("")
    P("--- Limitaciones ---")
    P("* El ruido es el mismo en base y en degradacion (el delta es aditivo).")
    P("* Los ciclos del pool se reutilizan: las semillas no son independientes entre si.")
    P("* Resultados condicionales a la sonda de origen; repetir en otras sondas.")
    P("* El largo de bloque es una decision de diseno: repetir con varios y comparar (sensibilidad).")
    if modo == "residuos":
        P("* Modo residuos: sin vaiven sub-umbral; el nivel se traslada a los saltos >= {} de cada ciclo.".format(hop_nivel_min))
    P("")
    P("Salidas en: {}".format(a.output_dir))
    with open(os.path.join(a.output_dir, "informe_bootstrap_{}.txt".format(sfx)), "w") as f:
        f.write("\n".join(L_inf) + "\n")
    print("\n".join(L_inf))


if __name__ == "__main__":
    main()
