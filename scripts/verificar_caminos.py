#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_caminos.py - Verificacion unica de los cambios de camino de una sonda.

Integra las 3 comprobaciones y las cruza en una sola salida:
  1. Eventos del extractor (eventos_*.csv): ciclos con cambio_longitud / cambio_ip_asn y su solapamiento.
  2. Cambios de camino REALES (historial_*.csv): AS-path colapsado y longitud entre ciclos consecutivos.
  3. Tramos consecutivos con la misma longitud (sostenidos vs parpadeos).
  + Cruce 1 x 2: cuantos eventos del extractor corresponden a un cambio de camino real.
  + LECTURA: conclusiones automaticas.

Uso:
  python3 verificar_caminos.py --historial historial_traceroute_measurement<M>_probe<P>.csv \
                               [--eventos eventos_measurement<M>_probe<P>.csv] [--persistencia 4] [--output-dir .]
Si no se indica --eventos se intenta con el nombre equivalente (historial_traceroute_ -> eventos_);
si no existe, se omiten las comprobaciones que lo necesitan.
"""
from __future__ import print_function, division
import os
import re
import io
import csv
import sys
import argparse
from collections import defaultdict, Counter

ASN_VACIOS = ("", "unknown", "na", "none", "nan", "null")


def asn_valido(a):
    return a.strip().lower() not in ASN_VACIOS


def aspath(h):
    """h: {hop: asn} -> AS-path colapsado (ASN consecutivos fusionados; ASN vacios/desconocidos omitidos)."""
    p = []
    for k in sorted(h):
        a = h[k].strip()
        if not asn_valido(a):
            continue
        if not p or p[-1] != a:
            p.append(a)
    return tuple(p)


def leer_historial(path):
    ciclos = defaultdict(dict)
    n = 0
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            n += 1
            if not (r.get("ip") or "").strip():
                continue
            try:
                ciclos[r["timestamp"].strip()][int(r["hop"])] = (r.get("asn") or "")
            except (ValueError, TypeError, KeyError):
                continue
    return ciclos, n


def leer_eventos(path):
    cats = defaultdict(Counter)
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            cats[r["timestamp"].strip()][r["categoria"].strip()] += 1
    return cats


def pct(a, b):
    return "{0:.1f}%".format(100.0 * a / b) if b else "n/d"


def ids_de_archivo(p):
    m = re.search(r"measurement(\d+)_probe(\d+)", os.path.basename(p))
    return (m.group(1), m.group(2)) if m else (None, None)


def main():
    ap = argparse.ArgumentParser(description="Verificacion unica de cambios de camino (eventos vs historial)")
    ap.add_argument("--historial", required=True)
    ap.add_argument("--eventos", default=None)
    ap.add_argument("--persistencia", type=int, default=4,
                    help="Ciclos minimos para que un tramo de misma longitud sea sostenido (default 4)")
    ap.add_argument("--output-dir", default=".")
    a = ap.parse_args()

    if not os.path.isfile(a.historial):
        sys.stderr.write("No existe: {0}\n".format(a.historial))
        sys.exit(1)
    mid, pid = ids_de_archivo(a.historial)
    ev_path = a.eventos
    if not ev_path:
        cand = os.path.join(os.path.dirname(a.historial),
                            os.path.basename(a.historial).replace("historial_traceroute_", "eventos_"))
        ev_path = cand if os.path.isfile(cand) else None
    elif not os.path.isfile(ev_path):
        sys.stderr.write("[AVISO] no existe --eventos ({0}): se omiten las comprobaciones 1 y el cruce.\n".format(ev_path))
        ev_path = None

    ciclos, n_lin = leer_historial(a.historial)
    tss = sorted(ciclos)
    L = []
    W = L.append
    W("=" * 74)
    W("VERIFICACION DE CAMBIOS DE CAMINO - Measurement {0} | Probe {1}".format(mid or "?", pid or "?"))
    W("=" * 74)
    W("Historial: {0} lineas, {1} ciclos con respuesta ({2} -> {3})".format(
        n_lin, len(tss), tss[0] if tss else "-", tss[-1] if tss else "-"))

    # ---- firmas por ciclo
    firma = {}
    for ts in tss:
        h = ciclos[ts]
        firma[ts] = (max(h), aspath(h))
    longs = Counter(f[0] for f in firma.values())
    paths = Counter(f[1] for f in firma.values())
    W("Longitudes observadas (saltos: ciclos): " + ", ".join(
        "{0}: {1}".format(k, v) for k, v in sorted(longs.items(), key=lambda x: -x[1])[:8]))
    W("AS-paths distintos (colapsados): {0}".format(len(paths)))
    for p, n in paths.most_common(5):
        W("  {0:>6} ciclos  {1}".format(n, " > ".join(p) if p else "(sin ASN)"))

    # ---- transiciones reales entre ciclos consecutivos
    clase = {}       # ts -> clase del cambio respecto del ciclo anterior (solo ciclos con cambio)
    ejemplos = defaultdict(list)
    prev = None
    for ts in tss:
        cur = firma[ts]
        if prev is not None:
            lg = cur[0] != prev[0]
            ap_ = cur[1] != prev[1]
            if lg and ap_:
                k = "AS-path y longitud"
            elif ap_:
                k = "solo AS-path (misma longitud)"
            elif lg:
                k = "solo longitud (mismo AS-path)"
            else:
                k = None
            if k:
                clase[ts] = k
                if len(ejemplos[k]) < 3:
                    ejemplos[k].append(ts)
        prev = cur
    ORDEN = ("AS-path y longitud", "solo AS-path (misma longitud)", "solo longitud (mismo AS-path)")
    real = Counter(clase.values())

    # ---- COMPROBACION 1
    cats = None
    A = Lc = both = None
    W("")
    W("-" * 74)
    W("COMPROBACION 1 - Eventos del extractor (ciclos)")
    W("-" * 74)
    if ev_path:
        cats = leer_eventos(ev_path)
        Lc = set(t for t, c in cats.items() if c["cambio_longitud"])
        A = set(t for t, c in cats.items() if c["cambio_ip_asn"])
        both = Lc & A
        lineas = sum(cats[t]["cambio_ip_asn"] for t in A)
        W("Archivo: {0}".format(os.path.basename(ev_path)))
        W("  Ciclos con cambio_longitud        : {0}".format(len(Lc)))
        W("  Ciclos con cambio_ip_asn          : {0}  (lineas: {1})".format(len(A), lineas))
        W("    ambos en el mismo ciclo         : {0}".format(len(both)))
        W("    solo cambio_longitud            : {0}".format(len(Lc - A)))
        W("    solo cambio_ip_asn              : {0}".format(len(A - Lc)))
        W("  Lineas de cambio_ip_asn por ciclo : {0}".format(
            dict(sorted(Counter(cats[t]["cambio_ip_asn"] for t in A).items()))))
    else:
        W("  (sin archivo de eventos: comprobacion omitida; use --eventos)")

    # ---- COMPROBACION 2
    W("")
    W("-" * 74)
    W("COMPROBACION 2 - Cambios de camino reales (AS-path colapsado entre ciclos consecutivos)")
    W("-" * 74)
    for k in ORDEN:
        W("  {0:<32}{1:>6}   ejemplos: {2}".format(k, real[k], ", ".join(ejemplos[k]) if ejemplos[k] else "-"))
    W("  {0:<32}{1:>6}".format("TOTAL ciclos con cambio de camino", sum(real.values())))

    # ---- COMPROBACION 3
    runs = []   # [longitud, ciclos, inicio]
    for ts in tss:
        lg = firma[ts][0]
        if not runs or runs[-1][0] != lg:
            runs.append([lg, 0, ts])
        runs[-1][1] += 1
    P = a.persistencia
    sost = [r for r in runs if r[1] >= P]
    corto1 = [r for r in runs if r[1] == 1]
    corto23 = [r for r in runs if 2 <= r[1] < P]
    W("")
    W("-" * 74)
    W("COMPROBACION 3 - Tramos consecutivos con la misma longitud")
    W("-" * 74)
    W("  Tramos: {0} (transiciones: {1})".format(len(runs), max(len(runs) - 1, 0)))
    W("    de 1 ciclo (parpadeo)           : {0}".format(len(corto1)))
    W("    de 2 a {0} ciclos (transitorios)  : {1}".format(P - 1, len(corto23)))
    W("    de >= {0} ciclos (sostenidos)     : {1}".format(P, len(sost)))
    W("  Ciclos en tramos sostenidos: {0} de {1} ({2})".format(
        sum(r[1] for r in sost), len(tss), pct(sum(r[1] for r in sost), len(tss))))
    dur = defaultdict(list)
    for r in sost:
        dur[r[0]].append(r[1])
    W("  Por longitud (tramos sostenidos): " + "; ".join(
        "{0} saltos: {1} tramos, {2} ciclos (mediana {3})".format(
            k, len(v), sum(v), sorted(v)[len(v) // 2]) for k, v in sorted(dur.items())))
    W("  Tramos sostenidos (longitud, ciclos, inicio):")
    for h, n, t in sost[:40]:
        W("    {0:>2} saltos  {1:>6} ciclos  desde {2}".format(h, n, t))
    if len(sost) > 40:
        W("    ... ({0} tramos mas; ver el CSV)".format(len(sost) - 40))

    # ---- CRUCE
    W("")
    W("-" * 74)
    W("CRUCE 1 x 2 - Eventos del extractor vs cambio de camino real (por ciclo)")
    W("-" * 74)
    cruce = None
    if cats is not None:
        en_hist = set(tss)
        fuera = [t for t in A if t not in en_hist]
        reales_ts = set(clase)
        asr = set(t for t, k in clase.items() if k != "solo longitud (mismo AS-path)")
        ip_sin_camino = [t for t in A if t in en_hist and t not in reales_ts]
        ip_con_aspath = [t for t in A if t in asr]
        ip_solo_long = [t for t in A if clase.get(t) == "solo longitud (mismo AS-path)"]
        long_sin_real = [t for t in Lc if t in en_hist and t not in reales_ts]
        real_sin_long = [t for t in reales_ts if t not in Lc and clase[t] != "solo AS-path (misma longitud)"]
        cruce = dict(ip_sin_camino=len(ip_sin_camino), ip_con_aspath=len(ip_con_aspath),
                     ip_solo_long=len(ip_solo_long), long_sin_real=len(long_sin_real),
                     real_sin_long=len(real_sin_long))
        W("  Ciclos con cambio_ip_asn ........................... {0}".format(len(A)))
        W("    - coinciden con cambio de AS-path real ........... {0}".format(len(ip_con_aspath)))
        W("    - coinciden con cambio solo de longitud .......... {0}   (desplazamiento posicional)".format(len(ip_solo_long)))
        W("    - SIN cambio de camino real ...................... {0}".format(len(ip_sin_camino)))
        W("    - fuera del historial (timestamp no encontrado) .. {0}".format(len(fuera)))
        W("  Ciclos con cambio_longitud sin cambio real ......... {0}   (debe ser 0)".format(len(long_sin_real)))
        W("  Cambios de longitud reales sin evento cambio_longitud: {0}   (debe ser 0)".format(len(real_sin_long)))
        W("  Cambios de AS-path reales SIN cambio_ip_asn ........ {0}   (debe ser 0: un cambio real de ASN lo detecta la regla posicional)".format(
            len([t for t in asr if t not in A])))
    else:
        W("  (omitido: requiere archivo de eventos)")

    # ---- LECTURA
    W("")
    W("-" * 74)
    W("LECTURA")
    W("-" * 74)
    n_as = real["AS-path y longitud"] + real["solo AS-path (misma longitud)"]
    n_lo = real["solo longitud (mismo AS-path)"]
    if n_as == 0 and n_lo == 0:
        W("  * Camino estable: no hay cambios de AS-path ni de longitud en el periodo.")
    elif n_as == 0:
        W("  * NO hay cambios de AS-path: los {0} cambios de longitud ocurren dentro de los mismos AS (insercion/".format(n_lo))
        W("    remocion de saltos en un AS, no cambio de proveedor ni de ruta BGP).")
    else:
        W("  * Hay {0} ciclos con cambio de AS-path (cambio de ruta a nivel de AS), {1} de solo longitud.".format(n_as, n_lo))
    if cruce is not None:
        if cruce["ip_solo_long"] and n_as == 0:
            W("  * Los {0} ciclos con cambio_ip_asn son un ARTEFACTO de la comparacion posicional: coinciden con cambios".format(
                cruce["ip_solo_long"]))
            W("    de longitud sin cambio de AS-path (al insertar/quitar un salto se desplazan los indices).")
        elif cruce["ip_solo_long"]:
            W("  * {0} ciclos con cambio_ip_asn coinciden solo con cambio de longitud (artefacto posicional); {1} son cambios".format(
                cruce["ip_solo_long"], cruce["ip_con_aspath"]))
            W("    de AS-path reales.")
        if cruce["ip_sin_camino"]:
            W("  * {0} ciclos con cambio_ip_asn SIN cambio de camino real: revisar (ASN que alterna en un salto".format(cruce["ip_sin_camino"]))
            W("    sin cambiar el AS-path colapsado, o ciclos fuera de secuencia).")
        if cruce["long_sin_real"] or cruce["real_sin_long"]:
            W("  * INCONSISTENCIA: hay diferencias entre cambio_longitud y la longitud medida del historial; revisar la regla.")
        else:
            W("  * cambio_longitud del extractor y longitud del historial coinciden ciclo a ciclo.")
    cor = len(corto1) + len(corto23)
    W("  * De {0} transiciones de longitud, {1} tramos son transitorios (< {2} ciclos) y {3} sostenidos:".format(
        max(len(runs) - 1, 0), cor, P, len(sost)))
    W("    para contar cambios de camino conviene usar los tramos sostenidos, no los ciclos con evento.")
    W("  * Cambios sostenidos entre tramos de distinta longitud: {0} (ver pelt_degradacion.py para su efecto en RTT).".format(
        max(len(sost) - 1, 0)))

    texto = "\n".join(L) + "\n"
    print(texto)
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)
    suf = "_measurement{0}_probe{1}".format(mid, pid) if mid else ""
    out = os.path.join(a.output_dir, "verificacion_caminos{0}.txt".format(suf))
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(texto if sys.version_info[0] >= 3 else texto.decode("utf-8"))
    print("Guardado en: {0}".format(out))


if __name__ == "__main__":
    main()
