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
import ipaddress
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
    ips = defaultdict(dict)
    rtts = defaultdict(dict)
    n = 0
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            n += 1
            ip = (r.get("ip") or "").strip()
            if not ip:
                continue
            try:
                ts = r["timestamp"].strip()
                hop = int(r["hop"])
            except (ValueError, TypeError, KeyError):
                continue
            ciclos[ts][hop] = (r.get("asn") or "")
            ips[ts][hop] = ip
            try:
                rtts[ts][hop] = float(r.get("rtt_ms"))
            except (ValueError, TypeError):
                rtts[ts][hop] = None
    return ciclos, ips, rtts, n


def es_privada(ip):
    """IP privada / loopback / link-local: pertenece a la red local del probe, no al destino."""
    try:
        o = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return o.is_private or o.is_loopback or o.is_link_local


def clasificar_destino(tss, ciclos, ips, min_frac=0.02):
    """Decide en que ciclos se llego al destino, sin conocer dst_addr (el historial no lo trae).

    El destino puede ser un nombre de dominio que resuelve a varias IP (o a una IP distinta segun el ciclo), asi que
    no se compara con una unica IP. Un ciclo 'llego' si su ultimo salto responde, no es una IP privada y:
      - pertenece al ASN modal del ultimo salto, o
      - es una IP frecuente (>= min_frac de los ciclos) como ultimo salto, cuando no se conoce el ASN.
    Devuelve (alcanzados:set, info:dict)."""
    ult = {}
    for t in tss:
        k = max(ips[t])
        ult[t] = (k, ips[t][k], (ciclos[t].get(k) or "").strip())
    n = len(tss)
    cnt_ip = Counter(v[1] for v in ult.values() if not es_privada(v[1]))
    asn_cnt = Counter(v[2] for v in ult.values() if not es_privada(v[1]) and asn_valido(v[2]))
    asn_modal = asn_cnt.most_common(1)[0][0] if asn_cnt else None
    asn_de_ip = {}
    for ip in cnt_ip:
        c = Counter(v[2] for v in ult.values() if v[1] == ip and asn_valido(v[2]))
        asn_de_ip[ip] = c.most_common(1)[0][0] if c else None
    frecuentes = set(ip for ip, c in cnt_ip.items()
                     if c >= max(1, min_frac * n) and (asn_modal is None or asn_de_ip[ip] in (None, asn_modal)))
    ok = set()
    for t, (k, ip, asn) in ult.items():
        if es_privada(ip):
            continue
        if (asn_modal is not None and asn == asn_modal) or ip in frecuentes:
            ok.add(t)
    return ok, dict(ult=ult, asn_modal=asn_modal, frecuentes=frecuentes)


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
    ap.add_argument("--max-hops", type=int, default=None,
                    help="Valor de --max-hops usado al extraer (default del extractor: 15); si la longitud maxima observada "
                         "lo iguala, las rutas pueden estar truncadas")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--sin-html", action="store_true",
                    help="No generar el visor HTML (caminos_measurement<M>_probe<P>.html)")
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

    ciclos, ips, rtts, n_lin = leer_historial(a.historial)
    todos = sorted(ciclos)
    if not todos:
        sys.stderr.write("El historial no tiene ciclos con respuesta.\n")
        sys.exit(1)
    # Ciclos en los que no se llego al destino (ultimo salto privado / fuera del ASN del destino): no son cambios de
    # camino, son perdida de conectividad. Se excluyen del analisis de caminos y se informan aparte.
    alcanzados, info_dest = clasificar_destino(todos, ciclos, ips)
    no_alc = [t for t in todos if t not in alcanzados]
    tss = [t for t in todos if t in alcanzados]
    if not tss:   # nada llega al destino: no tiene sentido excluir todo
        tss, no_alc, alcanzados = list(todos), [], set(todos)
    idx_todos = dict((t, i) for i, t in enumerate(todos))
    # Ciclos cuyos eventos del extractor no son confiables: el propio ciclo sin destino y el siguiente
    # (el extractor lo compara posicionalmente contra un ciclo roto).
    adj_na = set(no_alc)
    for t in no_alc:
        i = idx_todos[t]
        if i + 1 < len(todos):
            adj_na.add(todos[i + 1])
    L = []
    W = L.append
    W("=" * 74)
    W("VERIFICACION DE CAMBIOS DE CAMINO - Measurement {0} | Probe {1}".format(mid or "?", pid or "?"))
    W("=" * 74)
    W("Historial: {0} lineas, {1} ciclos con respuesta ({2} -> {3})".format(
        n_lin, len(todos), todos[0], todos[-1]))
    W("Ciclos con destino NO alcanzado (excluidos del analisis de caminos): {0} ({1}) - ver seccion DESTINO NO ALCANZADO".format(
        len(no_alc), pct(len(no_alc), len(todos))))

    # ---- firmas por ciclo
    firma = {}
    for ts in tss:
        h = ciclos[ts]
        firma[ts] = (max(h), aspath(h))
    longs = Counter(f[0] for f in firma.values())
    paths = Counter(f[1] for f in firma.values())
    W("Longitudes observadas (saltos: ciclos): " + ", ".join(
        "{0}: {1}".format(k, v) for k, v in sorted(longs.items(), key=lambda x: -x[1])[:8]))
    if longs:
        mx = max(longs)
        modal_l = longs.most_common(1)[0][0]
        trunc = False
        if a.max_hops is not None and mx >= a.max_hops:
            W("[AVISO] la longitud maxima observada ({0}) iguala --max-hops ({1}): las rutas mas largas pueden estar".format(mx, a.max_hops))
            W("        truncadas. Re-extraer con un --max-hops mayor (p. ej. 30) antes de interpretar longitudes ni destino.")
            trunc = True
        elif modal_l == mx and mx >= 10 and longs[mx] > 0.5 * len(tss):
            W("[AVISO] la longitud modal ({0}) es tambien la maxima observada: patron tipico de truncamiento por el limite".format(mx))
            W("        de saltos del extractor (--max-hops, default 15). Pasar --max-hops con el valor usado o re-extraer con uno mayor.")
            trunc = True
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
    ev_L_all, ev_A_all = set(), set()
    W("")
    W("-" * 74)
    W("COMPROBACION 1 - Eventos del extractor (ciclos)")
    W("-" * 74)
    if ev_path:
        cats = leer_eventos(ev_path)
        Lc = set(t for t, c in cats.items() if c["cambio_longitud"])
        A = set(t for t, c in cats.items() if c["cambio_ip_asn"])
        both = Lc & A
        ev_L_all, ev_A_all = set(Lc), set(A)
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
    modal_run = longs.most_common(1)[0][0] if longs else None
    no_mod = [r for r in sost if r[0] != modal_run]
    W("  Tramos sostenidos de longitud distinta de la modal ({0} saltos): {1}".format(modal_run, len(no_mod)))
    for h, n, t in no_mod:
        W("    {0:>2} saltos  {1:>6} ciclos  desde {2}".format(h, n, t))
    mod = [r for r in sost if r[0] == modal_run]
    W("  Tramos sostenidos de longitud modal: {0} (primeros 15; todos en el CSV)".format(len(mod)))
    for h, n, t in mod[:15]:
        W("    {0:>2} saltos  {1:>6} ciclos  desde {2}".format(h, n, t))
    if len(mod) > 15:
        W("    ... ({0} tramos mas)".format(len(mod) - 15))

    # ---- BLOQUES: tramos sostenidos de igual longitud separados solo por transitorios
    from datetime import datetime
    seq = []   # [longitud, idx_ini, idx_fin]
    for i, ts in enumerate(tss):
        lg = firma[ts][0]
        if seq and seq[-1][0] == lg:
            seq[-1][2] = i
        else:
            seq.append([lg, i, i])
    sust_idx = [r for r in seq if r[2] - r[1] + 1 >= P]
    bloques = []
    for r in sust_idx:
        if bloques and bloques[-1][0] == r[0]:
            bloques[-1][2] = r[2]
        else:
            bloques.append(list(r))
    dts = sorted((datetime.strptime(todos[i + 1], "%Y-%m-%d %H:%M:%S") -
                  datetime.strptime(todos[i], "%Y-%m-%d %H:%M:%S")).total_seconds() for i in range(len(todos) - 1))
    cad_s = dts[len(dts) // 2] if dts else 0.0
    W("")
    W("  BLOQUES (tramos sostenidos de igual longitud unidos cuando los separan solo tramos transitorios):")
    W("    Cadencia mediana: {0:.0f} s. Bloques: {1}; cambios entre bloques: {2}".format(
        cad_s, len(bloques), max(len(bloques) - 1, 0)))
    W("    {0:>9} {1:>20} {2:>20} {3:>9} {4:>8} {5:>10}".format("longitud", "inicio", "fin", "dur(h)", "ciclos", "fuera"))
    modal_b = longs.most_common(1)[0][0] if longs else None
    mostrados = 0
    for lg, i0, i1 in bloques:
        if lg == modal_b and len(bloques) > 12:
            continue
        n = i1 - i0 + 1
        fuera = sum(1 for i in range(i0, i1 + 1) if firma[tss[i]][0] != lg)
        W("    {0:>6} sal {1:>20} {2:>20} {3:>9.1f} {4:>8} {5:>10}".format(
            lg, tss[i0], tss[i1], n * cad_s / 3600.0, n, fuera))
        mostrados += 1
    if mostrados < len(bloques):
        W("    (bloques de longitud modal omitidos: {0}; 'fuera' = ciclos del bloque con otra longitud)".format(len(bloques) - mostrados))
    cambios_bloque = max(len(bloques) - 1, 0)

    # ---- CRUCE
    W("")
    W("-" * 74)
    W("CRUCE 1 x 2 - Eventos del extractor vs cambio de camino real (por ciclo)")
    W("-" * 74)
    cruce = None
    if cats is not None:
        # eventos del ciclo sin destino y del siguiente: no se cruzan (se comparan contra un ciclo roto)
        A_na, L_na = A & adj_na, Lc & adj_na
        A, Lc = A - adj_na, Lc - adj_na
        if adj_na:
            W("  (Se excluyen del cruce los ciclos sin destino y el siguiente: cambio_ip_asn {0}, cambio_longitud {1})".format(
                len(A_na), len(L_na)))
        en_hist = set(tss)
        fuera = [t for t in A if t not in en_hist]
        reales_ts = set(t for t in clase if t not in adj_na)
        asr = set(t for t, k in clase.items() if k != "solo longitud (mismo AS-path)" and t not in adj_na)
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

    # ---- DETALLE de los ciclos con cambio_ip_asn sin cambio de camino real
    detalle = None
    if cats is not None and cruce is not None and cruce["ip_sin_camino"]:
        W("")
        W("-" * 74)
        W("DETALLE - Ciclos con cambio_ip_asn SIN cambio de camino real (comparacion posicional con el ciclo anterior)")
        W("-" * 74)
        idx = dict((t, i) for i, t in enumerate(tss))
        tipos = Counter()
        valores = Counter()
        ej = []
        sin_rep = 0
        for t in sorted(x for x in A if x in idx and x not in clase):
            i = idx[t]
            if i == 0:
                continue
            pt = tss[i - 1]
            dif = []
            for k in sorted(set(ciclos[t]) & set(ciclos[pt])):
                a0, a1 = ciclos[pt][k].strip(), ciclos[t][k].strip()
                if a0 != a1:
                    v0, v1 = asn_valido(a0), asn_valido(a1)
                    tipo = ("ASN valido <-> sin ASN" if v0 != v1 else
                            "frontera entre ASN desplazada" if v0 and v1 else "formato de ASN vacio distinto")
                    dif.append((k, ips[pt][k], a0, ips[t][k], a1, tipo))
            if not dif:
                sin_rep += 1
            for d in dif:
                tipos[d[5]] += 1
                valores[(d[2] or "(vacio)", d[4] or "(vacio)")] += 1
            if dif and len(ej) < 8:
                ej.append((t, dif[0]))
        W("  Diferencias de ASN por salto en esos ciclos:")
        for k, v in tipos.most_common():
            W("    {0:<34}{1:>5}".format(k, v))
        if sin_rep:
            W("    {0:<34}{1:>5}   (el evento no se reproduce comparando con el ciclo anterior)".format(
                "ciclos sin diferencia reproducible", sin_rep))
        W("  Pares de ASN mas frecuentes (antes -> ahora):")
        for (x, y), v in valores.most_common(5):
            W("    {0:>5} x  {1} -> {2}".format(v, x, y))
        if ej:
            W("  Ejemplos (ciclo, salto, IP/ASN anterior -> IP/ASN actual):")
            for t, d in ej:
                W("    {0}  hop {1}: {2} [{3}] -> {4} [{5}]".format(
                    t, d[0], d[1], d[2] or "(vacio)", d[3], d[4] or "(vacio)"))
        detalle = tipos

    # ---- DESTINO ALCANZADO en ciclos de longitud no modal
    W("")
    W("-" * 74)
    W("DESTINO NO ALCANZADO (ultimo salto privado o fuera del destino)")
    W("-" * 74)
    ult = info_dest["ult"]
    if info_dest["asn_modal"] is not None:
        W("  Criterio: se llega al destino si el ultimo salto es una IP publica del ASN modal del ultimo salto (AS{0}).".format(
            info_dest["asn_modal"]))
    else:
        W("  Criterio (sin ASN en el historial): ultimo salto = IP publica frecuente como ultimo salto.")
    W("  IP(s) finales frecuentes: {0}".format(", ".join(sorted(info_dest["frecuentes"])) or "-"))
    W("  Ciclos sin destino: {0} de {1} ({2}); excluidos de las comprobaciones 2 y 3, de los bloques y del cruce.".format(
        len(no_alc), len(todos), pct(len(no_alc), len(todos))))
    if no_alc:
        W("  {0:<20}{1:>5}  {2:<30}{3:<9}{4:>10}".format("ciclo (UTC)", "salt", "ultimo salto (ip)", "asn", "rtt_ms"))
        for t in no_alc[:40]:
            k, ip, asn = ult[t]
            r = rtts[t].get(k)
            W("  {0:<20}{1:>5}  {2:<30}{3:<9}{4:>10}{5}".format(
                t, k, ip, asn or "-", "n/d" if r is None else "{0:.1f}".format(r), "  [PRIVADA]" if es_privada(ip) else ""))
        if len(no_alc) > 40:
            W("  ... ({0} ciclos mas)".format(len(no_alc) - 40))
        W("  Un ultimo salto privado (RFC1918) es la red local del probe: perdida de conectividad, no un cambio de ruta.")
        W("  Revisar si el ciclo siguiente cambia de camino (recambio de ruta con perdida durante la convergencia).")

    # ---- LECTURA
    W("")
    W("-" * 74)
    W("LECTURA")
    W("-" * 74)
    if no_alc:
        W("  * {0} ciclos no llegaron al destino ({1} con ultimo salto privado): son perdidas de conectividad, no cambios de".format(
            len(no_alc), sum(1 for t in no_alc if es_privada(ult[t][1]))))
        W("    camino; se excluyeron de los conteos de caminos, tramos y bloques.")
    n_as = real["AS-path y longitud"] + real["solo AS-path (misma longitud)"]
    n_lo = real["solo longitud (mismo AS-path)"]
    if n_as == 0 and n_lo == 0:
        W("  * Camino estable: no hay cambios de AS-path ni de longitud en el periodo.")
    elif n_as == 0:
        W("  * NO hay cambios de AS-path: los {0} cambios de longitud ocurren dentro de los mismos AS (insercion/".format(n_lo))
        W("    remocion de saltos en un AS, no cambio de proveedor ni de ruta BGP). Antes de interpretarlos como cambio de")
        W("    ruta, descartar antes que sean perdidas de conectividad (ya excluidas: ver DESTINO NO ALCANZADO).")
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
    cambios_sost = cambios_bloque
    W("  * Cambios de camino sostenidos (entre bloques de distinta longitud): {0}.".format(cambios_sost))
    if len(bloques) > 1 and len(set(b[0] for b in bloques)) == 1:
        W("    Los tramos sostenidos son todos de la misma longitud ({0} saltos): las caidas a otra longitud son".format(sost[0][0]))
        W("    transitorias, asi que el camino es estable en el periodo.")
    if detalle:
        mayor = detalle.most_common(1)[0]
        W("  * Los cambio_ip_asn sin camino real son principalmente: '{0}' ({1} saltos).".format(mayor[0], mayor[1]))

    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)
    suf0 = "_measurement{0}_probe{1}".format(mid, pid) if mid else ""
    csvp = os.path.join(a.output_dir, "tramos_longitud{0}.csv".format(suf0))
    with open(csvp, "w", newline="") if sys.version_info[0] >= 3 else open(csvp, "wb") as fc:
        wr = csv.writer(fc)
        wr.writerow(["inicio", "longitud", "ciclos", "sostenido"])
        for h, n, t in runs:
            wr.writerow([t, h, n, int(n >= P)])
    W("")
    W("Tramos exportados a: {0}".format(csvp))
    if not a.sin_html:
        htmlp = os.path.join(a.output_dir, "caminos{0}.html".format(suf0))
        datos = datos_visor(mid, pid, cad_s, todos, ciclos, ips, rtts, alcanzados, info_dest,
                            [(lg, idx_todos[tss[i0]], idx_todos[tss[i1]]) for lg, i0, i1 in bloques],
                            sorted(idx_todos[t] for t in ev_L_all if t in idx_todos),
                            sorted(idx_todos[t] for t in ev_A_all if t in idx_todos), P)
        with io.open(htmlp, "w", encoding="utf-8") as fh:
            fh.write(HTML_TEMPLATE.replace("__DATA__", datos))
        W("Visor HTML: {0}  (abrir en el navegador; no necesita conexion)".format(htmlp))
    texto = "\n".join(L) + "\n"
    print(texto)
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)
    suf = "_measurement{0}_probe{1}".format(mid, pid) if mid else ""
    out = os.path.join(a.output_dir, "verificacion_caminos{0}.txt".format(suf))
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(texto if sys.version_info[0] >= 3 else texto.decode("utf-8"))
    print("Guardado en: {0}".format(out))


def datos_visor(mid, pid, cad_s, todos, ciclos, ips, rtts, alcanzados, info_dest, bloques, ev_l, ev_a, persist):
    """Datos del visor HTML en JSON compacto: tabla de IP, ciclos (epoch, ip por salto, rtt por salto, bandera)."""
    import json
    import time
    import calendar
    tabla, pos = [], {}
    asn_de = defaultdict(Counter)
    for t in todos:
        for k, ip in ips[t].items():
            asn_de[ip][(ciclos[t].get(k) or "").strip()] += 1
    for ip in sorted(asn_de, key=lambda x: -sum(asn_de[x].values())):
        c = asn_de[ip]
        validos = [(n, a) for a, n in c.items() if asn_valido(a)]
        asn = max(validos)[1] if validos else ""
        pos[ip] = len(tabla)
        tabla.append([ip, asn])
    cs = []
    for t in todos:
        n = max(ips[t])
        sal = [pos[ips[t][k]] if k in ips[t] else -1 for k in range(1, n + 1)]
        rt = [None if (k not in ips[t] or rtts[t].get(k) is None) else round(rtts[t][k], 2) for k in range(1, n + 1)]
        ep = calendar.timegm(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
        cs.append([ep, sal, rt, 0 if t in alcanzados else 1])
    meta = dict(m=mid or "?", p=pid or "?", cad=int(cad_s or 600), persist=persist,
                asn=info_dest.get("asn_modal") or "", nfail=sum(1 for c in cs if c[3] & 1),
                gen=time.strftime("%Y-%m-%d %H:%M"))
    d = dict(meta=meta, ips=tabla, c=cs, blocks=[list(b) for b in bloques], evL=ev_l, evA=ev_a)
    return json.dumps(d, separators=(",", ":")).replace("</", "<\\/")


HTML_TEMPLATE = r'''<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visor de caminos</title>
<style>
:root{
  --bg:#f4f6f9;--panel:#ffffff;--ink:#1b2430;--mut:#667386;--line:#d8dee7;--accent:#2563eb;
  --src:#2563eb;--tgt:#16a34a;--fail:#dc2626;--sel:#f59e0b;--node:#7a8799;--edge:#9aa5b4;--modal:#64748b;--ev:#7c3aed;
  --chip:#eef2f7;--hover:#e8eef8;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#11151b;--panel:#1a2029;--ink:#e6ebf2;--mut:#93a0b3;--line:#2c3542;--accent:#6ea0ff;
    --src:#6ea0ff;--tgt:#34d399;--fail:#f87171;--sel:#fbbf24;--node:#8896aa;--edge:#5b6778;--modal:#8a99ad;--ev:#a78bfa;
    --chip:#232c38;--hover:#222c3a;
  }
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:16px;margin:0;font-weight:650}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;color:var(--mut);font-size:12px}
.chip b{color:var(--ink);font-weight:600}
main{padding:12px 16px;display:grid;gap:12px;grid-template-columns:minmax(0,1fr) 390px;align-items:start}
@media (max-width:1000px){main{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding:8px 10px;border-bottom:1px solid var(--line)}
.grp{display:flex;align-items:center;gap:4px}
.grp>label{color:var(--mut);font-size:12px;margin-right:2px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--ink);padding:4px 9px;cursor:pointer;font:inherit}
.seg button+button{border-left:1px solid var(--line)}
.seg button.on{background:var(--accent);color:#fff}
button.b,select,input[type=text]{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:4px 9px}
button.b{cursor:pointer}
button.b:hover,.seg button:hover:not(.on){background:var(--hover)}
input[type=text]{width:190px}
input[type=range]{accent-color:var(--accent)}
.count{margin-left:auto;color:var(--mut)}
#tlwrap{padding:8px 10px 6px}
#tl{width:100%;height:170px;display:block;cursor:crosshair;border-radius:6px}
.tlbar{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin-top:6px}
.tlbar input[type=range]{flex:1;min-width:160px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;color:var(--mut);font-size:12px;padding:0 10px 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.legend s{display:inline-block;width:12px;height:6px;margin-right:5px;vertical-align:1px;text-decoration:none}
#stage{position:relative;height:460px;overflow:hidden}
#svg{width:100%;height:100%;display:block;cursor:grab;user-select:none}
#svg.pan{cursor:grabbing}
.tip{position:fixed;z-index:20;max-width:320px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;
  box-shadow:0 4px 18px rgba(0,0,0,.18);padding:8px 10px;font-size:12px;pointer-events:none}
.tip.pin{pointer-events:auto}
.tip h4{margin:0 0 4px;font-size:13px;word-break:break-all}
.tip .k{color:var(--mut)}
.tip .x{float:right;cursor:pointer;color:var(--mut);margin-left:8px}
.zoombtns{position:absolute;right:10px;top:10px;display:flex;flex-direction:column;gap:4px}
.hint{position:absolute;left:10px;bottom:8px;color:var(--mut);font-size:11px}
aside{display:flex;flex-direction:column;gap:12px}
.tabs{display:flex;border-bottom:1px solid var(--line)}
.tabs button{flex:1;border:0;background:transparent;color:var(--mut);padding:8px 4px;cursor:pointer;font:inherit;border-bottom:2px solid transparent}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.pane{padding:10px;max-height:560px;overflow:auto}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:3px 6px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:500;font-size:12px;position:sticky;top:0;background:var(--panel)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
#pane td:last-child{white-space:normal;word-break:break-all}
tr.click{cursor:pointer}
tr.click:hover{background:var(--hover)}
.chg{color:var(--sel);font-weight:600}
.bad{color:var(--fail);font-weight:600}
.mut{color:var(--mut)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin-bottom:8px}
.kv span:nth-child(odd){color:var(--mut)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.helpbtn{margin-left:auto}
#help{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.45);display:none;align-items:flex-start;justify-content:center;padding:4vh 12px;overflow:auto}
#help.open{display:flex}
#help .box{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:12px;max-width:860px;width:100%;padding:18px 22px 22px;box-shadow:0 12px 40px rgba(0,0,0,.35)}
#help h2{margin:0 0 4px;font-size:17px}
#help h3{margin:18px 0 6px;font-size:14px}
#help p,#help li{margin:4px 0}
#help ul{margin:4px 0 4px 18px;padding:0}
#help table{margin:6px 0}
#help td,#help th{white-space:normal;vertical-align:top}
#help .top{display:flex;align-items:center;gap:12px}
#help .top .b{margin-left:auto}
#help code{background:var(--chip);border-radius:4px;padding:0 4px}
footer{padding:4px 16px 16px;color:var(--mut);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>Visor de caminos</h1>
  <span class="chip">Measurement <b id="hM"></b></span>
  <span class="chip">Probe <b id="hP"></b></span>
  <span class="chip"><b id="hN"></b> ciclos</span>
  <span class="chip" id="hRange"></span>
  <span class="chip" id="hDest"></span>
  <button class="b helpbtn" id="bHelp" title="Cómo interpretar el visor (tecla ?)">? Ayuda</button>
</header>
<main>
  <section class="card">
    <div class="toolbar">
      <div class="grp"><label>Vista</label><span class="seg" id="sgView"><button data-v="ip">IP</button><button data-v="asn">ASN</button></span></div>
      <div class="grp"><label>Distancia</label><span class="seg" id="sgDist"><button data-v="hops">SALTOS</button><button data-v="rtt">RTT</button></span></div>
      <div class="grp"><label>Disposición</label><span class="seg" id="sgLay"><button data-v="h">Horizontal</button><button data-v="v">Vertical</button></span></div>
      <div class="grp"><label>Ciclos</label><span class="seg" id="sgSucc"><button data-v="all">Todos</button><button data-v="ok">Con destino</button><button data-v="fail">Sin destino</button></span></div>
      <div class="grp"><label><input type="checkbox" id="cIP" checked> IP</label><label><input type="checkbox" id="cASN" checked> ASN</label></div>
    </div>
    <div class="toolbar">
      <div class="grp"><label>Filtro</label><input type="text" id="q" placeholder="IP, ASN… (varios separados por espacio)">
        <select id="qm"><option value="inc">incluir</option><option value="exc">excluir</option></select></div>
      <div class="grp"><label>RTT destino</label><input type="range" id="rLo" min="0" max="1000" value="0"><input type="range" id="rHi" min="0" max="1000" value="1000"><span id="rTxt" class="mut"></span></div>
      <div class="grp"><button class="b" id="bExp">Exportar ▾</button>
        <button class="b" id="bLink">Copiar enlace</button></div>
      <span class="count" id="count"></span>
    </div>
    <div id="tlwrap">
      <canvas id="tl"></canvas>
      <div class="tlbar">
        <button class="b" id="bPrev" title="Ciclo anterior (←)">◀</button>
        <button class="b" id="bNext" title="Ciclo siguiente (→)">▶</button>
        <input type="range" id="slider" min="0" value="0">
        <input type="text" id="goto" style="width:160px" placeholder="AAAA-MM-DD HH:MM">
        <button class="b" id="bGo">Ir</button>
        <button class="b" id="bZoomR" title="Ampliar la línea de tiempo al rango elegido">Zoom al rango</button>
        <button class="b" id="bAll">Ver todo</button>
      </div>
    </div>
    <div class="legend" id="legendTl"></div>
    <div id="stage">
      <svg id="svg" xmlns="http://www.w3.org/2000/svg"><g id="vp"></g></svg>
      <div class="zoombtns"><button class="b" id="zIn">+</button><button class="b" id="zOut">−</button><button class="b" id="zFit" title="Ajustar">⤢</button></div>
      <div class="hint">Arrastrar: mover · Rueda: zoom · Hover: detalle · Clic en un nodo: fijar el cuadro</div>
    </div>
    <div class="legend">
      <span><i style="background:var(--src)"></i>Sonda</span>
      <span><i style="background:var(--tgt)"></i>Destino</span>
      <span><i style="background:var(--fail)"></i>Último salto antes de perder el destino</span>
      <span><i style="background:var(--node)"></i>Salto intermedio</span>
      <span><i style="background:none;border:1.5px dashed var(--node)"></i>Sin respuesta (*)</span>
      <span><i style="background:none;border:2px solid var(--sel)"></i>Ciclo seleccionado</span>
    </div>
  </section>
  <aside>
    <section class="card">
      <div class="tabs" id="tabs">
        <button data-t="det" class="on">Ciclo</button><button data-t="blk">Bloques</button><button data-t="nod">Sin destino</button><button data-t="evt">Eventos</button>
      </div>
      <div class="pane" id="pane"></div>
    </section>
  </aside>
</main>
<footer id="foot"></footer>

<div id="help" role="dialog" aria-modal="true" aria-label="Ayuda">
<div class="box">
<div class="top"><h2>Cómo interpretar el visor</h2><button class="b" id="hClose">Cerrar ✕</button></div>
<p class="mut">Una sola sonda a lo largo del tiempo. Arriba, la línea de tiempo de todos los ciclos; en el centro, el grafo agregado de caminos del rango elegido; a la derecha, el detalle. Todas las horas son UTC.</p>

<h3>1. Línea de tiempo</h3>
<ul>
<li><b>Tira "saltos":</b> un trazo por ciclo, coloreado por la cantidad de saltos hasta el destino (gris = longitud modal). Rojo = ciclo sin destino.</li>
<li><b>Fila "AS-path":</b> color según la secuencia de ASN del ciclo. Si cambia de color hay un cambio real de camino a nivel de AS.</li>
<li><b>Curva "RTT al destino":</b> línea = mediana por columna; banda = mínimo y máximo. Un escalón sostenido es un cambio de régimen.</li>
<li><b>Marcas "eventos":</b> violeta = <code>cambio_longitud</code> del extractor; naranja = <code>cambio_ip_asn</code>. Hay muchas más marcas que bloques: la mayoría son parpadeos.</li>
<li><b>Interacción:</b> arrastrar = elegir <i>rango</i> (el grafo se recalcula con ese rango); clic = elegir <i>ciclo</i> (línea naranja); doble clic = volver al rango completo; rueda = zoom de la línea de tiempo; ◀ ▶ o flechas del teclado = ciclo anterior/siguiente (con Shift, de a 10); "Ir" = saltar a una fecha; "Zoom al rango" y "Ver todo" ajustan la ventana.</li>
</ul>

<h3>2. Grafo de caminos</h3>
<ul>
<li>Cada nodo es una IP (vista IP) o un ASN (vista ASN). Cada arista es un tramo que usaron los ciclos del rango; su grosor y el tamaño del nodo dependen de cuántos ciclos pasan por ahí. Si hay una sola línea, el camino fue estable; si se abre en ramas, hubo alternativas (ECMP o cambios de ruta).</li>
<li><b>Colores:</b> azul = sonda; verde = destino; rojo = último salto antes de perder el destino (es el último que respondió en algún ciclo fallido, no necesariamente un router con falla); gris = salto intermedio; círculo punteado = salto sin respuesta (*); contorno y línea naranja = camino del ciclo seleccionado.</li>
<li><b>Distancia:</b> SALTOS ubica los nodos por número de salto; RTT los ubica según su RTT mediano (un salto lejano en RTT se aleja en el gráfico). Útil para ver dónde "se gasta" la latencia. Un router intermedio puede tener RTT mayor que el siguiente porque prioriza poco las respuestas ICMP.</li>
<li><b>Disposición:</b> Horizontal o Vertical, solo cambia la orientación.</li>
<li><b>Hover</b> sobre un nodo: IP/ASN, posición, cuántos ciclos pasan, RTT mediano/mínimo/p95 y el RTT del ciclo seleccionado. <b>Clic</b> sobre un nodo fija el cuadro; clic en un espacio vacío los cierra. Arrastrar el fondo mueve el grafo y la rueda hace zoom; ⤢ ajusta.</li>
</ul>

<h3>3. Filtros (afectan al grafo, no a la línea de tiempo)</h3>
<table>
<tr><th>Control</th><th>Efecto</th></tr>
<tr><td>Ciclos: Todos / Con destino / Sin destino</td><td>Elige qué ciclos del rango entran al grafo. "Sin destino" deja solo los ciclos fallidos.</td></tr>
<tr><td>Filtro de texto</td><td>Una o varias IP o ASN separadas por espacio (por ejemplo <code>AS3356 62.67.27.10</code>). "incluir" deja solo los ciclos que pasan por alguno; "excluir" los quita.</td></tr>
<tr><td>RTT destino (dos controles)</td><td>Deja los ciclos cuyo RTT al último salto está entre el mínimo y el máximo. No afecta a los ciclos sin destino. Con el rango completo muestra <code>ALL</code>. Sirve para ver por dónde pasan solo los ciclos lentos o solo los rápidos y compararlos.</td></tr>
<tr><td>Etiquetas IP / ASN</td><td>Muestra u oculta los textos de los nodos.</td></tr>
</table>
<p>El contador "N ciclos en el grafo" indica cuántos ciclos quedaron después del rango y los filtros.</p>

<h3>4. Panel de la derecha</h3>
<p><b>Ciclo</b> es dinámico (sigue al ciclo seleccionado). <b>Bloques</b>, <b>Sin destino</b> y <b>Eventos</b> son listas calculadas una vez por el script; son fijas, pero cada fila es un acceso directo.</p>
<ul>
<li><b>Ciclo:</b> encabezado con fecha, longitud, destino, AS-path y bloque. La tabla muestra IP, ASN y RTT por salto y, en "vs ciclo anterior", lo que cambió: IP anterior en naranja, variación de RTT (naranja si es de 5 ms o más) y "salto nuevo". La comparación es por posición: si se inserta un salto, los siguientes aparecen cambiados; el cambio real está en el primero.</li>
<li><b>Bloques:</b> tramos sostenidos de igual longitud (≥ persistencia mínima de ciclos), unidos si solo los separan tramos transitorios. La cantidad de cambios de camino sostenidos es la de bloques menos uno. Clic: el rango pasa a ser ese bloque, así el grafo muestra solo sus caminos; haciendo clic en dos bloques seguidos se comparan sus grafos.</li>
<li><b>Sin destino:</b> ciclos cuyo último salto no es el destino. Un último salto con IP privada (RFC1918) es la red local de la sonda: pérdida de conectividad, no cambio de ruta. Clic: selecciona el ciclo y muestra ±20 ciclos; con ◀ ▶ se ve qué pasó antes y después.</li>
<li><b>Eventos:</b> ciclos con <code>cambio_longitud</code> según el extractor, con una marca si coinciden con un ciclo sin destino o un evento de IP/ASN. Compararlos con los bloques permite descartar los parpadeos: un evento que no está en el borde de un bloque es transitorio. Queda vacía si no se pasó el archivo de eventos.</li>
</ul>

<h3>5. Receta rápida</h3>
<ul>
<li>¿Cambió el camino? En <b>Bloques</b> mirá si hay más de un bloque; la fila AS-path de la línea de tiempo confirma si cambió de AS.</li>
<li>¿Dónde cambió? Clic en dos bloques consecutivos y buscá el nodo que aparece o desaparece; en <b>Ciclo</b> usá ◀ ▶ en el borde del bloque.</li>
<li>¿Empeoró el RTT? Mirá el escalón en la curva y repetí el análisis con el filtro RTT destino.</li>
<li>¿Hubo cortes? Pestaña <b>Sin destino</b>.</li>
</ul>

<h3>6. Compartir</h3>
<p>"Copiar enlace" guarda en la URL la vista, el rango, el ciclo y los filtros: abrir ese enlace reproduce lo que estás viendo. "Exportar" ofrece PNG, SVG, CSV o JSON de los ciclos que están en el grafo. Atajos: <code>?</code> abre esta ayuda y <code>Esc</code> la cierra.</p>
</div>
</div>

<script>
(function(){
"use strict";
var D = __DATA__;
var N = D.c.length;
function $(s){return document.querySelector(s);}
function pad(n){return (n<10?"0":"")+n;}
function fmt(t){return new Date(t*1000).toISOString().slice(0,19).replace("T"," ");}
function fmtS(t){var d=new Date(t*1000);return pad(d.getUTCMonth()+1)+"-"+pad(d.getUTCDate())+" "+pad(d.getUTCHours())+":"+pad(d.getUTCMinutes());}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c];});}
function med(a){if(!a.length)return null;var s=a.slice().sort(function(x,y){return x-y;});var n=s.length;return n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2;}
function quant(a,q){if(!a.length)return null;var s=a.slice().sort(function(x,y){return x-y;});return s[Math.min(s.length-1,Math.max(0,Math.floor(q*(s.length-1))))];}
function f1(x){return x==null?"n/d":(Math.abs(x)>=100?x.toFixed(0):x.toFixed(1))+" ms";}
function validAsn(a){if(!a)return false;a=String(a).toLowerCase();return a!=="unknown"&&a!=="na"&&a!=="none"&&a!=="nan"&&a!=="null"&&a!=="";}
function shortIp(ip){return ip.length>22?ip.slice(0,12)+"…"+ip.slice(-6):ip;}

// ---------- datos derivados
var T=[],FAIL=[],LEN=[],DR=[],HAY=[];
for(var i=0;i<N;i++){
  var c=D.c[i];T.push(c[0]);FAIL.push((c[3]&1)===1);LEN.push(c[1].length);
  DR.push(FAIL[i]?null:c[2][c[2].length-1]);
}
var ASNP=[],ASNPID=[];   // AS-path colapsado por ciclo
(function(){
  var cnt={},keys=[];
  for(var i=0;i<N;i++){
    var p=[],ips=D.c[i][1];
    for(var h=0;h<ips.length;h++){if(ips[h]<0)continue;var a=D.ips[ips[h]][1];if(!validAsn(a))continue;if(!p.length||p[p.length-1]!==a)p.push(a);}
    var s=p.join(" > ")||"(sin ASN)";ASNP.push(s);cnt[s]=(cnt[s]||0)+(FAIL[i]?0:1);
  }
  keys=Object.keys(cnt).sort(function(a,b){return cnt[b]-cnt[a];});
  var id={};keys.forEach(function(k,j){id[k]=j;});
  for(var i=0;i<N;i++)ASNPID.push(id[ASNP[i]]);
})();
var lenCnt={};for(var i=0;i<N;i++)if(!FAIL[i])lenCnt[LEN[i]]=(lenCnt[LEN[i]]||0)+1;
var MODAL=+Object.keys(lenCnt).sort(function(a,b){return lenCnt[b]-lenCnt[a];})[0];
var OTHER=Object.keys(lenCnt).map(Number).filter(function(x){return x!==MODAL;}).sort(function(a,b){return a-b;});
var PAL=["#d97706","#0ea5e9","#10b981","#c026d3","#8b5cf6","#84cc16","#14b8a6","#a16207","#ec4899","#0891b2"];
var C={};
function readColors(){var cs=getComputedStyle(document.documentElement);["ink","mut","line","accent","src","tgt","fail","sel","node","edge","bg","panel","modal","ev"].forEach(function(k){C[k]=cs.getPropertyValue("--"+k).trim();});}
function lenColor(l){if(l===MODAL)return C.modal;var j=OTHER.indexOf(l);return PAL[(j<0?0:j)%PAL.length];}
function pathColor(id){return id===0?C.modal:PAL[(id-1)%PAL.length];}
var RV=DR.filter(function(x){return x!=null;});
var RMIN=RV.length?Math.min.apply(null,RV):0, RMAX=RV.length?Math.max.apply(null,RV):1;
if(RMAX<=RMIN)RMAX=RMIN+1;
function hay(i){
  if(HAY[i]!==undefined)return HAY[i];
  var s=[],ips=D.c[i][1];
  for(var h=0;h<ips.length;h++){if(ips[h]<0)continue;var p=D.ips[ips[h]];s.push(p[0].toLowerCase());if(validAsn(p[1])){s.push("as"+p[1]);s.push(String(p[1]));}}
  HAY[i]=" "+s.join(" ")+" ";return HAY[i];
}

// ---------- estado
var S={sel:Math.max(0,N-1),r0:0,r1:N-1,v0:0,v1:N-1,view:"ip",dist:"hops",lay:"h",succ:"all",q:"",qm:"inc",rlo:RMIN,rhi:RMAX,lblIP:true,lblASN:true,tx:0,ty:0,k:1,tab:"det"};
var dirtyGraph=true,G=null,pinned=[];

function setStateFromHash(){
  var h=location.hash.replace(/^#/,"");if(!h)return;
  h.split("&").forEach(function(kv){
    var p=kv.split("=");var k=p[0],v=decodeURIComponent(p.slice(1).join("="));
    if(k==="v"&&(v==="ip"||v==="asn"))S.view=v;
    else if(k==="d"&&(v==="hops"||v==="rtt"))S.dist=v;
    else if(k==="l"&&(v==="h"||v==="v"))S.lay=v;
    else if(k==="s"&&(v==="all"||v==="ok"||v==="fail"))S.succ=v;
    else if(k==="sel"&&+v>=0&&+v<N)S.sel=+v;
    else if(k==="r"){var a=v.split("-").map(Number);if(a.length===2&&a[0]>=0&&a[1]<N&&a[0]<=a[1]){S.r0=a[0];S.r1=a[1];}}
    else if(k==="w"){var b=v.split("-").map(Number);if(b.length===2&&b[0]>=0&&b[1]<N&&b[0]<b[1]){S.v0=b[0];S.v1=b[1];}}
    else if(k==="q")S.q=v;
    else if(k==="qm"&&(v==="inc"||v==="exc"))S.qm=v;
    else if(k==="rtt"){var c=v.split("-").map(Number);if(c.length===2&&isFinite(c[0])&&isFinite(c[1])){S.rlo=c[0];S.rhi=c[1];}}
    else if(k==="li")S.lblIP=v==="1";
    else if(k==="la")S.lblASN=v==="1";
  });
}
var hashTimer=null;
function saveHash(){
  clearTimeout(hashTimer);
  hashTimer=setTimeout(function(){
    var a=["v="+S.view,"d="+S.dist,"l="+S.lay,"s="+S.succ,"sel="+S.sel,"r="+S.r0+"-"+S.r1,"w="+S.v0+"-"+S.v1,
      "q="+encodeURIComponent(S.q),"qm="+S.qm,"rtt="+S.rlo.toFixed(2)+"-"+S.rhi.toFixed(2),"li="+(S.lblIP?1:0),"la="+(S.lblASN?1:0)];
    try{history.replaceState(null,"","#"+a.join("&"));}catch(e){}
  },250);
}

// ---------- filtros
function terms(){return S.q.toLowerCase().split(/[\s,]+/).filter(Boolean);}
function rttActive(){return S.rlo>RMIN+1e-9||S.rhi<RMAX-1e-9;}
function passes(i,tm,ra){
  if(S.succ==="ok"&&FAIL[i])return false;
  if(S.succ==="fail"&&!FAIL[i])return false;
  if(ra&&!FAIL[i]&&DR[i]!=null&&(DR[i]<S.rlo||DR[i]>S.rhi))return false;
  if(tm.length){
    var h=hay(i),m=false;
    for(var j=0;j<tm.length;j++){if(h.indexOf(tm[j])>=0){m=true;break;}}
    if(S.qm==="inc"&&!m)return false;
    if(S.qm==="exc"&&m)return false;
  }
  return true;
}
function included(){
  var out=[],tm=terms(),ra=rttActive();
  for(var i=S.r0;i<=S.r1;i++)if(passes(i,tm,ra))out.push(i);
  return out;
}

// ---------- secuencia de nodos de un ciclo
function seqIP(i){
  var c=D.c[i],s=[{k:"S",layer:0}];
  for(var h=0;h<c[1].length;h++){
    var ip=c[1][h];
    if(ip<0)s.push({k:"*"+(h+1),layer:h+1,nul:true,rtt:null});
    else s.push({k:"i"+ip,layer:h+1,ip:ip,rtt:c[2][h]});
  }
  return s;
}
function seqASN(i){
  var c=D.c[i],n=c[1].length,as=[],h;
  for(h=0;h<n;h++){var ip=c[1][h];as.push(ip<0?"":(validAsn(D.ips[ip][1])?String(D.ips[ip][1]):""));}
  // huecos de ASN desconocido entre dos saltos del mismo ASN se absorben
  for(h=0;h<n;h++){
    if(as[h]!=="")continue;
    var a=h-1;while(a>=0&&as[a]==="")a--;
    var b=h+1;while(b<n&&as[b]==="")b++;
    if(a>=0&&b<n&&as[a]===as[b])as[h]=as[a];
  }
  var g=[],prev="S";
  for(h=0;h<n;h++){
    var key=as[h]!==""?"a"+as[h]:"?"+prev;
    var ip2=c[1][h],r=c[2][h];
    if(g.length&&g[g.length-1].k===key){var L=g[g.length-1];if(r!=null)L.rs.push(r);if(ip2>=0)L.ips.push(ip2);}
    else g.push({k:key,asn:as[h],rs:r!=null?[r]:[],ips:ip2>=0?[ip2]:[]});
    if(as[h]!=="")prev=as[h];
  }
  var s=[{k:"S",layer:0}];
  g.forEach(function(x,j){s.push({k:x.k,layer:j+1,asn:x.asn,rtt:med(x.rs),ips:x.ips,nul:x.asn===""});});
  return s;
}
function seqOf(i){return S.view==="asn"?seqASN(i):seqIP(i);}

// ---------- grafo agregado
function buildGraph(){
  var idx=included(),nodes={},edges={},tot=idx.length;
  idx.forEach(function(i){
    var sq=seqOf(i),last=sq[sq.length-1];
    for(var j=0;j<sq.length;j++){
      var s=sq[j],nd=nodes[s.k];
      if(!nd){nd={k:s.k,n:0,lay:{},rtts:[],ips:{},asn:s.asn,ip:s.ip,nul:!!s.nul,tgt:0,fail:0};nodes[s.k]=nd;}
      nd.n++;nd.lay[s.layer]=(nd.lay[s.layer]||0)+1;
      if(s.rtt!=null)nd.rtts.push(s.rtt);
      if(s.ips)s.ips.forEach(function(x){nd.ips[x]=(nd.ips[x]||0)+1;});
      if(j>0){var ek=sq[j-1].k+"|"+s.k,e=edges[ek];if(!e){e={a:sq[j-1].k,b:s.k,n:0};edges[ek]=e;}e.n++;}
    }
    if(last.k!=="S"){if(FAIL[i])nodes[last.k].fail++;else nodes[last.k].tgt++;}
  });
  Object.keys(nodes).forEach(function(k){
    var nd=nodes[k],best=-1,bl=0;
    for(var l in nd.lay){if(nd.lay[l]>best){best=nd.lay[l];bl=+l;}}
    nd.layer=bl;nd.rmed=med(nd.rtts);
    if(nd.ip!=null&&nd.asn==null)nd.asn=D.ips[nd.ip][1];
  });
  return {nodes:nodes,edges:Object.keys(edges).map(function(k){return edges[k];},{}),total:tot,idx:idx};
}
// posicion de los nodos del ciclo seleccionado cuando difiere de su salto modal (p.ej. un desvio con mas saltos)
function selOverride(g){
  var o={},sq=seqOf(S.sel);
  for(var j=1;j<sq.length;j++){var nd=g.nodes[sq[j].k];if(nd&&o[sq[j].k]===undefined&&nd.layer!==j)o[sq[j].k]=j;}
  return o;
}
function layoutGraph(g,over){
  over=over||{};
  var nodes=g.nodes,keys=Object.keys(nodes),SP=60,GAP=130;
  keys.forEach(function(k){nodes[k].eff=over[k]!==undefined?over[k]:nodes[k].layer;});
  var layers={};
  keys.forEach(function(k){var nd=nodes[k];(layers[nd.eff]=layers[nd.eff]||[]).push(nd);});
  var Ls=Object.keys(layers).map(Number).sort(function(a,b){return a-b;});
  var preds={},succs={};
  g.edges.forEach(function(e){(succs[e.a]=succs[e.a]||[]).push(e);(preds[e.b]=preds[e.b]||[]).push(e);});
  Ls.forEach(function(L){layers[L].sort(function(a,b){return b.n-a.n;});});
  function place(L,fwd){
    var arr=layers[L];
    arr.forEach(function(nd){
      var s=0,w=0,es=(fwd?preds:succs)[nd.k]||[];
      es.forEach(function(e){var o=nodes[fwd?e.a:e.b];if(o&&o.v!==undefined){s+=o.v*e.n;w+=e.n;}});
      nd.bary=w?s/w:null;
    });
    arr.sort(function(x,y){
      if(x.bary===null&&y.bary===null)return y.n-x.n;
      if(x.bary===null)return 1;if(y.bary===null)return -1;
      return (x.bary-y.bary)||(y.n-x.n);
    });
    var hi=0;arr.forEach(function(nd,j){if(nd.n>arr[hi].n)hi=j;});
    arr.forEach(function(nd,j){nd.v=(j-hi)*SP;});
  }
  Ls.forEach(function(L){place(L,true);});
  for(var r=0;r<2;r++){
    for(var a=Ls.length-1;a>=0;a--)place(Ls[a],false);
    for(var b=0;b<Ls.length;b++)place(Ls[b],true);
  }
  // coordenada a lo largo del camino
  if(S.dist==="hops"){
    keys.forEach(function(k){nodes[k].u=nodes[k].eff*GAP;});
  }else{
    var ms=keys.map(function(k){return nodes[k].rmed;}).filter(function(x){return x!=null&&x>=0;});
    var top=quant(ms,0.98)||1,px=760/Math.max(top,1);
    Ls.forEach(function(L){
      layers[L].forEach(function(nd){
        var mp=0;(preds[nd.k]||[]).forEach(function(e){var o=nodes[e.a];if(o&&o.u!==undefined&&o.u>mp)mp=o.u;});
        if(nd.k==="S"){nd.u=0;return;}
        var want=nd.rmed!=null?Math.min(nd.rmed,top)*px:mp+60;
        nd.u=Math.max(want,mp+70);
      });
    });
  }
  keys.forEach(function(k){
    var nd=nodes[k];
    if(S.lay==="h"){nd.x=nd.u;nd.y=nd.v;}else{nd.x=nd.v;nd.y=nd.u;}
    nd.r=nd.k==="S"?9:6+7*Math.sqrt(nd.n/Math.max(1,g.total));
  });
}

// ---------- dibujo del grafo
var svg=$("#svg"),vp=$("#vp");
function nodeRole(nd){
  if(nd.k==="S")return "src";
  if(nd.fail>0&&nd.fail>=nd.tgt)return "fail";
  if(nd.tgt>0)return "tgt";
  return nd.nul?"nul":"mid";
}
function renderGraph(){
  $("#stage").style.height=S.lay==="v"?"700px":"460px";
  var relay=false;
  if(dirtyGraph||!G){G=buildGraph();G.over={};G.sig="";dirtyGraph=false;relay=true;}
  var ov=selOverride(G),sig=JSON.stringify(ov);
  if(relay||sig!==G.sig){var moved=!relay;G.over=ov;G.sig=sig;layoutGraph(G,ov);if(moved)fitSoon();}
  var g=G,sq=seqOf(S.sel),selNodes={},selEdges={};
  for(var j=0;j<sq.length;j++){selNodes[sq[j].k]=1;if(j>0)selEdges[sq[j-1].k+"|"+sq[j].k]=1;}
  var out=[],tot=Math.max(1,g.total);
  g.edges.sort(function(a,b){return a.n-b.n;});
  function path(a,b){
    var A=g.nodes[a],B=g.nodes[b];if(!A||!B)return "";
    if(S.lay==="h"){var mx=(A.x+B.x)/2;return "M"+A.x+" "+A.y+"C"+mx+" "+A.y+","+mx+" "+B.y+","+B.x+" "+B.y;}
    var my=(A.y+B.y)/2;return "M"+A.x+" "+A.y+"C"+A.x+" "+my+","+B.x+" "+my+","+B.x+" "+B.y;
  }
  g.edges.forEach(function(e){
    var w=1+5*Math.sqrt(e.n/tot);
    out.push("<path d='"+path(e.a,e.b)+"' fill='none' stroke='"+C.edge+"' stroke-opacity='"+(0.35+0.5*e.n/tot).toFixed(2)+"' stroke-width='"+w.toFixed(1)+"'><title>"+e.n+" ciclos</title></path>");
  });
  Object.keys(selEdges).forEach(function(k){
    var p=k.split("|");if(!g.nodes[p[0]]||!g.nodes[p[1]])return;
    out.push("<path d='"+path(p[0],p[1])+"' fill='none' stroke='"+C.sel+"' stroke-width='3.2' stroke-linecap='round'/>");
  });
  var ks=Object.keys(g.nodes);
  ks.sort(function(a,b){return g.nodes[a].n-g.nodes[b].n;});
  ks.forEach(function(k){
    var nd=g.nodes[k],role=nodeRole(nd),sel=!!selNodes[k];
    var fill=role==="src"?C.src:role==="tgt"?C.tgt:role==="fail"?C.fail:role==="nul"?"none":C.node;
    var stroke=sel?C.sel:(role==="nul"?C.node:C.panel),sw=sel?3.2:(role==="nul"?1.6:1.5);
    var dash=role==="nul"?" stroke-dasharray='3 2'":"";
    out.push("<circle data-k='"+esc(k)+"' cx='"+nd.x+"' cy='"+nd.y+"' r='"+nd.r.toFixed(1)+"' fill='"+fill+"' stroke='"+stroke+"' stroke-width='"+sw+"'"+dash+" style='cursor:pointer'/>");
  });
  ks.forEach(function(k){
    var nd=g.nodes[k];if(nd.k==="S"){out.push(lbl(nd,"Sonda "+D.meta.p,null,true));return;}
    var t1=null,t2=null;
    if(S.view==="ip"){
      if(nd.nul)return;
      if(S.lblIP)t1=shortIp(D.ips[nd.ip][0]);
      if(S.lblASN&&validAsn(nd.asn))t2="AS"+nd.asn;
    }else{
      if(S.lblASN)t1=nd.asn!==""&&nd.asn!=null?"AS"+nd.asn:"sin ASN";
    }
    if(t1||t2)out.push(lbl(nd,t1,t2,false));
  });
  vp.innerHTML=out.join("");
  vp.setAttribute("transform","translate("+S.tx+","+S.ty+") scale("+S.k+")");
  function lbl(nd,t1,t2,src){
    var s="";
    var fs=11,col=C.ink;
    if(S.lay==="h"){
      if(t1)s+="<text x='"+nd.x+"' y='"+(nd.y+nd.r+13)+"' text-anchor='middle' font-size='"+fs+"' fill='"+col+"'>"+esc(t1)+"</text>";
      if(t2)s+="<text x='"+nd.x+"' y='"+(nd.y-nd.r-5)+"' text-anchor='middle' font-size='"+(fs-1)+"' fill='"+C.mut+"'>"+esc(t2)+"</text>";
    }else{
      if(t1)s+="<text x='"+(nd.x+nd.r+6)+"' y='"+(nd.y+4)+"' font-size='"+fs+"' fill='"+col+"'>"+esc(t1)+"</text>";
      if(t2)s+="<text x='"+(nd.x-nd.r-6)+"' y='"+(nd.y+4)+"' text-anchor='end' font-size='"+(fs-1)+"' fill='"+C.mut+"'>"+esc(t2)+"</text>";
    }
    return s;
  }
  var inSel=ks.filter(function(k){return selNodes[k];}).length;
  var cnt=g.total;
  $("#count").textContent=cnt+" ciclos en el grafo (rango de "+(S.r1-S.r0+1)+")"+(inSel<Object.keys(selNodes).length?" · el ciclo seleccionado no está en el grafo (fuera del rango o filtrado)":"");
}
function fitView(){
  if(!G)return;
  var ks=Object.keys(G.nodes);if(!ks.length)return;
  var x0=1e18,y0=1e18,x1=-1e18,y1=-1e18;
  ks.forEach(function(k){var n=G.nodes[k];x0=Math.min(x0,n.x-n.r-40);x1=Math.max(x1,n.x+n.r+40);y0=Math.min(y0,n.y-n.r-24);y1=Math.max(y1,n.y+n.r+24);});
  var W=svg.clientWidth,H=svg.clientHeight,k=Math.min(W/(x1-x0),H/(y1-y0),2.2);
  S.k=k;S.tx=(W-(x1-x0)*k)/2-x0*k;S.ty=(H-(y1-y0)*k)/2-y0*k;
  vp.setAttribute("transform","translate("+S.tx+","+S.ty+") scale("+S.k+")");
}

// ---------- tooltips / popups
var tip=null;
function nodeHtml(nd){
  var g=G,h="";
  var role=nodeRole(nd);
  if(nd.k==="S")return "<h4>Sonda "+D.meta.p+"</h4><div class='mut'>Origen de la medición</div>";
  if(S.view==="ip"){
    var p=D.ips[nd.ip]||["*",""];
    if(nd.nul)return "<h4>Salto sin respuesta (*)</h4><div><span class='k'>Salto </span>"+nd.layer+"</div><div><span class='k'>Ciclos </span>"+nd.n+" ("+(100*nd.n/Math.max(1,g.total)).toFixed(1)+"%)</div>";
    h+="<h4>"+esc(p[0])+"</h4>";
    h+="<div><span class='k'>ASN </span>"+(validAsn(p[1])?"AS"+esc(p[1]):"sin ASN")+"</div>";
  }else{
    h+="<h4>"+(nd.asn?"AS"+esc(nd.asn):"Sin ASN")+"</h4>";
    var l=Object.keys(nd.ips).sort(function(a,b){return nd.ips[b]-nd.ips[a];}).slice(0,6);
    if(l.length)h+="<div><span class='k'>IP </span>"+l.map(function(x){return esc(D.ips[x][0])+" ("+nd.ips[x]+")";}).join(", ")+"</div>";
  }
  h+="<div><span class='k'>Posición </span>"+nd.layer+(nd.eff!==undefined&&nd.eff!==nd.layer?" <span class='chg'>(en el ciclo seleccionado: "+nd.eff+")</span>":"")+"</div>";
  h+="<div><span class='k'>Ciclos </span>"+nd.n+" ("+(100*nd.n/Math.max(1,g.total)).toFixed(1)+"% del grafo)</div>";
  if(nd.rtts.length)h+="<div><span class='k'>RTT </span>med "+f1(med(nd.rtts))+" · mín "+f1(Math.min.apply(null,nd.rtts))+" · p95 "+f1(quant(nd.rtts,0.95))+"</div>";
  if(role==="tgt")h+="<div><span class='sw' style='background:"+C.tgt+"'></span>Destino en "+nd.tgt+" ciclos</div>";
  if(nd.fail)h+="<div><span class='sw' style='background:"+C.fail+"'></span>Último salto antes de perder el destino en "+nd.fail+" ciclos</div>";
  var sq=seqOf(S.sel);for(var j=0;j<sq.length;j++)if(sq[j].k===nd.k&&sq[j].rtt!=null){h+="<div><span class='sw' style='background:"+C.sel+"'></span>Ciclo seleccionado: "+f1(sq[j].rtt)+"</div>";break;}
  return h;
}
function showTip(html,x,y,pin){
  var d=document.createElement("div");d.className="tip"+(pin?" pin":"");d.innerHTML=(pin?"<span class='x'>✕</span>":"")+html;
  document.body.appendChild(d);
  var w=d.offsetWidth,h=d.offsetHeight;
  d.style.left=Math.min(innerWidth-w-8,x+14)+"px";d.style.top=Math.min(innerHeight-h-8,y+14)+"px";
  if(pin){d.querySelector(".x").onclick=function(){d.remove();pinned=pinned.filter(function(z){return z!==d;});};}
  return d;
}
function hideTip(){if(tip){tip.remove();tip=null;}}
svg.addEventListener("mousemove",function(ev){
  var t=ev.target;
  if(t.tagName==="circle"&&G){
    var nd=G.nodes[t.getAttribute("data-k")];
    if(!nd)return;
    hideTip();tip=showTip(nodeHtml(nd),ev.clientX,ev.clientY,false);
  }else hideTip();
});
svg.addEventListener("mouseleave",hideTip);
svg.addEventListener("click",function(ev){
  var t=ev.target;
  if(dragMoved)return;
  if(t.tagName==="circle"&&G){
    var nd=G.nodes[t.getAttribute("data-k")];if(!nd)return;
    hideTip();pinned.push(showTip(nodeHtml(nd),ev.clientX,ev.clientY,true));
  }else{pinned.forEach(function(d){d.remove();});pinned=[];}
});
// mover y zoom
var drag=null,dragMoved=false;
svg.addEventListener("mousedown",function(ev){drag={x:ev.clientX,y:ev.clientY,tx:S.tx,ty:S.ty};dragMoved=false;svg.classList.add("pan");});
window.addEventListener("mousemove",function(ev){
  if(!drag)return;
  var dx=ev.clientX-drag.x,dy=ev.clientY-drag.y;
  if(Math.abs(dx)+Math.abs(dy)>3)dragMoved=true;
  if(dragMoved){S.tx=drag.tx+dx;S.ty=drag.ty+dy;vp.setAttribute("transform","translate("+S.tx+","+S.ty+") scale("+S.k+")");}
});
window.addEventListener("mouseup",function(){drag=null;svg.classList.remove("pan");setTimeout(function(){dragMoved=false;},0);});
function zoomAt(f,cx,cy){
  var nk=Math.max(0.1,Math.min(6,S.k*f));f=nk/S.k;
  S.tx=cx-(cx-S.tx)*f;S.ty=cy-(cy-S.ty)*f;S.k=nk;
  vp.setAttribute("transform","translate("+S.tx+","+S.ty+") scale("+S.k+")");
}
svg.addEventListener("wheel",function(ev){ev.preventDefault();var r=svg.getBoundingClientRect();zoomAt(ev.deltaY<0?1.15:1/1.15,ev.clientX-r.left,ev.clientY-r.top);},{passive:false});
$("#zIn").onclick=function(){zoomAt(1.3,svg.clientWidth/2,svg.clientHeight/2);};
$("#zOut").onclick=function(){zoomAt(1/1.3,svg.clientWidth/2,svg.clientHeight/2);};
$("#zFit").onclick=fitView;

// ---------- línea de tiempo
var cv=$("#tl"),ctx=cv.getContext("2d"),TL={};
function bisect(t){var lo=0,hi=N;while(lo<hi){var m=(lo+hi)>>1;if(T[m]<t)lo=m+1;else hi=m;}return lo;}
function nearest(t){var j=bisect(t);if(j<=0)return 0;if(j>=N)return N-1;return (t-T[j-1])<=(T[j]-t)?j-1:j;}
function drawTimeline(){
  var dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
  cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  var t0=T[S.v0],t1=T[S.v1];if(t1<=t0)t1=t0+1;
  var padL=58,padR=8,pw=W-padL-padR;
  TL={t0:t0,t1:t1,padL:padL,pw:pw,W:W,H:H};
  function X(t){return padL+(t-t0)/(t1-t0)*pw;}
  var sev=new Int8Array(W),lenAt=new Int16Array(W),pidAt=new Int16Array(W),mn=new Float64Array(W),mx=new Float64Array(W),sm=new Float64Array(W),cn=new Int32Array(W);
  for(var c=0;c<W;c++){mn[c]=1e18;mx[c]=-1e18;}
  var vals=[];
  for(var i=S.v0;i<=S.v1;i++){
    var c=Math.floor(X(T[i]));if(c<0||c>=W)continue;
    var s=FAIL[i]?3:(LEN[i]!==MODAL?2:1);
    if(s>sev[c]){sev[c]=s;lenAt[c]=LEN[i];pidAt[c]=ASNPID[i];}
    if(DR[i]!=null){if(DR[i]<mn[c])mn[c]=DR[i];if(DR[i]>mx[c])mx[c]=DR[i];sm[c]+=DR[i];cn[c]++;vals.push(DR[i]);}
  }
  var cw=Math.max(1,Math.min(4,pw/Math.max(1,S.v1-S.v0+1)));
  ctx.font="11px system-ui,sans-serif";
  // tira de longitud y fila de AS-path
  ctx.fillStyle=C.mut;ctx.textAlign="right";ctx.textBaseline="middle";
  ctx.fillText("saltos",padL-6,9);ctx.fillText("AS-path",padL-6,25);
  for(var c=0;c<W;c++){
    if(!sev[c])continue;
    ctx.fillStyle=sev[c]===3?C.fail:lenColor(lenAt[c]);ctx.fillRect(c,2,cw,14);
    ctx.fillStyle=sev[c]===3?C.fail:pathColor(pidAt[c]);ctx.fillRect(c,19,cw,12);
  }
  // RTT
  var top=54,bot=H-40,lo=quant(vals,0.01),hi=quant(vals,0.99);
  if(lo==null){lo=0;hi=1;}if(hi<=lo)hi=lo+1;
  var pr=(hi-lo)*0.08;lo-=pr;hi+=pr;
  function Y(r){return bot-(Math.max(lo,Math.min(hi,r))-lo)/(hi-lo)*(bot-top);}
  ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.textAlign="right";ctx.fillStyle=C.mut;
  for(var g=0;g<=3;g++){var r=lo+(hi-lo)*g/3,y=Y(r);ctx.beginPath();ctx.moveTo(padL,y+.5);ctx.lineTo(W-padR,y+.5);ctx.stroke();ctx.fillText(r.toFixed(r>=100?0:1),padL-6,y);}
  ctx.save();ctx.beginPath();ctx.rect(padL,top-4,pw,bot-top+8);ctx.clip();
  ctx.fillStyle=C.accent;ctx.globalAlpha=.18;
  for(var c=0;c<W;c++){if(!cn[c])continue;var ya=Y(mx[c]),yb=Y(mn[c]);ctx.fillRect(c,ya,cw,Math.max(1,yb-ya));}
  ctx.globalAlpha=1;ctx.strokeStyle=C.accent;ctx.lineWidth=1.3;ctx.beginPath();var started=false;
  for(var c=0;c<W;c++){if(!cn[c])continue;var y=Y(sm[c]/cn[c]);if(!started){ctx.moveTo(c,y);started=true;}else ctx.lineTo(c,y);}
  ctx.stroke();ctx.restore();
  ctx.fillStyle=C.mut;ctx.textAlign="left";ctx.fillText("RTT al destino (ms)",padL+2,top-14);
  // eventos
  var ey=bot+8;
  ctx.textAlign="right";ctx.fillStyle=C.mut;ctx.fillText("eventos",padL-6,ey+5);
  ctx.fillStyle=C.ev;
  (D.evL||[]).forEach(function(i){if(i<S.v0||i>S.v1)return;ctx.fillRect(Math.floor(X(T[i])),ey,Math.max(1.5,cw),10);});
  ctx.fillStyle=C.sel;
  (D.evA||[]).forEach(function(i){if(i<S.v0||i>S.v1)return;ctx.fillRect(Math.floor(X(T[i])),ey+11,Math.max(1.5,cw),4);});
  // eje de tiempo
  ctx.fillStyle=C.mut;ctx.textBaseline="alphabetic";ctx.textAlign="center";
  var nt=Math.max(2,Math.floor(pw/120));
  for(var k=0;k<=nt;k++){var t=t0+(t1-t0)*k/nt,x=padL+pw*k/nt;ctx.fillText(fmtS(t),Math.min(W-30,Math.max(padL+24,x)),H-3);}
  // rango y selección
  if(S.r0>S.v0||S.r1<S.v1||S.r0!==0||S.r1!==N-1){
    var xa=X(T[S.r0]),xb=X(T[S.r1])+cw;
    ctx.fillStyle=C.accent;ctx.globalAlpha=.14;ctx.fillRect(xa,0,xb-xa,H-18);ctx.globalAlpha=1;
    ctx.strokeStyle=C.accent;ctx.lineWidth=1.2;ctx.strokeRect(xa+.5,.5,Math.max(1,xb-xa),H-19);
  }
  if(S.sel>=S.v0&&S.sel<=S.v1){var xs=X(T[S.sel]);ctx.strokeStyle=C.sel;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(xs+.5,0);ctx.lineTo(xs+.5,H-18);ctx.stroke();}
  if(brush){ctx.fillStyle=C.sel;ctx.globalAlpha=.22;ctx.fillRect(Math.min(brush.x0,brush.x1),0,Math.abs(brush.x1-brush.x0),H-18);ctx.globalAlpha=1;}
}
var brush=null,tlTip=null;
function tlX(ev){var r=cv.getBoundingClientRect();return ev.clientX-r.left;}
function tlIdx(x){return nearest(TL.t0+(x-TL.padL)/TL.pw*(TL.t1-TL.t0));}
cv.addEventListener("mousedown",function(ev){brush={x0:tlX(ev),x1:tlX(ev),moved:false};});
window.addEventListener("mousemove",function(ev){
  if(!brush)return;brush.x1=tlX(ev);if(Math.abs(brush.x1-brush.x0)>4)brush.moved=true;if(brush.moved)drawTimeline();
});
window.addEventListener("mouseup",function(ev){
  if(!brush)return;
  var b=brush;brush=null;
  if(b.moved){
    var a=tlIdx(Math.min(b.x0,b.x1)),z=tlIdx(Math.max(b.x0,b.x1));
    if(z<a){var q=a;a=z;z=q;}
    S.r0=a;S.r1=z;if(S.sel<a||S.sel>z)S.sel=a;
    changed(true,true);
  }else{
    var r=cv.getBoundingClientRect();
    if(ev.clientX>=r.left&&ev.clientX<=r.right&&ev.clientY>=r.top&&ev.clientY<=r.bottom){S.sel=tlIdx(b.x0);changed(false,true);}
  }
});
cv.addEventListener("dblclick",function(){S.r0=0;S.r1=N-1;changed(true,true);});
cv.addEventListener("mousemove",function(ev){
  if(brush)return;
  if(tlTip){tlTip.remove();tlTip=null;}
  var x=tlX(ev);if(x<TL.padL)return;
  var i=tlIdx(x),h=fmt(T[i])+"<br><span class='k'>Longitud </span>"+LEN[i]+" saltos<br>";
  h+=FAIL[i]?"<span class='bad'>Destino no alcanzado</span>":"<span class='k'>RTT destino </span>"+f1(DR[i]);
  tlTip=showTip(h,ev.clientX,ev.clientY,false);
});
cv.addEventListener("mouseleave",function(){if(tlTip){tlTip.remove();tlTip=null;}});
cv.addEventListener("wheel",function(ev){
  ev.preventDefault();
  var x=tlX(ev),tc=TL.t0+(x-TL.padL)/TL.pw*(TL.t1-TL.t0),f=ev.deltaY<0?0.8:1.25;
  var a=tc-(tc-TL.t0)*f,b=tc+(TL.t1-tc)*f;
  var i0=Math.max(0,bisect(a)),i1=Math.min(N-1,bisect(b));
  if(i1-i0<5){return;}
  S.v0=i0;S.v1=i1;drawTimeline();saveHash();
},{passive:false});

// ---------- paneles
function renderLegendTl(){
  var h="<span><s style='background:"+C.modal+"'></s>Longitud modal ("+MODAL+" saltos)</span>";
  OTHER.slice(0,8).forEach(function(l){h+="<span><s style='background:"+lenColor(l)+"'></s>"+l+" saltos</span>";});
  h+="<span><s style='background:"+C.fail+"'></s>Destino no alcanzado</span><span><s style='background:"+C.ev+"'></s>cambio_longitud (extractor)</span><span><s style='background:"+C.sel+"'></s>cambio_ip_asn</span>";
  h+="<span class='mut'>Arrastrar: elegir rango · Clic: elegir ciclo · Doble clic: rango completo · Rueda: zoom</span>";
  $("#legendTl").innerHTML=h;
}
function blockOf(i){
  for(var j=0;j<D.blocks.length;j++){var b=D.blocks[j];if(i>=b[1]&&i<=b[2])return j;}
  return -1;
}
function renderDetail(){
  var i=S.sel,c=D.c[i],h="";
  var bj=blockOf(i),b=bj>=0?D.blocks[bj]:null;
  h+="<div class='kv'>";
  h+="<span>Ciclo</span><span><b>"+fmt(T[i])+"</b> UTC <span class='mut'>(#"+(i+1)+" de "+N+")</span></span>";
  h+="<span>Longitud</span><span>"+LEN[i]+" saltos"+(LEN[i]!==MODAL?" <span class='chg'>(modal "+MODAL+")</span>":"")+"</span>";
  h+="<span>Destino</span><span>"+(FAIL[i]?"<span class='bad'>NO alcanzado</span>":"alcanzado · RTT "+f1(DR[i]))+"</span>";
  h+="<span>AS-path</span><span>"+esc(ASNP[i])+"</span>";
  h+="<span>Bloque</span><span>"+(b?b[0]+" saltos, "+fmt(T[b[1]]).slice(0,16)+" → "+fmt(T[b[2]]).slice(0,16):"—")+"</span>";
  var ev=[];if((D.evL||[]).indexOf(i)>=0)ev.push("cambio_longitud");if((D.evA||[]).indexOf(i)>=0)ev.push("cambio_ip_asn");
  h+="<span>Eventos</span><span>"+(ev.join(", ")||"—")+"</span>";
  h+="</div>";
  var p=i>0?D.c[i-1]:null;
  h+="<table><thead><tr><th class='n'>#</th><th>IP</th><th>ASN</th><th class='n'>RTT</th>"+(p?"<th>vs ciclo anterior</th>":"")+"</tr></thead><tbody>";
  for(var k=0;k<c[1].length;k++){
    var ip=c[1][k],r=c[2][k];
    var cip=ip<0?"*":D.ips[ip][0],asn=ip<0?"":D.ips[ip][1];
    var cmp="";
    if(p){
      var pip=k<p[1].length?p[1][k]:-2;
      if(pip===-2)cmp="<span class='chg'>salto nuevo</span>";
      else{
        var same=pip===ip,pr=p[2][k];
        if(!same){cmp="<span class='chg'>"+esc(pip<0?"*":D.ips[pip][0])+"</span>";}
        if(r!=null&&pr!=null){var d=r-pr;cmp+=(cmp?" · ":"")+"<span class='"+(Math.abs(d)>=5?"chg":"mut")+"'>"+(d>=0?"+":"")+d.toFixed(1)+" ms</span>";}
      }
    }
    h+="<tr><td class='n'>"+(k+1)+"</td><td"+(ip<0?" class='mut'":"")+">"+esc(cip)+"</td><td>"+(validAsn(asn)?"AS"+esc(asn):"<span class='mut'>—</span>")+"</td><td class='n'>"+(r==null?"<span class='mut'>*</span>":r.toFixed(1))+"</td>"+(p?"<td>"+cmp+"</td>":"")+"</tr>";
  }
  if(p&&p[1].length>c[1].length)h+="<tr><td colspan='5' class='chg'>El ciclo anterior tenía "+p[1].length+" saltos.</td></tr>";
  h+="</tbody></table>";
  return h;
}
function renderBlocks(){
  var h="<div class='mut' style='margin-bottom:6px'>Tramos sostenidos de igual longitud (≥ "+D.meta.persist+" ciclos) unidos cuando los separan solo tramos transitorios. Clic: ver el bloque.</div>";
  h+="<table><thead><tr><th class='n'>#</th><th class='n'>saltos</th><th>inicio</th><th>fin</th><th class='n'>dur (h)</th></tr></thead><tbody>";
  D.blocks.forEach(function(b,j){
    h+="<tr class='click' data-b='"+j+"'><td class='n'>"+(j+1)+"</td><td class='n'><span class='sw' style='background:"+lenColor(b[0])+"'></span>"+b[0]+"</td><td>"+fmt(T[b[1]]).slice(0,16)+"</td><td>"+fmt(T[b[2]]).slice(0,16)+"</td><td class='n'>"+((T[b[2]]-T[b[1]]+D.meta.cad)/3600).toFixed(1)+"</td></tr>";
  });
  return h+"</tbody></table>";
}
function renderFails(){
  var l=[];for(var i=0;i<N;i++)if(FAIL[i])l.push(i);
  var h="<div class='mut' style='margin-bottom:6px'>Ciclos cuyo último salto no es el destino (IP privada del probe o fuera del ASN del destino). Excluidos del análisis de caminos. Clic: ver el ciclo.</div>";
  if(!l.length)return h+"<div>Ninguno.</div>";
  h+="<table><thead><tr><th>ciclo</th><th class='n'>saltos</th><th>último salto</th><th class='n'>RTT</th></tr></thead><tbody>";
  l.slice(0,400).forEach(function(i){
    var c=D.c[i],k=c[1].length-1,ip=D.ips[c[1][k]][0];
    h+="<tr class='click' data-i='"+i+"'><td>"+fmt(T[i])+"</td><td class='n'>"+(k+1)+"</td><td class='bad'>"+esc(ip)+"</td><td class='n'>"+f1(c[2][k])+"</td></tr>";
  });
  if(l.length>400)h+="<tr><td colspan='4' class='mut'>… "+(l.length-400)+" más</td></tr>";
  return h+"</tbody></table>";
}
function renderEvents(){
  var l=(D.evL||[]).slice();
  var h="<div class='mut' style='margin-bottom:6px'>Ciclos con cambio_longitud en el extractor ("+l.length+"). Clic: ver el ciclo.</div>";
  if(!l.length)return h+"<div>Sin archivo de eventos o sin eventos.</div>";
  h+="<table><thead><tr><th>ciclo</th><th class='n'>saltos</th><th>otros</th></tr></thead><tbody>";
  l.slice(0,400).forEach(function(i){
    var tags=[];if(FAIL[i])tags.push("<span class='bad'>sin destino</span>");if((D.evA||[]).indexOf(i)>=0)tags.push("ip/asn");
    h+="<tr class='click' data-i='"+i+"'><td>"+fmt(T[i])+"</td><td class='n'>"+LEN[i]+"</td><td>"+tags.join(", ")+"</td></tr>";
  });
  if(l.length>400)h+="<tr><td colspan='3' class='mut'>… "+(l.length-400)+" más</td></tr>";
  return h+"</tbody></table>";
}
function renderPane(){
  var h=S.tab==="det"?renderDetail():S.tab==="blk"?renderBlocks():S.tab==="nod"?renderFails():renderEvents();
  var p=$("#pane");p.innerHTML=h;
  Array.prototype.forEach.call(document.querySelectorAll("#tabs button"),function(b){b.classList.toggle("on",b.getAttribute("data-t")===S.tab);});
  Array.prototype.forEach.call(p.querySelectorAll("tr[data-b]"),function(tr){tr.onclick=function(){
    var b=D.blocks[+tr.getAttribute("data-b")];S.r0=b[1];S.r1=b[2];S.sel=b[1];
    var m=Math.max(10,Math.round((b[2]-b[1])*0.3));S.v0=Math.max(0,b[1]-m);S.v1=Math.min(N-1,b[2]+m);S.tab="det";changed(true,true);fitSoon();};});
  Array.prototype.forEach.call(p.querySelectorAll("tr[data-i]"),function(tr){tr.onclick=function(){
    var i=+tr.getAttribute("data-i");S.sel=i;S.r0=Math.max(0,i-20);S.r1=Math.min(N-1,i+20);
    S.v0=Math.max(0,i-120);S.v1=Math.min(N-1,i+120);S.tab="det";changed(true,true);fitSoon();};});
}
$("#tabs").addEventListener("click",function(ev){var t=ev.target.getAttribute&&ev.target.getAttribute("data-t");if(t){S.tab=t;renderPane();}});

// ---------- controles
function fitSoon(){setTimeout(fitView,0);}
function changed(graph,timeline){
  if(graph)dirtyGraph=true;
  syncControls();
  renderGraph();drawTimeline();renderPane();saveHash();
}
function seg(id,key,afterGraph){
  $(id).addEventListener("click",function(ev){
    var v=ev.target.getAttribute&&ev.target.getAttribute("data-v");if(!v)return;
    S[key]=v;changed(true,false);fitSoon();
  });
}
seg("#sgView","view");seg("#sgDist","dist");seg("#sgLay","lay");seg("#sgSucc","succ");
$("#cIP").onchange=function(){S.lblIP=this.checked;changed(false,false);};
$("#cASN").onchange=function(){S.lblASN=this.checked;changed(false,false);};
$("#q").oninput=function(){S.q=this.value;changed(true,false);};
$("#qm").onchange=function(){S.qm=this.value;changed(true,false);};
function rttFromSliders(){
  var a=+$("#rLo").value,b=+$("#rHi").value;if(a>b){var t=a;a=b;b=t;}
  S.rlo=a<=0?RMIN:RMIN+(RMAX-RMIN)*a/1000;S.rhi=b>=1000?RMAX:RMIN+(RMAX-RMIN)*b/1000;
}
$("#rLo").oninput=$("#rHi").oninput=function(){rttFromSliders();changed(true,false);};
$("#bPrev").onclick=function(){step(-1);};$("#bNext").onclick=function(){step(1);};
function step(d){S.sel=Math.max(0,Math.min(N-1,S.sel+d));if(S.sel<S.v0||S.sel>S.v1){var w=S.v1-S.v0;S.v0=Math.max(0,S.sel-(w>>1));S.v1=Math.min(N-1,S.v0+w);}changed(false,true);}
$("#slider").max=N-1;
$("#slider").oninput=function(){S.sel=+this.value;changed(false,true);};
function goto(){
  var s=$("#goto").value.trim().replace(/\//g,"-");if(!s)return;
  if(s.length===10)s+=" 00:00";if(s.length===16)s+=":00";
  var t=Date.parse(s.replace(" ","T")+"Z");if(isNaN(t))return;
  S.sel=nearest(t/1000);step(0);
}
$("#bGo").onclick=goto;$("#goto").onkeydown=function(e){if(e.key==="Enter")goto();};
$("#bZoomR").onclick=function(){if(S.r1>S.r0){var m=Math.max(2,Math.round((S.r1-S.r0)*0.05));S.v0=Math.max(0,S.r0-m);S.v1=Math.min(N-1,S.r1+m);changed(false,true);}};
$("#bAll").onclick=function(){S.v0=0;S.v1=N-1;S.r0=0;S.r1=N-1;changed(true,true);fitSoon();};
document.addEventListener("keydown",function(ev){
  if(ev.target&&(ev.target.tagName==="INPUT"||ev.target.tagName==="SELECT"))return;
  if(ev.key==="ArrowLeft"){step(ev.shiftKey?-10:-1);ev.preventDefault();}
  else if(ev.key==="ArrowRight"){step(ev.shiftKey?10:1);ev.preventDefault();}
  else if(ev.key==="Home"){S.sel=0;step(0);}
  else if(ev.key==="End"){S.sel=N-1;step(0);}
});
function syncControls(){
  function on(id,v){Array.prototype.forEach.call($(id).children,function(b){b.classList.toggle("on",b.getAttribute("data-v")===v);});}
  on("#sgView",S.view);on("#sgDist",S.dist);on("#sgLay",S.lay);on("#sgSucc",S.succ);
  $("#cIP").checked=S.lblIP;$("#cASN").checked=S.lblASN;
  if($("#q").value!==S.q)$("#q").value=S.q;$("#qm").value=S.qm;
  $("#slider").value=S.sel;
  $("#rLo").value=Math.round((S.rlo-RMIN)/(RMAX-RMIN)*1000);$("#rHi").value=Math.round((S.rhi-RMIN)/(RMAX-RMIN)*1000);
  $("#rTxt").textContent=rttActive()?f1(S.rlo)+" – "+(S.rhi>=RMAX-1e-9?"ALL":f1(S.rhi)):"ALL";
  $("#hRange").innerHTML="Rango: <b>"+fmt(T[S.r0]).slice(0,16)+"</b> → <b>"+fmt(T[S.r1]).slice(0,16)+"</b>";
}
// exportar
function download(name,type,data){var a=document.createElement("a");a.href=URL.createObjectURL(new Blob([data],{type:type}));a.download=name;document.body.appendChild(a);a.click();setTimeout(function(){a.remove();URL.revokeObjectURL(a.href);},500);}
function rowsFiltered(){
  var idx=G?G.idx:included();
  return idx.map(function(i){return {timestamp:fmt(T[i]),saltos:LEN[i],destino_alcanzado:!FAIL[i],rtt_destino_ms:DR[i],as_path:ASNP[i],ips:D.c[i][1].map(function(x){return x<0?"*":D.ips[x][0];}).join(" ")};});
}
function svgText(){
  var s=svg.cloneNode(true);s.setAttribute("xmlns","http://www.w3.org/2000/svg");
  var W=svg.clientWidth,H=svg.clientHeight;s.setAttribute("width",W);s.setAttribute("height",H);s.setAttribute("viewBox","0 0 "+W+" "+H);
  var bg=document.createElementNS("http://www.w3.org/2000/svg","rect");bg.setAttribute("width",W);bg.setAttribute("height",H);bg.setAttribute("fill",C.panel);
  s.insertBefore(bg,s.firstChild);
  s.setAttribute("font-family","system-ui,Segoe UI,Roboto,sans-serif");
  return new XMLSerializer().serializeToString(s);
}
$("#bExp").onclick=function(ev){
  var o=prompt("Exportar:\n1 = PNG (2x)\n2 = SVG\n3 = CSV (ciclos del grafo)\n4 = JSON (ciclos del grafo)","1");
  var name="caminos_m"+D.meta.m+"_p"+D.meta.p;
  if(o==="1"){
    var W=svg.clientWidth,H=svg.clientHeight,img=new Image();
    img.onload=function(){var k=2,c=document.createElement("canvas");c.width=W*k;c.height=H*k;var x=c.getContext("2d");x.drawImage(img,0,0,W*k,H*k);
      c.toBlob(function(b){var a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=name+".png";a.click();});};
    img.src="data:image/svg+xml;charset=utf-8,"+encodeURIComponent(svgText());
  }else if(o==="2")download(name+".svg","image/svg+xml",svgText());
  else if(o==="3"){var r=rowsFiltered(),cols=Object.keys(r[0]||{timestamp:1});
    download(name+".csv","text/csv",[cols.join(",")].concat(r.map(function(x){return cols.map(function(c){return '"'+String(x[c]==null?"":x[c]).replace(/"/g,'""')+'"';}).join(",");})).join("\n"));}
  else if(o==="4")download(name+".json","application/json",JSON.stringify(rowsFiltered(),null,1));
};
$("#bLink").onclick=function(){saveHash();setTimeout(function(){var u=location.href;if(navigator.clipboard)navigator.clipboard.writeText(u).catch(function(){});$("#bLink").textContent="Enlace copiado";setTimeout(function(){$("#bLink").textContent="Copiar enlace";},1500);},300);};

// ayuda
function helpOpen(o){$("#help").classList.toggle("open",o);}
$("#bHelp").onclick=function(){helpOpen(true);};
$("#hClose").onclick=function(){helpOpen(false);};
$("#help").addEventListener("click",function(ev){if(ev.target===this)helpOpen(false);});
document.addEventListener("keydown",function(ev){
  if(ev.key==="Escape")helpOpen(false);
  else if(ev.key==="?"&&!(ev.target&&(ev.target.tagName==="INPUT"||ev.target.tagName==="SELECT")))helpOpen(true);
});
// ---------- inicio
$("#hM").textContent=D.meta.m;$("#hP").textContent=D.meta.p;$("#hN").textContent=N;
$("#hDest").innerHTML="Destino: "+(D.meta.asn?"AS"+esc(D.meta.asn):"?")+" · "+D.meta.nfail+" ciclos sin destino";
$("#foot").textContent="Generado "+D.meta.gen+" por verificar_caminos.py · Hora en UTC · Caminos con destino no alcanzado excluidos de los bloques; se muestran en rojo.";
setStateFromHash();
if(S.v0>=S.v1){S.v0=0;S.v1=N-1;}
readColors();renderLegendTl();
if(window.matchMedia){var mq=window.matchMedia("(prefers-color-scheme: dark)");(mq.addEventListener||mq.addListener).call(mq,"change",function(){readColors();renderLegendTl();changed(false,true);});}
window.addEventListener("resize",function(){drawTimeline();});
syncControls();renderGraph();fitView();drawTimeline();renderPane();
})();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
