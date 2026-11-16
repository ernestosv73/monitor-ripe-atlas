# -*- coding: utf-8 -*-
"""
=============================================================================
evaluar_segmentacion.py  (v3.1)
Evalua la salida del HDP-HMM contra la VERDAD del generador sintetico de bloque (generador_rtt_bloque.py).
=============================================================================
Metodologia: la serie sintetica es UN bloque de ruta fija (longitud y AS-path constantes). La evaluacion es una sola:

  1. PUNTOS DE CAMBIO (change_points_*.csv): cambios de nivel del RTT del ultimo salto. Metricas: VP, FN, FP,
     precision, recall, F1, desfase en ciclos y sensibilidad a la tolerancia.
  2. ETIQUETAS de los segmentos (segmentos_caracterizados_*.csv): el tipo que asigno el script contra el que
     corresponde segun la verdad (mediana de delta_ms del segmento y la misma regla de severidad), matriz de
     confusion y error de la desviacion respecto de la referencia (script - verdad).
  3. CHEQUEO DEL BLOQUE (opcional, --bloques-csv): verificar_caminos debe devolver UN solo bloque y no perder ciclos.

Que se considera "verdad":
  * Serie evaluada = ciclos que llegaron al destino (sin_destino = 0) y no son huecos; con --ciclos-excluidos tambien
    se quitan esos ciclos (igual que hace el script HDP-HMM).
  * Punto de cambio verdadero = dos ciclos consecutivos con |nivel_ultimo_ms| distinto en al menos --delta-min-ms.
    Los cambios menores se informan aparte (sub-umbral).

Que NO se exige (FN "no evaluables", se informan pero no penalizan el recall):
  * 'meseta_corta': alguno de los dos niveles dura menos de --min-meseta ciclos; el suavizado (min_dwell) lo borra.
    Use el MISMO valor que --min-dwell del HDP-HMM.

Identificabilidad de la referencia: el script usa como RTT de referencia el P25 de los ciclos del bloque. Si el regimen
base ocupa menos de --min-frac-base de los ciclos, ese P25 cae dentro de una degradacion y las etiquetas y la
desviacion no pueden acertarse por construccion. En ese caso los segmentos se informan aparte ('no_identificable').

Tolerancia en segundos: --tolerancia-ciclos x interval + --jitter-s (o --tolerancia-s fijo).
Banda de ambiguedad: un segmento cuyo |delta| verdadero esta a <= --banda-ms (default 0.5) de un umbral (2 ms moderada,
5 ms severa) no se cuenta en el acierto; se informa aparte.

Cambios de la v3.1 respecto de la v3:
  * Recall de puntos de cambio = VP evaluables / evaluables (antes el numerador incluia VP de eventos no evaluables).
  * Segunda pasada con --tolerancia-ampliada-ciclos (default 5): un FN y un FP que aparecen a esa distancia son el mismo
    evento detectado con desfase ('detectado_con_desfase' / 'desfase_de_evento'). Se informa recall ampliado.
  * Falsos positivos clasificados como justificados (desfase de un evento real, cambio real sub-umbral) o ESPURIOS
    (cerca de spike, sin causa); se informa la precision depurada. La cruda sigue en el resumen.
  * Micro-segmentos (< --min-ciclos-segmento, default 10): evaluados aparte; el error de la desviacion principal se
    resume con la MEDIANA y excluye los micro-segmentos.

Uso tipico:
  python evaluar_segmentacion.py \\
      --verdad  sintetico/verdad_ciclos_measurement90001_probe1.csv \\
      --change-points  hd/change_points_measurement90001_probe1_auto.csv \\
      --segmentos      hd/segmentos_caracterizados_measurement90001_probe1_auto.csv \\
      --bloques-csv vc/bloques_longitud_measurement90001_probe1.csv \\
      --ciclos-excluidos vc/ciclos_excluidos_measurement90001_probe1.csv \\
      --interval 900 --min-meseta 5 --output-dir evaluacion --etiqueta semilla42 --resumen-csv comparacion.csv

Solo usa la biblioteca estandar y numpy. Compatible con Python 2.7 y 3.
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import csv
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime

import numpy as np

FMT = "%Y-%m-%d %H:%M:%S"
EPOCH = datetime(1970, 1, 1)
TOLERANCIAS_SENSIBILIDAD = (0, 1, 2, 5, 10)   # en ciclos


# ============================================================================
# UTILIDADES
# ============================================================================
def abrir(path, modo):
    if sys.version_info[0] >= 3:
        return open(path, modo, newline='')
    return open(path, modo + 'b')


def a_seg(ts):
    return (datetime.strptime(ts.strip()[:19], FMT) - EPOCH).total_seconds()


def de_seg(t):
    return (EPOCH + timedelta(seconds=t)).strftime(FMT)


def leer_csv(path):
    with abrir(path, 'r') as f:
        return list(csv.DictReader(f))


def a_float(x, defecto=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return defecto


def stats(v):
    if not len(v):
        return None
    v = np.asarray(v, dtype=float)
    return {'n': len(v), 'media': float(np.mean(v)), 'mae': float(np.mean(np.abs(v))), 'mediana': float(np.median(v)),
            'p90': float(np.percentile(np.abs(v), 90)), 'max_abs': float(np.max(np.abs(v)))}


def fmt(x, nd=2):
    return "-" if x is None else ("{:." + str(nd) + "f}").format(x)


# Reglas de severidad: deben coincidir con clasificar_tipo_degradacion / clasificar_cambio_rtt del script HDP-HMM
def tipo_esperado(delta, dur_min, sev, mod):
    mag = abs(delta)
    pref = 'degradacion' if delta > 0 else 'mejora'
    if dur_min > 60 and mag > sev: return pref + '_severa_sostenida'
    elif dur_min > 30 and mag > mod: return pref + '_moderada_sostenida'
    elif dur_min < 10 and mag > 10: return 'pico_transitorio'
    elif mag < 1: return 'regimen_base'
    else: return 'variacion_normal'


def ambiguo(delta, args):
    """True si |delta| esta a <= banda de un umbral de severidad (la etiqueta depende del sesgo de medicion)."""
    mag = abs(delta)
    return any(abs(mag - u) <= args.banda_ms for u in (args.umbral_moderada, args.umbral_severa))


def emparejar(t_verdad, t_det, tol_s):
    """Empareja uno a uno (greedy por menor distancia temporal). Devuelve lista de (i_verdad, j_det)."""
    det_ord = sorted(range(len(t_det)), key=lambda j: t_det[j])
    det_t = [t_det[j] for j in det_ord]
    pares = []
    for i, t in enumerate(t_verdad):
        lo = bisect_left(det_t, t - tol_s)
        hi = bisect_right(det_t, t + tol_s)
        for k in range(lo, hi):
            pares.append((abs(det_t[k] - t), i, det_ord[k]))
    pares.sort()
    usados_v, usados_d, res = set(), set(), []
    for _, i, j in pares:
        if i in usados_v or j in usados_d:
            continue
        usados_v.add(i); usados_d.add(j); res.append((i, j))
    return res


def metricas_pr(vp, fn_eval, fp):
    prec = vp / (vp + fp) if (vp + fp) else None
    rec = vp / (vp + fn_eval) if (vp + fn_eval) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return prec, rec, f1


# ============================================================================
# VERDAD
# ============================================================================
def cargar_excluidos(path):
    if not path:
        return set()
    return set(r['ciclo'].strip() for r in leer_csv(path) if (r.get('ciclo') or '').strip())


def cargar_verdad(path, excluidos):
    """Serie evaluada: ciclos con destino, sin huecos y no excluidos por verificar_caminos."""
    serie = []
    for r in leer_csv(path):
        if (r.get('hueco') or '0').strip() not in ('', '0'):
            continue
        if (r.get('sin_destino') or '0').strip() not in ('', '0'):
            continue
        ts = r['timestamp'].strip()
        if ts in excluidos:
            continue
        nivel = a_float(r.get('nivel_ultimo_ms'))
        if nivel is None:
            continue
        serie.append({'ts': ts, 't': a_seg(ts), 'nivel': nivel,
                      'delta': a_float(r.get('delta_ms'), 0.0) or 0.0,
                      'spike': a_float(r.get('spike_ms'), 0.0) or 0.0})
    return serie


def derivar_eventos_verdad(serie, args):
    """Devuelve (cps, cambios_subumbral) de la verdad."""
    n = len(serie)
    front = [0]
    for i in range(1, n):
        if abs(serie[i]['nivel'] - serie[i - 1]['nivel']) >= 0.05:
            front.append(i)
    front.append(n)
    pos_front = dict((f, idx) for idx, f in enumerate(front))
    cps, subumbral = [], []
    for i in range(1, n):
        a, b = serie[i - 1], serie[i]
        d = b['nivel'] - a['nivel']
        if abs(d) < 0.05:
            continue
        f = pos_front[i]
        largo_antes = i - front[f - 1]
        largo_despues = front[f + 1] - i
        ev = {'ts': b['ts'], 't': b['t'], 'delta': d, 'largo_antes': largo_antes, 'largo_despues': largo_despues,
              'estructural': ''}
        if abs(d) < args.delta_min_ms:
            subumbral.append(ev)
            continue
        if min(largo_antes, largo_despues) < args.min_meseta:
            ev['estructural'] = 'meseta_corta'
        cps.append(ev)
    return cps, subumbral


def identificabilidad(serie, min_frac_base):
    n_base = sum(1 for s in serie if abs(s['delta']) < 0.05)
    frac = n_base / float(len(serie))
    return {'ciclos': len(serie), 'ciclos_base': n_base, 'frac_base': frac, 'identificable': frac >= min_frac_base}


def chequeo_bloque(path, serie, args):
    """verificar_caminos debe dar UN bloque que cubra la serie. Devuelve dict con el resultado."""
    bl = []
    for r in leer_csv(path):
        bl.append((a_seg(r['inicio']), a_seg(r['fin']), int(a_float(r.get('ciclos'), 0))))
    bl.sort()
    fuera = sum(1 for s in serie if not any(b[0] <= s['t'] <= b[1] for b in bl))
    cortos = sum(1 for b in bl if b[2] < args.min_bloque)
    return {'n_bloques': len(bl), 'ciclos_fuera': fuera, 'bloques_cortos': cortos}


# ============================================================================
# 1. PUNTOS DE CAMBIO
# ============================================================================
def evaluar_cps(cps_verdad, det_rows, serie, args):
    """Empareja en dos pasadas: (1) tolerancia normal -> VP; (2) tolerancia ampliada entre lo que quedo sin emparejar
    -> el MISMO evento detectado con desfase (ni falso negativo ni falso positivo 'reales')."""
    interval = args.interval
    t_v = [c['t'] for c in cps_verdad]
    t_d = [a_seg(r['timestamp']) for r in det_rows]
    tol_s = args.tol_s
    pares = emparejar(t_v, t_d, tol_s)
    par_v = dict(pares)
    usados_d = set(j for _, j in pares)
    # segunda pasada, solo con lo que no se emparejo
    libres_v = [i for i in range(len(cps_verdad)) if i not in par_v]
    libres_d = [j for j in range(len(det_rows)) if j not in usados_d]
    pares2 = emparejar([t_v[i] for i in libres_v], [t_d[j] for j in libres_d], args.tol_amp_s)
    par2_v = dict((libres_v[i], libres_d[j]) for i, j in pares2)
    usados2_d = set(libres_d[j] for _, j in pares2)

    t_spikes = [s['t'] for s in serie if s['spike'] > 0]
    filas, offsets, offsets2 = [], [], []
    for i, c in enumerate(cps_verdad):
        if i in par_v:
            j = par_v[i]
            off = (t_d[j] - c['t']) / interval
            offsets.append(off)
            filas.append(['VP', c['ts'], det_rows[j]['timestamp'], fmt(off, 1), fmt(c['delta']), ''])
        elif i in par2_v:
            j = par2_v[i]
            off = (t_d[j] - c['t']) / interval
            offsets2.append(off)
            filas.append(['FN', c['ts'], det_rows[j]['timestamp'], fmt(off, 1), fmt(c['delta']),
                          c['estructural'] or 'detectado_con_desfase'])
        else:
            filas.append(['FN', c['ts'], '', '', fmt(c['delta']), c['estructural'] or 'no_detectado'])
    t_sub = [c['t'] for c in args._subumbral]
    fp_cat = defaultdict(int)
    for j, r in enumerate(det_rows):
        if j in usados_d:
            continue
        td = t_d[j]
        if j in usados2_d: cat = 'desfase_de_evento'
        elif any(abs(td - ts) <= 2 * interval for ts in t_spikes): cat = 'cerca_de_spike'
        elif any(abs(td - ts) <= tol_s for ts in t_sub): cat = 'cambio_subumbral'
        else: cat = 'sin_causa_identificada'
        fp_cat[cat] += 1
        filas.append(['FP', '', r['timestamp'], '', '', cat])
    n_det = len(det_rows)
    n_vp = len(pares)
    vp_eval = sum(1 for i in par_v if not cps_verdad[i]['estructural'])
    n_eval = sum(1 for c in cps_verdad if not c['estructural'])
    fn_total = len(cps_verdad) - n_vp
    fn_estruct = sum(1 for i, c in enumerate(cps_verdad) if i not in par_v and c['estructural'])
    fn_eval = n_eval - vp_eval
    fp = n_det - n_vp
    # con desfase ampliado
    desf_eval = sum(1 for i in par2_v if not cps_verdad[i]['estructural'])
    fn_real = fn_eval - desf_eval          # sin ninguna deteccion cerca
    # precision depurada: justificadas = VP + desfase + cambios reales sub-umbral; espurias = spike + sin causa
    justificadas = n_vp + fp_cat['desfase_de_evento'] + fp_cat['cambio_subumbral']
    espurias = fp_cat['cerca_de_spike'] + fp_cat['sin_causa_identificada']
    sens = []
    for tc in sorted(set(TOLERANCIAS_SENSIBILIDAD) | set([args.tolerancia_ciclos])):
        pr = emparejar(t_v, t_d, tc * interval + args.jitter_s)
        vp_t = len(pr)
        vp_t_eval = sum(1 for i, _ in pr if not cps_verdad[i]['estructural'])
        fn_eval_t = n_eval - vp_t_eval
        p = vp_t / n_det if n_det else None
        r_ = vp_t_eval / n_eval if n_eval else None
        sens.append((tc, vp_t, fn_eval_t, n_det - vp_t, p, r_, None))
    prec = (n_vp / n_det) if n_det else None
    rec = (vp_eval / n_eval) if n_eval else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    rec_amp = ((vp_eval + desf_eval) / n_eval) if n_eval else None
    return {'filas': filas, 'vp': n_vp, 'vp_eval': vp_eval, 'fn': fn_total, 'fn_estructural': fn_estruct,
            'fn_eval': fn_eval, 'fn_real': fn_real, 'desfase_eval': desf_eval, 'fp': fp, 'fp_cat': dict(fp_cat),
            'n_verdad': len(cps_verdad), 'n_eval': n_eval, 'n_det': n_det, 'precision': prec,
            'precision_depurada': (justificadas / n_det) if n_det else None, 'espurias': espurias,
            'recall_eval': rec, 'recall_ampliado': rec_amp,
            'recall_todos': (n_vp / len(cps_verdad)) if cps_verdad else None, 'f1': f1,
            'offsets': stats(offsets), 'offsets_desfase': stats(offsets2), 'sens': sens}


# ============================================================================
# 2. ETIQUETAS DE SEGMENTOS
# ============================================================================
def evaluar_segmentos(seg_rows, serie, info, args):
    """info: resultado de identificabilidad() (nivel de bloque)."""
    ts_lista = [s['ts'] for s in serie]
    filas = []
    ev = {'aciertos': [], 'pesos': []}
    conf = defaultdict(lambda: defaultdict(int))
    dev_ok, dev_noid, purezas = [], [], []
    n_noid, n_amb = 0, 0
    amb_res, noid_res = [], []
    micro_res, micro_err, micro_ciclos = [], [], 0
    todos = []                       # acierto sin filtros (comparable con la v1)
    for r in seg_rows:
        ini, fin = r['start_ts'].strip(), r['end_ts'].strip()
        i0, i1 = bisect_left(ts_lista, ini), bisect_right(ts_lista, fin)
        ciclos = serie[i0:i1]
        if not ciclos:
            continue
        dur = a_float(r['duracion_minutos'], 0.0)
        n = int(a_float(r['n_ciclos'], len(ciclos)))
        deltas = [c['delta'] for c in ciclos]
        d_med = float(np.median(deltas))
        pureza = sum(1 for d in deltas if abs(d - d_med) < 0.05) / len(deltas)
        identif = info['identificable']
        esperado = tipo_esperado(d_med, dur, args.umbral_severa, args.umbral_moderada)
        obtenido = r['tipo_degradacion'].strip()
        ok = (esperado == obtenido)
        dev = a_float(r.get('desviacion_ms'))
        err = (dev - d_med) if dev is not None else None
        amb = ambiguo(d_med, args)
        todos.append(ok)
        if not identif:
            estado = 'no_identificable'; n_noid += 1; noid_res.append(ok)
            if err is not None: dev_noid.append(err)
        elif n < args.min_ciclos_segmento:
            estado = 'micro'; micro_res.append(ok); micro_ciclos += n
            if err is not None: micro_err.append(err)
        elif amb:
            estado = 'ambiguo'; n_amb += 1; amb_res.append(ok)
            if err is not None: dev_ok.append(err)
        else:
            estado = 'evaluado'
            ev['aciertos'].append(ok); ev['pesos'].append(n); purezas.append(pureza)
            conf[esperado][obtenido] += 1
            if err is not None: dev_ok.append(err)
        filas.append([r['start_ts'], r['end_ts'], n, fmt(dur, 0), r.get('longitud', ''), fmt(d_med),
                      fmt(a_float(r.get('mediana'))), fmt(dev), fmt(err), esperado, obtenido,
                      'si' if ok else 'no', fmt(pureza, 2), estado])
    n_seg = len(ev['aciertos'])
    pes = ev['pesos']
    return {'filas': filas, 'n': n_seg, 'n_total': len(todos), 'acc_sin_filtros': (sum(todos) / len(todos)) if todos else None,
            'acc': (sum(ev['aciertos']) / n_seg) if n_seg else None,
            'acc_pond': (sum(p for p, a in zip(pes, ev['aciertos']) if a) / sum(pes)) if sum(pes) else None,
            'dev_err': stats(dev_ok), 'dev_err_noid': stats(dev_noid),
            'pureza_media': float(np.mean(purezas)) if purezas else None,
            'n_impuros': sum(1 for p in purezas if p < 0.9), 'conf': conf, 'n_noid': n_noid, 'n_amb': n_amb,
            'amb_ok': sum(amb_res), 'noid_ok': sum(noid_res),
            'n_micro': len(micro_res), 'micro_ok': sum(micro_res), 'micro_ciclos': micro_ciclos,
            'micro_err': stats(micro_err), 'total_ciclos': sum(int(a_float(r['n_ciclos'], 0)) for r in seg_rows)}




# ============================================================================
# REPORTE
# ============================================================================
def reporte(args, serie, info, chq, ev_cp, ev_seg, n_sub):
    L = []
    P = L.append
    P("=" * 74)
    P("EVALUACION DE LA SEGMENTACION CONTRA LA VERDAD SINTETICA (BLOQUE)" + ((" [" + args.etiqueta + "]") if args.etiqueta else ""))
    P("=" * 74)
    P("Ciclos evaluados (con destino, sin huecos ni excluidos): {}".format(len(serie)))
    P("Tolerancia de emparejamiento: {:g} s (= {:g} ciclos x {:g} s + jitter {:g} s) | delta minimo exigido: {:g} ms".format(
        args.tol_s, args.tolerancia_ciclos, args.interval, args.jitter_s, args.delta_min_ms))
    P("min. meseta: {} | banda de ambiguedad: +-{:g} ms | fraccion minima de base: {:g}".format(
        args.min_meseta, args.banda_ms, args.min_frac_base))
    P("")
    P("Referencia (P25 de los ciclos del bloque): ciclos en regimen base {} de {} ({:.1%}) -> {}".format(
        info['ciclos_base'], info['ciclos'], info['frac_base'],
        "IDENTIFICABLE" if info['identificable'] else
        "NO IDENTIFICABLE: el P25 cae dentro de una degradacion; las etiquetas no pueden acertarse por construccion"))
    if chq is not None:
        P("Chequeo del bloque (verificar_caminos): {} bloque(s) | ciclos fuera de bloque: {} | bloques cortos: {} -> {}".format(
            chq['n_bloques'], chq['ciclos_fuera'], chq['bloques_cortos'],
            "OK" if (chq['n_bloques'] == 1 and chq['ciclos_fuera'] == 0 and chq['bloques_cortos'] == 0) else "REVISAR"))
    P("")
    P("--- 1. PUNTOS DE CAMBIO ---")
    P("Verdad: {} (evaluables: {}; no evaluables: {}) | detectados: {} | cambios sub-umbral ignorados: {}".format(
        ev_cp['n_verdad'], ev_cp['n_eval'], ev_cp['n_verdad'] - ev_cp['n_eval'], ev_cp['n_det'], n_sub))
    P("VP {} (evaluables: {}) | FN {} (no evaluables: {}) | FP {}".format(
        ev_cp['vp'], ev_cp['vp_eval'], ev_cp['fn'], ev_cp['fn_estructural'], ev_cp['fp']))
    P("Precision {} | Recall (VP evaluables / evaluables) {} | Recall (todos) {} | F1 {}".format(
        fmt(ev_cp['precision'], 3), fmt(ev_cp['recall_eval'], 3), fmt(ev_cp['recall_todos'], 3), fmt(ev_cp['f1'], 3)))
    o = ev_cp['offsets']
    if o:
        P("Desfase de los VP (ciclos, detectado - verdad): media {} | mediana {} | p90 |.| {} | max |.| {}".format(
            fmt(o['media'], 2), fmt(o['mediana'], 2), fmt(o['p90'], 2), fmt(o['max_abs'], 2)))
    P("")
    P("Lectura con tolerancia ampliada ({:g} ciclos + jitter): lo que no empareja a {:g} ciclos se vuelve a buscar mas lejos".format(
        args.tolerancia_ampliada_ciclos, args.tolerancia_ciclos))
    P("  Falsos negativos evaluables ({}): detectados con desfase {} | sin ninguna deteccion cerca {}".format(
        ev_cp['fn_eval'], ev_cp['desfase_eval'], ev_cp['fn_real']))
    P("  Recall ampliado (VP + desfase) / evaluables: {}".format(fmt(ev_cp['recall_ampliado'], 3)))
    od = ev_cp['offsets_desfase']
    if od:
        P("  Desfase de esos eventos (ciclos): media {} | min/max |.| {} | {} (n={})".format(
            fmt(od['media'], 1), fmt(od['mediana'], 1), fmt(od['max_abs'], 1), od['n']))
    fc = ev_cp['fp_cat']
    P("  Falsos positivos ({}): desfase de un evento real {} | cambio real sub-umbral {} | ESPURIOS: cerca de spike {}, "
      "sin causa {}".format(ev_cp['fp'], fc.get('desfase_de_evento', 0), fc.get('cambio_subumbral', 0),
                            fc.get('cerca_de_spike', 0), fc.get('sin_causa_identificada', 0)))
    P("  Precision depurada (VP + desfase + sub-umbral reales) / detectados: {} | detecciones espurias: {} ({:.1%})".format(
        fmt(ev_cp['precision_depurada'], 3), ev_cp['espurias'],
        ev_cp['espurias'] / float(ev_cp['n_det']) if ev_cp['n_det'] else 0.0))
    P("Sensibilidad a la tolerancia (ciclos, + jitter): VP / FN eval / FP / precision / recall eval")
    for tc, vp, fn, fp, p, r, f1 in ev_cp['sens']:
        P("  {:>3}: {:>4} / {:>4} / {:>4} / {} / {}".format(tc, vp, fn, fp, fmt(p, 3), fmt(r, 3)))
    cats = defaultdict(int)
    for f in ev_cp['filas']:
        if f[0] in ('FN', 'FP'):
            cats[f[0] + ' ' + f[5]] += 1
    if cats:
        P("Detalle de errores: " + "; ".join("{}: {}".format(k, v) for k, v in sorted(cats.items())))
    P("")
    P("--- 2. ETIQUETAS DE SEGMENTOS ---")
    P("Segmentos totales: {} | evaluados: {} | micro-segmentos (< {} ciclos): {} | no identificables: {} | "
      "en banda de ambiguedad (+-{:g} ms): {}".format(ev_seg['n_total'], ev_seg['n'], args.min_ciclos_segmento,
                                                     ev_seg['n_micro'], ev_seg['n_noid'], args.banda_ms, ev_seg['n_amb']))
    P("Acierto (evaluados) {} | ponderado por ciclos {} | referencia sin filtros: {}".format(
        fmt(ev_seg['acc'], 3), fmt(ev_seg['acc_pond'], 3), fmt(ev_seg['acc_sin_filtros'], 3)))
    if ev_seg['n_amb']:
        P("  En banda de ambiguedad (aparte): {} segmentos, {} acertados".format(ev_seg['n_amb'], ev_seg['amb_ok']))
    if ev_seg['n_noid']:
        P("  No identificables (aparte): {} segmentos, {} acertados | error desviacion: {}".format(
            ev_seg['n_noid'], ev_seg['noid_ok'],
            ("mediana {} ms, MAE {} ms".format(fmt(ev_seg['dev_err_noid']['mediana'], 2), fmt(ev_seg['dev_err_noid']['mae'], 2))
             if ev_seg['dev_err_noid'] else '-')))
    if ev_seg['n_micro']:
        me = ev_seg['micro_err']
        P("  Micro-segmentos (aparte): {} ({:.2%} de los ciclos), {} acertados | error desviacion: {}".format(
            ev_seg['n_micro'], ev_seg['micro_ciclos'] / float(max(ev_seg['total_ciclos'], 1)), ev_seg['micro_ok'],
            ("mediana {} ms, media {} ms, MAE {} ms, max |.| {} ms".format(
                fmt(me['mediana'], 2), fmt(me['media'], 2), fmt(me['mae'], 2), fmt(me['max_abs'], 2)) if me else '-')))
    d = ev_seg['dev_err']
    if d:
        P("Error de la desviacion (script - verdad, ms; segmentos >= {} ciclos, sin micro): MEDIANA {} | media {} | MAE {} | "
          "max |.| {}".format(args.min_ciclos_segmento, fmt(d['mediana'], 2), fmt(d['media'], 2), fmt(d['mae'], 2),
                             fmt(d['max_abs'], 2)))
    P("Pureza media de los segmentos evaluados: {} | con pureza < 0.9 (mezclan niveles): {}".format(
        fmt(ev_seg['pureza_media'], 3), ev_seg['n_impuros']))
    esperados = sorted(ev_seg['conf'].keys())
    obtenidos = sorted(set(k for e_ in ev_seg['conf'].values() for k in e_))
    if esperados:
        P("Matriz de confusion (evaluados; filas = esperado, columnas = asignado por el script):")
        ancho = max(len(x) for x in esperados + obtenidos) + 2
        P("  " + " " * ancho + "".join(x.rjust(ancho) for x in obtenidos))
        for e_ in esperados:
            P("  " + e_.ljust(ancho) + "".join(str(ev_seg['conf'][e_].get(o_, 0)).rjust(ancho) for o_ in obtenidos))
    P("=" * 74)
    return "\n".join(L)


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Evalua la segmentacion HDP-HMM contra la verdad del generador de bloque")
    ap.add_argument('--verdad', required=True, help='verdad_ciclos_*.csv del generador')
    ap.add_argument('--interval', type=float, required=True, help='Cadencia en segundos (la misma del generador)')
    ap.add_argument('--change-points', required=True, help='change_points_*.csv del script HDP-HMM')
    ap.add_argument('--segmentos', required=True, help='segmentos_caracterizados_*.csv del script HDP-HMM')
    ap.add_argument('--bloques-csv', default=None, help='bloques_longitud_*.csv: chequea que haya un solo bloque (opcional)')
    ap.add_argument('--ciclos-excluidos', default=None,
                    help='ciclos_excluidos_*.csv de verificar_caminos (quita tambien esos ciclos de la verdad)')
    ap.add_argument('--tolerancia-ciclos', type=float, default=2.0,
                    help='Tolerancia de los puntos de cambio en ciclos (default 2: el script suaviza con min_dwell)')
    ap.add_argument('--jitter-s', type=float, default=5.0, help='Margen por jitter del reloj en segundos (default 5)')
    ap.add_argument('--tolerancia-s', type=float, default=None,
                    help='Tolerancia fija en segundos (reemplaza a --tolerancia-ciclos x interval + jitter)')
    ap.add_argument('--delta-min-ms', type=float, default=2.0,
                    help='Cambio de nivel minimo para exigir su deteccion (default 2 ms = umbral de moderada)')
    ap.add_argument('--min-meseta', type=int, default=5,
                    help='Ciclos minimos de cada nivel para exigir el cambio (default 5 = --min-dwell de la corrida de referencia)')
    ap.add_argument('--min-bloque', type=int, default=10,
                    help='Ciclos minimos de un bloque (default 10, como el script); solo para el chequeo del bloque')
    ap.add_argument('--min-frac-base', type=float, default=0.25,
                    help='Fraccion minima de ciclos en regimen base para que el P25 de referencia sea identificable (default 0.25)')
    ap.add_argument('--banda-ms', type=float, default=0.5,
                    help='Banda de ambiguedad alrededor de los umbrales de severidad (default 0.5 ms: el error maximo '
                         'de la desviacion en segmentos evaluados de la corrida de referencia fue 0.51 ms)')
    ap.add_argument('--tolerancia-ampliada-ciclos', type=float, default=5.0,
                    help='Segunda pasada de emparejamiento, en ciclos (default 5): lo que queda sin emparejar y '
                         'aparece a esa distancia es el mismo evento detectado con desfase')
    ap.add_argument('--min-ciclos-segmento', type=int, default=10,
                    help='Segmentos con menos ciclos son "micro-segmentos" y se evaluan aparte (default 10)')
    ap.add_argument('--umbral-severa', type=float, default=5.0)
    ap.add_argument('--umbral-moderada', type=float, default=2.0)
    ap.add_argument('--output-dir', default='evaluacion')
    ap.add_argument('--etiqueta', default='', help='Nombre de la corrida (p. ej. semilla42); sufija los archivos')
    ap.add_argument('--resumen-csv', default=None, help='Agrega UNA fila con las metricas clave a este CSV')
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("--interval debe ser mayor que 0")
    args.tol_s = args.tolerancia_s if args.tolerancia_s is not None else args.tolerancia_ciclos * args.interval + args.jitter_s
    args.tol_amp_s = max(args.tolerancia_ampliada_ciclos * args.interval + args.jitter_s, args.tol_s)

    excluidos = cargar_excluidos(args.ciclos_excluidos)
    serie = cargar_verdad(args.verdad, excluidos)
    if len(serie) < 2:
        sys.stderr.write("[ERROR] La verdad no tiene ciclos evaluables.\n"); sys.exit(1)
    info = identificabilidad(serie, args.min_frac_base)
    chq = chequeo_bloque(args.bloques_csv, serie, args) if args.bloques_csv else None

    cps_v, sub_v = derivar_eventos_verdad(serie, args)
    args._subumbral = sub_v
    ev_cp = evaluar_cps(cps_v, leer_csv(args.change_points), serie, args)
    ev_seg = evaluar_segmentos(leer_csv(args.segmentos), serie, info, args)

    texto = reporte(args, serie, info, chq, ev_cp, ev_seg, len(sub_v))
    print(texto)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    suf = ("_" + args.etiqueta) if args.etiqueta else ""

    def escribir(nombre, cab, filas):
        p = os.path.join(args.output_dir, nombre + suf + ".csv")
        with abrir(p, 'w') as f:
            w = csv.writer(f); w.writerow(cab); w.writerows(filas)
        return p
    escribir('evaluacion_puntos_de_cambio',
             ['resultado', 'timestamp_verdad', 'timestamp_detectado', 'desfase_ciclos', 'delta_verdad_ms',
              'categoria'], ev_cp['filas'])
    escribir('evaluacion_segmentos',
             ['start_ts', 'end_ts', 'n_ciclos', 'duracion_min', 'longitud', 'delta_verdad_ms',
              'mediana_rtt_ms', 'desviacion_script_ms', 'error_desviacion_ms', 'tipo_esperado', 'tipo_script',
              'coincide', 'pureza', 'estado'], ev_seg['filas'])
    with open(os.path.join(args.output_dir, 'evaluacion_resumen' + suf + '.txt'), 'w') as f:
        f.write(texto + "\n")

    if args.resumen_csv:
        cab = ['etiqueta', 'ciclos', 'frac_base', 'identificable', 'bloques', 'ciclos_fuera_bloque',
               'cp_verdad_eval', 'cp_vp', 'cp_vp_eval', 'cp_fn_eval', 'cp_desfase_eval', 'cp_fn_real', 'cp_fp',
               'cp_fp_espurios', 'cp_precision', 'cp_precision_depurada', 'cp_recall_eval', 'cp_recall_ampliado',
               'cp_desfase_mediana', 'cp_desfase_p90', 'seg_total', 'seg_n', 'seg_micro', 'seg_no_identificables',
               'seg_ambiguos', 'seg_acierto', 'seg_acierto_pond', 'seg_acierto_sin_filtros', 'seg_micro_acierto',
               'seg_desviacion_mediana', 'seg_desviacion_mae']
        o = ev_cp['offsets'] or {}
        d = ev_seg['dev_err'] or {}
        c = chq or {}
        fila = [args.etiqueta, len(serie), fmt(info['frac_base'], 3), int(info['identificable']),
                c.get('n_bloques', ''), c.get('ciclos_fuera', ''),
                ev_cp['n_eval'], ev_cp['vp'], ev_cp['vp_eval'], ev_cp['fn_eval'], ev_cp['desfase_eval'],
                ev_cp['fn_real'], ev_cp['fp'], ev_cp['espurias'], fmt(ev_cp['precision'], 3),
                fmt(ev_cp['precision_depurada'], 3), fmt(ev_cp['recall_eval'], 3), fmt(ev_cp['recall_ampliado'], 3),
                fmt(o.get('mediana'), 2), fmt(o.get('p90'), 2), ev_seg['n_total'], ev_seg['n'], ev_seg['n_micro'],
                ev_seg['n_noid'], ev_seg['n_amb'], fmt(ev_seg['acc'], 3), fmt(ev_seg['acc_pond'], 3),
                fmt(ev_seg['acc_sin_filtros'], 3),
                fmt(ev_seg['micro_ok'] / float(ev_seg['n_micro']), 3) if ev_seg['n_micro'] else '',
                fmt(d.get('mediana'), 2), fmt(d.get('mae'), 2)]
        # si el archivo existe con otras columnas, la fila va a <nombre>_v2, _v3... (el primero con las mismas columnas)
        destino, k = args.resumen_csv, 1
        while os.path.exists(destino):
            with abrir(destino, 'r') as f:
                primera = next(csv.reader(f), None)
            if primera == cab:
                break
            k += 1
            base, ext = os.path.splitext(args.resumen_csv)
            destino = "{}_v{}{}".format(base, k, ext)
        if destino != args.resumen_csv:
            print("[AVISO] {} tiene columnas de otra version; esta fila va a {}".format(args.resumen_csv, destino))
        existe = os.path.exists(destino)
        with abrir(destino, 'a') as f:
            w = csv.writer(f)
            if not existe:
                w.writerow(cab)
            w.writerow(fila)
    print("Archivos escritos en: {}".format(args.output_dir))


if __name__ == '__main__':
    main()
