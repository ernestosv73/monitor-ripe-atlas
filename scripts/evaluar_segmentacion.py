# -*- coding: utf-8 -*-
"""
=============================================================================
evaluar_segmentacion.py  (v1)
Evalua la salida del script HDP-HMM contra la VERDAD del generador sintetico.
=============================================================================
Compara tres cosas, siempre contra verdad_ciclos_*.csv del generador:

  1. PUNTOS DE CAMBIO internos (change_points_*.csv): cambios de nivel del RTT del ultimo salto DENTRO de una misma
     ruta. Metricas: verdaderos positivos (VP), falsos negativos (FN), falsos positivos (FP), precision, recall,
     F1 y desfase en ciclos. Se reporta tambien la sensibilidad a la tolerancia.
  2. CAMBIOS DE RUTA (cambios_de_ruta_*.csv, opcional): frontera entre bloques. Metricas: VP/FN/FP, error del
     delta de RTT medido y acierto del tipo (mejora_*/degradacion_*).
  3. ETIQUETAS de los segmentos (segmentos_caracterizados_*.csv): el tipo que asigno el script contra el que
     corresponde segun la verdad (mediana de delta_ms del segmento, con la misma regla de severidad), matriz de
     confusion y error de la desviacion respecto de la referencia de ruta.

Que se considera "verdad":
  * Serie evaluada = ciclos del generador que llegaron al destino (sin_destino = 0) y no son huecos; si se da
    --ciclos-excluidos tambien se quitan esos ciclos (igual que hace el script HDP-HMM).
  * Punto de cambio verdadero = dos ciclos consecutivos de la misma ruta con |nivel_ultimo_ms| distinto en al menos
    --delta-min-ms. Los cambios menores no se exigen (se informan aparte).
  * Cambio de ruta verdadero = dos ciclos consecutivos con distinto ruta_id.

Que NO se exige (FN "no evaluables", se informan pero no penalizan el recall principal):
  * 'bloque_corto'  : la ruta/bloque dura menos de --min-bloque ciclos; el script omite esos bloques.
  * 'meseta_corta'  : alguno de los dos niveles dura menos de --min-meseta ciclos; el suavizado (min_dwell) lo borra.
  * 'misma_longitud': la ruta nueva tiene el mismo numero de saltos; los bloques se definen por longitud.

Uso tipico (una corrida):
  python evaluar_segmentacion.py \\
      --verdad  sintetico_m90001_p1/verdad_ciclos_measurement90001_probe1.csv \\
      --change-points  resultados_hdphmm/change_points_measurement90001_probe1.csv \\
      --segmentos      resultados_hdphmm/segmentos_caracterizados_measurement90001_probe1.csv \\
      --cambios-ruta   resultados_hdphmm/cambios_de_ruta_measurement90001_probe1.csv \\
      --ciclos-excluidos vc/ciclos_excluidos_measurement90001_probe1.csv \\
      --interval 900 --output-dir evaluacion --etiqueta semilla42 --resumen-csv comparacion.csv

Varias semillas: ejecutar la cadena generador -> verificar_caminos -> HDP-HMM -> este script para cada semilla,
con la misma --resumen-csv; cada corrida agrega UNA fila (parametro --etiqueta) a ese CSV.

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


def tipo_cambio_rtt(delta, sev, mod):
    mag = abs(delta)
    if mag > sev: nivel = 'severa'
    elif mag > mod: nivel = 'moderada'
    else: return 'sin_cambio_relevante'
    return ('degradacion_' if delta > 0 else 'mejora_') + nivel


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


# ============================================================================
# VERDAD
# ============================================================================
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
        serie.append({'ts': ts, 't': a_seg(ts), 'ruta': r['ruta_id'].strip(), 'nivel': nivel,
                      'delta': a_float(r.get('delta_ms'), 0.0) or 0.0,
                      'spike': a_float(r.get('spike_ms'), 0.0) or 0.0,
                      'nsaltos': int(a_float(r.get('n_saltos_emitidos'), 0))})
    return serie


def cargar_excluidos(path):
    if not path:
        return set()
    return set(r['ciclo'].strip() for r in leer_csv(path) if (r.get('ciclo') or '').strip())


def derivar_eventos_verdad(serie, args):
    """Devuelve (cambios_ruta, cps_internos, cambios_subumbral) de la verdad."""
    n = len(serie)
    # runs de ruta: inicio y fin (indices) de cada tramo consecutivo con la misma ruta
    run_ini, run_fin, k = [0] * n, [0] * n, 0
    while k < n:
        j = k
        while j + 1 < n and serie[j + 1]['ruta'] == serie[k]['ruta']:
            j += 1
        for m in range(k, j + 1):
            run_ini[m], run_fin[m] = k, j
        k = j + 1
    # fronteras de nivel (cualquier cambio de ruta o de nivel >= 0.05 ms)
    front = [0]
    for i in range(1, n):
        if serie[i]['ruta'] != serie[i - 1]['ruta'] or abs(serie[i]['nivel'] - serie[i - 1]['nivel']) >= 0.05:
            front.append(i)
    front.append(n)
    pos_front = dict((f, idx) for idx, f in enumerate(front))

    cambios_ruta, cps, subumbral = [], [], []
    for i in range(1, n):
        a, b = serie[i - 1], serie[i]
        if a['ruta'] != b['ruta']:
            run_prev = run_fin[i - 1] - run_ini[i - 1] + 1
            run_next = run_fin[i] - run_ini[i] + 1
            w = min(args.ventana_borde, run_prev, run_next)
            antes = float(np.median([s['nivel'] for s in serie[i - w:i]]))
            despues = float(np.median([s['nivel'] for s in serie[i:i + w]]))
            if min(run_prev, run_next) < args.min_bloque: cat = 'bloque_corto'
            elif a['nsaltos'] == b['nsaltos']: cat = 'misma_longitud'
            else: cat = ''
            cambios_ruta.append({'ts': b['ts'], 't': b['t'], 'de': a['ruta'], 'a': b['ruta'], 'rtt_antes': antes,
                                 'rtt_despues': despues, 'delta': despues - antes,
                                 'tipo': tipo_cambio_rtt(despues - antes, args.umbral_severa, args.umbral_moderada),
                                 'estructural': cat, 'ventana': w})
        elif abs(b['nivel'] - a['nivel']) >= 0.05:
            d = b['nivel'] - a['nivel']
            f = pos_front[i]
            largo_antes = i - front[f - 1]
            largo_despues = front[f + 1] - i
            largo_bloque = run_fin[i] - run_ini[i] + 1
            ev = {'ts': b['ts'], 't': b['t'], 'delta': d, 'ruta': b['ruta'], 'largo_antes': largo_antes,
                  'largo_despues': largo_despues, 'estructural': ''}
            if abs(d) < args.delta_min_ms:
                subumbral.append(ev)
                continue
            if largo_bloque < args.min_bloque: ev['estructural'] = 'bloque_corto'
            elif min(largo_antes, largo_despues) < args.min_meseta: ev['estructural'] = 'meseta_corta'
            cps.append(ev)
    return cambios_ruta, cps, subumbral


# ============================================================================
# EVALUACION 1 y 2: PUNTOS DE CAMBIO / CAMBIOS DE RUTA
# ============================================================================
def metricas_pr(vp, fn_eval, fp):
    prec = vp / (vp + fp) if (vp + fp) else None
    rec = vp / (vp + fn_eval) if (vp + fn_eval) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return prec, rec, f1


def evaluar_cps(cps_verdad, det_rows, serie, args, interval):
    t_v = [c['t'] for c in cps_verdad]
    t_d = [a_seg(r['timestamp']) for r in det_rows]
    tol_s = args.tolerancia_ciclos * interval
    pares = emparejar(t_v, t_d, tol_s)
    par_v = dict(pares)
    usados_d = set(j for _, j in pares)

    t_spikes = [s['t'] for s in serie if s['spike'] > 0]
    filas = []
    offsets = []
    for i, c in enumerate(cps_verdad):
        if i in par_v:
            j = par_v[i]
            off = (t_d[j] - c['t']) / interval
            offsets.append(off)
            filas.append(['VP', c['ts'], det_rows[j]['timestamp'], fmt(off, 1), fmt(c['delta']), '', c['ruta']])
        else:
            cat = c['estructural'] or 'no_detectado'
            filas.append(['FN', c['ts'], '', '', fmt(c['delta']), cat, c['ruta']])
    t_sub = [c['t'] for c in args._subumbral]
    for j, r in enumerate(det_rows):
        if j in usados_d:
            continue
        td = t_d[j]
        if any(abs(td - ts) <= 2 * interval for ts in t_spikes): cat = 'cerca_de_spike'
        elif any(abs(td - ts) <= tol_s for ts in t_sub): cat = 'cambio_subumbral'
        else: cat = 'sin_causa_identificada'
        filas.append(['FP', '', r['timestamp'], '', '', cat, ''])
    n_vp = len(pares)
    fn_total = len(cps_verdad) - n_vp
    fn_estruct = sum(1 for i, c in enumerate(cps_verdad) if i not in par_v and c['estructural'])
    n_eval = len(cps_verdad) - sum(1 for c in cps_verdad if c['estructural'])
    fp = len(det_rows) - n_vp
    # sensibilidad a la tolerancia
    sens = []
    for tc in sorted(set(TOLERANCIAS_SENSIBILIDAD) | set([args.tolerancia_ciclos])):
        pr = emparejar(t_v, t_d, tc * interval)
        vp_t = len(pr)
        usados_v_t = set(i for i, _ in pr)
        fn_eval_t = sum(1 for i, c in enumerate(cps_verdad) if i not in usados_v_t and not c['estructural'])
        p, r_, f1 = metricas_pr(vp_t, fn_eval_t, len(det_rows) - vp_t)
        sens.append((tc, vp_t, fn_eval_t, len(det_rows) - vp_t, p, r_, f1))
    # los VP que son de eventos 'estructurales' (detectados igual) no cuentan como FN
    prec, rec, f1 = metricas_pr(n_vp, fn_total - fn_estruct, fp)
    return {'filas': filas, 'vp': n_vp, 'fn': fn_total, 'fn_estructural': fn_estruct, 'fp': fp,
            'n_verdad': len(cps_verdad), 'n_eval': n_eval, 'n_det': len(det_rows), 'precision': prec,
            'recall_eval': rec, 'recall_todos': (n_vp / len(cps_verdad)) if cps_verdad else None, 'f1': f1,
            'offsets': stats(offsets), 'sens': sens}


def evaluar_rutas(rutas_verdad, det_rows, args, interval):
    t_v = [c['t'] for c in rutas_verdad]
    t_d = [a_seg(r['timestamp']) for r in det_rows]
    pares = emparejar(t_v, t_d, args.tolerancia_ciclos * interval)
    par_v = dict(pares)
    usados_d = set(j for _, j in pares)
    filas, err, offsets, acierto = [], [], [], []
    for i, c in enumerate(rutas_verdad):
        base = [c['de'] + '->' + c['a']]
        if i in par_v:
            r = det_rows[par_v[i]]
            d_det = a_float(r['delta_rtt_ms'])
            e = d_det - c['delta']
            err.append(e)
            off = (t_d[par_v[i]] - c['t']) / interval
            offsets.append(off)
            ok = (r['tipo_cambio'] == c['tipo'])
            acierto.append(ok)
            filas.append(['VP'] + base + [c['ts'], r['timestamp'], fmt(off, 1), fmt(c['delta']), fmt(d_det), fmt(e),
                                          c['tipo'], r['tipo_cambio'], 'si' if ok else 'no', ''])
        else:
            filas.append(['FN'] + base + [c['ts'], '', '', fmt(c['delta']), '', '', c['tipo'], '', '',
                                          c['estructural'] or 'no_detectado'])
    for j, r in enumerate(det_rows):
        if j not in usados_d:
            filas.append(['FP', '', '', r['timestamp'], '', '', fmt(a_float(r['delta_rtt_ms'])), '', '',
                          r['tipo_cambio'], '', 'sin_cambio_de_ruta_en_la_verdad'])
    vp = len(pares)
    fn_est = sum(1 for i, c in enumerate(rutas_verdad) if i not in par_v and c['estructural'])
    fn_eval = sum(1 for i, c in enumerate(rutas_verdad) if i not in par_v and not c['estructural'])
    fp = len(det_rows) - vp
    prec, rec, f1 = metricas_pr(vp, fn_eval, fp)
    return {'filas': filas, 'vp': vp, 'fn': len(rutas_verdad) - vp, 'fn_estructural': fn_est, 'fp': fp,
            'n_verdad': len(rutas_verdad), 'n_eval': len(rutas_verdad) - sum(1 for c in rutas_verdad if c['estructural']),
            'precision': prec, 'recall_eval': rec, 'f1': f1, 'err_delta': stats(err), 'offsets': stats(offsets),
            'acierto_tipo': (sum(acierto) / len(acierto)) if acierto else None}


# ============================================================================
# EVALUACION 3: ETIQUETAS DE SEGMENTOS
# ============================================================================
def evaluar_segmentos(seg_rows, serie, args):
    ts_lista = [s['ts'] for s in serie]
    filas, aciertos, pesos, dev_err, purezas = [], [], [], [], []
    conf = defaultdict(lambda: defaultdict(int))
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
        esperado = tipo_esperado(d_med, dur, args.umbral_severa, args.umbral_moderada)
        obtenido = r['tipo_degradacion'].strip()
        ok = (esperado == obtenido)
        dev = a_float(r.get('desviacion_ms'))
        err = (dev - d_med) if dev is not None else None
        if err is not None:
            dev_err.append(err)
        aciertos.append(ok); pesos.append(n); purezas.append(pureza)
        conf[esperado][obtenido] += 1
        filas.append([r['start_ts'], r['end_ts'], n, fmt(dur, 0), r.get('longitud', ''), fmt(d_med),
                      fmt(a_float(r.get('mediana'))), fmt(dev), fmt(err), esperado, obtenido,
                      'si' if ok else 'no', fmt(pureza, 2)])
    n_seg = len(aciertos)
    return {'filas': filas, 'n': n_seg, 'acc': (sum(aciertos) / n_seg) if n_seg else None,
            'acc_pond': (sum(p for p, a in zip(pesos, aciertos) if a) / sum(pesos)) if sum(pesos) else None,
            'dev_err': stats(dev_err), 'pureza_media': float(np.mean(purezas)) if purezas else None,
            'n_impuros': sum(1 for p in purezas if p < 0.9), 'conf': conf}


# ============================================================================
# REPORTE
# ============================================================================
def reporte(args, serie, ev_cp, ev_ruta, ev_seg, n_sub):
    L = []
    P = L.append
    P("=" * 72)
    P("EVALUACION DE SEGMENTACION HDP-HMM CONTRA LA VERDAD SINTETICA" + ((" [" + args.etiqueta + "]") if args.etiqueta else ""))
    P("=" * 72)
    P("Ciclos evaluados (con destino, sin huecos ni excluidos): {}".format(len(serie)))
    P("Tolerancia de emparejamiento: {} ciclos ({:g} s) | delta minimo exigido: {:g} ms | min. bloque: {} | "
      "min. meseta: {} | ventana de borde: {}".format(args.tolerancia_ciclos, args.tolerancia_ciclos * args.interval,
                                                      args.delta_min_ms, args.min_bloque, args.min_meseta,
                                                      args.ventana_borde))
    P("")
    P("--- 1. PUNTOS DE CAMBIO INTERNOS (dentro de una misma ruta) ---")
    P("Verdad: {} (evaluables: {}; no evaluables: {}) | detectados: {} | cambios sub-umbral ignorados: {}".format(
        ev_cp['n_verdad'], ev_cp['n_eval'], ev_cp['n_verdad'] - ev_cp['n_eval'], ev_cp['n_det'], n_sub))
    P("VP {} | FN {} (de ellos no evaluables: {}) | FP {}".format(ev_cp['vp'], ev_cp['fn'], ev_cp['fn_estructural'],
                                                                 ev_cp['fp']))
    P("Precision {} | Recall (evaluables) {} | Recall (todos) {} | F1 {}".format(
        fmt(ev_cp['precision'], 3), fmt(ev_cp['recall_eval'], 3), fmt(ev_cp['recall_todos'], 3), fmt(ev_cp['f1'], 3)))
    o = ev_cp['offsets']
    if o:
        P("Desfase (ciclos, detectado - verdad): media {} | mediana {} | p90 |.| {} | max |.| {}".format(
            fmt(o['media'], 2), fmt(o['mediana'], 2), fmt(o['p90'], 2), fmt(o['max_abs'], 2)))
    P("Sensibilidad a la tolerancia (ciclos): VP / FN eval / FP / precision / recall eval")
    for tc, vp, fn, fp, p, r, f1 in ev_cp['sens']:
        P("  {:>3}: {:>4} / {:>4} / {:>4} / {} / {}".format(tc, vp, fn, fp, fmt(p, 3), fmt(r, 3)))
    cats = defaultdict(int)
    for f in ev_cp['filas']:
        if f[0] in ('FN', 'FP'):
            cats[f[0] + ' ' + f[5]] += 1
    if cats:
        P("Detalle de errores: " + "; ".join("{}: {}".format(k, v) for k, v in sorted(cats.items())))
    if ev_ruta is not None:
        P("")
        P("--- 2. CAMBIOS DE RUTA (frontera entre bloques) ---")
        P("Verdad: {} (evaluables: {}) | VP {} | FN {} (no evaluables: {}) | FP {}".format(
            ev_ruta['n_verdad'], ev_ruta['n_eval'], ev_ruta['vp'], ev_ruta['fn'], ev_ruta['fn_estructural'],
            ev_ruta['fp']))
        P("Precision {} | Recall (evaluables) {} | F1 {}".format(fmt(ev_ruta['precision'], 3),
                                                                 fmt(ev_ruta['recall_eval'], 3), fmt(ev_ruta['f1'], 3)))
        e = ev_ruta['err_delta']
        if e:
            P("Error del delta de RTT medido (medido - verdad, ms): sesgo {} | MAE {} | max |.| {}".format(
                fmt(e['media'], 2), fmt(e['mae'], 2), fmt(e['max_abs'], 2)))
        if ev_ruta['acierto_tipo'] is not None:
            P("Acierto del tipo de cambio (mejora/degradacion y severidad): {:.1%}".format(ev_ruta['acierto_tipo']))
    P("")
    P("--- 3. ETIQUETAS DE SEGMENTOS ---")
    P("Segmentos evaluados: {} | acierto {} | acierto ponderado por ciclos {}".format(
        ev_seg['n'], fmt(ev_seg['acc'], 3), fmt(ev_seg['acc_pond'], 3)))
    d = ev_seg['dev_err']
    if d:
        P("Error de la desviacion respecto de la referencia de ruta (script - verdad, ms): sesgo {} | MAE {} | max |.| {}".format(
            fmt(d['media'], 2), fmt(d['mae'], 2), fmt(d['max_abs'], 2)))
    P("Pureza media de los segmentos: {} | segmentos con pureza < 0.9 (mezclan niveles): {}".format(
        fmt(ev_seg['pureza_media'], 3), ev_seg['n_impuros']))
    esperados = sorted(ev_seg['conf'].keys())
    obtenidos = sorted(set(k for e_ in ev_seg['conf'].values() for k in e_))
    if esperados:
        P("Matriz de confusion (filas = esperado segun la verdad, columnas = asignado por el script):")
        ancho = max(len(x) for x in esperados + obtenidos) + 2
        P("  " + " " * ancho + "".join(x.rjust(ancho) for x in obtenidos))
        for e_ in esperados:
            P("  " + e_.ljust(ancho) + "".join(str(ev_seg['conf'][e_].get(o_, 0)).rjust(ancho) for o_ in obtenidos))
    P("=" * 72)
    return "\n".join(L)


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Evalua la segmentacion HDP-HMM contra la verdad del generador sintetico")
    ap.add_argument('--verdad', required=True, help='verdad_ciclos_*.csv del generador')
    ap.add_argument('--change-points', required=True, help='change_points_*.csv del script HDP-HMM')
    ap.add_argument('--segmentos', required=True, help='segmentos_caracterizados_*.csv del script HDP-HMM')
    ap.add_argument('--cambios-ruta', default=None, help='cambios_de_ruta_*.csv del script HDP-HMM (opcional)')
    ap.add_argument('--ciclos-excluidos', default=None,
                    help='ciclos_excluidos_*.csv de verificar_caminos (quita tambien esos ciclos de la verdad)')
    ap.add_argument('--interval', type=float, required=True, help='Cadencia en segundos (la misma del generador)')
    ap.add_argument('--tolerancia-ciclos', type=float, default=2.0,
                    help='Tolerancia de emparejamiento en ciclos (default 2: el script suaviza con min_dwell=3)')
    ap.add_argument('--delta-min-ms', type=float, default=2.0,
                    help='Cambio de nivel minimo para exigir su deteccion (default 2 ms = umbral de moderada)')
    ap.add_argument('--min-meseta', type=int, default=3,
                    help='Ciclos minimos de cada nivel para exigir el cambio (default 3 = --min-dwell del script)')
    ap.add_argument('--min-bloque', type=int, default=10,
                    help='Ciclos minimos de un bloque/ruta para exigir su deteccion (default 10, como el script)')
    ap.add_argument('--ventana-borde', type=int, default=10,
                    help='Igual que --ventana-borde del script HDP-HMM (default 10)')
    ap.add_argument('--umbral-severa', type=float, default=5.0)
    ap.add_argument('--umbral-moderada', type=float, default=2.0)
    ap.add_argument('--output-dir', default='evaluacion')
    ap.add_argument('--etiqueta', default='', help='Nombre de la corrida (p. ej. semilla42); sufija los archivos')
    ap.add_argument('--resumen-csv', default=None, help='Agrega UNA fila con las metricas clave a este CSV')
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("--interval debe ser mayor que 0")

    excluidos = cargar_excluidos(args.ciclos_excluidos)
    serie = cargar_verdad(args.verdad, excluidos)
    if len(serie) < 2:
        sys.stderr.write("[ERROR] La verdad no tiene ciclos evaluables.\n"); sys.exit(1)

    rutas_v, cps_v, sub_v = derivar_eventos_verdad(serie, args)
    args._subumbral = sub_v

    cp_rows = leer_csv(args.change_points)
    ev_cp = evaluar_cps(cps_v, cp_rows, serie, args, args.interval)
    ev_ruta = None
    if args.cambios_ruta:
        ev_ruta = evaluar_rutas(rutas_v, leer_csv(args.cambios_ruta), args, args.interval)
    ev_seg = evaluar_segmentos(leer_csv(args.segmentos), serie, args)

    texto = reporte(args, serie, ev_cp, ev_ruta, ev_seg, len(sub_v))
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
              'categoria', 'ruta'], ev_cp['filas'])
    if ev_ruta is not None:
        escribir('evaluacion_cambios_de_ruta',
                 ['resultado', 'ruta', 'timestamp_verdad', 'timestamp_detectado', 'desfase_ciclos',
                  'delta_verdad_ms', 'delta_medido_ms', 'error_ms', 'tipo_verdad', 'tipo_detectado', 'tipo_ok',
                  'categoria'], ev_ruta['filas'])
    escribir('evaluacion_segmentos',
             ['start_ts', 'end_ts', 'n_ciclos', 'duracion_min', 'longitud', 'delta_verdad_ms', 'mediana_rtt_ms',
              'desviacion_script_ms', 'error_desviacion_ms', 'tipo_esperado', 'tipo_script', 'coincide', 'pureza'],
             ev_seg['filas'])
    with open(os.path.join(args.output_dir, 'evaluacion_resumen' + suf + '.txt'), 'w') as f:
        f.write(texto + "\n")

    if args.resumen_csv:
        existe = os.path.exists(args.resumen_csv)
        cab = ['etiqueta', 'ciclos', 'cp_verdad_eval', 'cp_vp', 'cp_fn_eval', 'cp_fp', 'cp_precision', 'cp_recall_eval',
               'cp_desfase_mediana', 'cp_desfase_p90', 'rutas_verdad_eval', 'rutas_vp', 'rutas_fn_eval', 'rutas_fp',
               'rutas_recall_eval', 'rutas_acierto_tipo', 'seg_n', 'seg_acierto', 'seg_acierto_pond',
               'seg_desviacion_sesgo']
        o = ev_cp['offsets'] or {}
        r = ev_ruta or {}
        d = ev_seg['dev_err'] or {}
        fila = [args.etiqueta, len(serie), ev_cp['n_eval'], ev_cp['vp'], ev_cp['fn'] - ev_cp['fn_estructural'],
                ev_cp['fp'], fmt(ev_cp['precision'], 3), fmt(ev_cp['recall_eval'], 3), fmt(o.get('mediana'), 2),
                fmt(o.get('p90'), 2), r.get('n_eval', ''), r.get('vp', ''),
                (r['fn'] - r['fn_estructural']) if r else '', r.get('fp', ''),
                fmt(r.get('recall_eval'), 3) if r else '', fmt(r.get('acierto_tipo'), 3) if r else '',
                ev_seg['n'], fmt(ev_seg['acc'], 3), fmt(ev_seg['acc_pond'], 3), fmt(d.get('media'), 2)]
        with abrir(args.resumen_csv, 'a') as f:
            w = csv.writer(f)
            if not existe:
                w.writerow(cab)
            w.writerow(fila)
    print("Archivos escritos en: {}".format(args.output_dir))


if __name__ == '__main__':
    main()
