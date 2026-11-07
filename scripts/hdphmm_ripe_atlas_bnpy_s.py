# -*- coding: utf-8 -*-
"""
=============================================================================
HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas (v5)
+ Análisis por BLOQUES (Lectura directa de CSV de bloques)
+ Caracterización de Segmentos y Puntos de Cambio
+ Fusión de Estados (Automática por Bhattacharyya)
+ Agrupamiento de Segmentos Consecutivos por Tipo de Degradación
=============================================================================
Cambios respecto de v5:
  1. --interval (segundos entre ciclos, obligatorio): duracion_minutos = n_ciclos * interval / 60
     (antes estaba fijo en 10 min por ciclo).
  2. --ciclos-excluidos (obligatorio): CSV ciclos_excluidos_measurement<M>_probe<P>.csv de verificar_caminos.py.
     Esos ciclos (destino no alcanzado) se quitan de la serie, y cada subserie se arma por el rango de tiempo
     [inicio, fin] del bloque y no por una cantidad de ciclos, que se desfasaba al contar ciclos excluidos.
  2b. El CSV de excluidos trae una columna 'motivo' (sin_destino, parpadeo, transitorio). Se quitan de la serie
     todos los ciclos del archivo; para verificar la ventana de cada bloque se descuentan de su 'ciclos' los
     parpadeos y transitorios que el CSV ubica en ese bloque (bloque_id). Los sin_destino no se descuentan, porque
     el CSV de bloques no los cuenta.
  3. --percentil-outliers (default 0 = desactivado): el filtro de percentil por bloque ya no se aplica por defecto,
     porque descartaba los ciclos de RTT mas alto, que en una degradacion sostenida son los del propio evento
     (truncaba la meseta y subestimaba su magnitud). Los ciclos sin destino ya se quitan con --ciclos-excluidos.
  4. La clasificacion de severidad usa la MEDIANA del segmento (no la media): los spikes inflaban la media y
     cambiaban la etiqueta del segmento, partiendo grupos. Resto de la regla (umbrales, P25 global) sin cambios.
  5. Se elimina la comparacion con eventos de Path Analysis (--categorias, cargar_categorias,
     detectar_cambios_topologicos, TOLERANCIA_MINUTOS y las columnas evento_topologico / categorias_topologicas).
     Esos eventos se basan en cambios de RTT entre ciclos consecutivos (light/medium/dark), mientras que este script
     busca puntos de cambio sostenidos dentro de cada bloque: no son comparables. El archivo de puntos de cambio pasa
     a llamarse change_points_*.csv (columnas: timestamp, estado_anterior, estado_nuevo).
  6. La severidad ya no se mide contra un unico P25 global de toda la serie. Cada segmento se compara contra el
     RTT de referencia de SU RUTA (P25 de los ciclos de todos los bloques con la misma longitud), y la clasificacion
     tiene signo: una mediana por debajo de la referencia es 'mejora_*', no 'degradacion_*' (antes abs() las
     mezclaba: un cambio de ruta que MEJORABA el RTT salia como degradacion severa). Ademas, el cambio de ruta entre
     bloques consecutivos se clasifica como evento propio (mediana de los ultimos/primeros --ventana-borde ciclos de
     cada bloque, con signo) y se exporta en cambios_de_ruta_*.csv. Los segmentos consecutivos del mismo tipo solo se
     agrupan si son de la misma longitud. segmentos_caracterizados_*.csv agrega bloque_id, longitud, rtt_referencia,
     desviacion_ms al final. 'Ruta' = longitud del bloque (verificar_caminos.py no entrega un identificador de ruta).
Basado en: Mouchet et al., "Large-Scale Characterization and Segmentation of
           Internet Path Delays with Infinite HMMs" (arXiv:1910.12714)

Entorno: Python 2.7 + bnpy (build 8019474-py27_1) + numpy 1.16.6 + pandas
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import argparse
import csv
import re
import warnings
from collections import defaultdict
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.stderr.write("Este script necesita numpy y pandas instalados.\n")
    sys.exit(1)


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

MAPA_MACRO_ESTADOS_DEFAULT = {
    10: 0, 13: 0, 14: 0,  # Macro 0: Base Óptimo
    0: 1, 1: 1,            # Macro 1: Degradación Leve
    2: 2,                  # Macro 2: Congestión
    5: 3,                  # Macro 3: Outlier
}


# Umbrales de magnitud (ms) para severidad, tanto de segmentos como de cambios de ruta
UMBRAL_SEVERA_MS = 5.0
UMBRAL_MODERADA_MS = 2.0


# ============================================================================
# UTILIDADES
# ============================================================================
def extraer_ids_de_archivo(filename):
    basename = os.path.basename(filename)
    match = re.search(r'measurement(\d+)_probe(\d+)', basename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def cargar_mapa_macro_estados(ruta_archivo):
    if not ruta_archivo or not os.path.exists(ruta_archivo):
        return dict(MAPA_MACRO_ESTADOS_DEFAULT)
    mapa = {}
    with open(ruta_archivo, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    mapa[int(row[0].strip())] = int(row[1].strip())
                except ValueError:
                    continue
    return mapa


# ============================================================================
# FASE 1: PREPROCESAMIENTO DE DATOS
# ============================================================================
def cargar_ciclos_excluidos(path_csv):
    """
    Lee el CSV ciclos_excluidos_*.csv generado por verificar_caminos.py. Columnas usadas:
      ciclo     : timestamp del ciclo (se quita de la serie, cualquiera sea el motivo)
      motivo    : sin_destino | parpadeo | transitorio (si falta la columna, todos se tratan como sin_destino)
      bloque_id : bloque que contiene al ciclo (puede estar vacio si cae entre dos bloques)
    Devuelve (excluidos, por_bloque):
      excluidos : set de timestamps a quitar de la serie
      por_bloque: {bloque_id: cantidad de parpadeos/transitorios dentro del bloque}. El CSV de bloques cuenta esos
                  ciclos en 'ciclos' (no los sin_destino), por lo que hay que descontarlos al verificar la ventana.
    """
    print("[FASE 1a] Cargando ciclos excluidos desde: {}".format(path_csv))
    excluidos = set()
    por_bloque = defaultdict(int)
    por_motivo = defaultdict(int)
    with open(path_csv, 'r') as f:
        reader = csv.DictReader(f)
        campos = reader.fieldnames or []
        if 'ciclo' not in campos:
            sys.stderr.write("[ERROR] {} no tiene la columna 'ciclo'. Debe ser el ciclos_excluidos_*.csv de "
                             "verificar_caminos.py.\n".format(path_csv))
            sys.exit(1)
        if 'motivo' not in campos:
            print("  [ADVERTENCIA] El archivo no tiene la columna 'motivo' (version anterior de verificar_caminos.py): "
                  "se tratan todos los ciclos como sin_destino.")
        for row in reader:
            ts = row['ciclo'].strip()
            if not ts:
                continue
            motivo = (row.get('motivo') or 'sin_destino').strip()
            excluidos.add(ts)
            por_motivo[motivo] += 1
            if motivo in ('parpadeo', 'transitorio'):
                bid = (row.get('bloque_id') or '').strip()
                if bid:
                    por_bloque[int(bid)] += 1
    print("  - Ciclos excluidos: {} ({})".format(
        len(excluidos), ", ".join("{}: {}".format(m, n) for m, n in sorted(por_motivo.items())) or "ninguno"))
    return excluidos, dict(por_bloque)


def cargar_csv_crudo(input_path, dest_hop=None, excluidos=None):
    print("[FASE 1] Cargando CSV crudo: {}".format(input_path))
    por_ciclo = defaultdict(dict)
    
    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        n_lineas = 0
        for row in reader:
            n_lineas += 1
            ts = row["timestamp"].strip()
            try:
                hop = int(row["hop"])
                rtt = float(row["rtt_ms"])
            except (ValueError, TypeError, KeyError):
                continue
            por_ciclo[ts][hop] = rtt
    
    serie = []
    longitudes = []
    cambios_longitud = []
    longitud_anterior = None
    
    n_excluidos = 0
    for ts in sorted(por_ciclo.keys()):
        hops = por_ciclo[ts]
        if not hops:
            continue
        if excluidos and ts in excluidos:
            # Ciclo sin destino alcanzado (verificar_caminos): su ultimo salto no es el destino y su RTT no es valido
            n_excluidos += 1
            continue
        longitud_actual = max(hops.keys())
        
        if longitud_anterior is not None and longitud_actual != longitud_anterior:
            cambios_longitud.append({
                'timestamp': ts,
                'longitud_anterior': longitud_anterior,
                'longitud_nueva': longitud_actual,
                'diferencia': longitud_actual - longitud_anterior
            })
        
        longitud_anterior = longitud_actual
        longitudes.append((ts, longitud_actual))
        
        if dest_hop is not None:
            if dest_hop in hops:
                serie.append((ts, hops[dest_hop], longitud_actual))
        else:
            serie.append((ts, hops[longitud_actual], longitud_actual))
    
    print("  - Líneas procesadas: {}".format(n_lineas))
    if excluidos:
        print("  - Ciclos excluidos quitados de la serie: {} (de {} en el archivo de excluidos)".format(
            n_excluidos, len(excluidos)))
    print("  - Ciclos extraídos: {}".format(len(serie)))
    return serie, longitudes, cambios_longitud


def cargar_bloques_desde_csv(bloques_path):
    """
    Lee directamente el CSV de bloques generado por verificar_caminos.py.
    Estructura esperada: bloque_id,longitud,inicio,fin,duracion_h,ciclos,fuera,tramos_sostenidos,longitud_modal
    """
    print("[FASE 1b] Cargando bloques desde: {}".format(bloques_path))
    df = pd.read_csv(bloques_path)
    
    bloques = []
    for idx, row in df.iterrows():
        bloques.append({
            'bloque_id': int(row['bloque_id']),
            'longitud': int(row['longitud']),
            'inicio': str(row['inicio']),
            'fin': str(row['fin']),
            'ciclos': int(row['ciclos']),
            'tramos_sostenidos': int(row['tramos_sostenidos'])
        })
        
    print("  - Bloques coherentes cargados: {}".format(len(bloques)))
    return bloques


def extraer_sub_serie(serie_completa, bloque, excluidos_en_bloque=0):
    """
    Extrae los ciclos del bloque por rango de tiempo [inicio, fin] (ambos inclusive), tal como los define
    bloques_longitud_*.csv, y se queda solo con los de la longitud del bloque (el target real).

    Antes se tomaba una ventana de 'ciclos' ciclos desde 'inicio'. Como serie_completa incluia ciclos que
    verificar_caminos.py habia excluido (destino no alcanzado) y la columna 'ciclos' no los cuenta, la ventana
    terminaba antes del fin real del bloque. Con los ciclos excluidos ya quitados de serie_completa, la cantidad de
    ciclos dentro de [inicio, fin] debe coincidir con 'ciclos'; si no, se informa.
    'excluidos_en_bloque' son los parpadeos/transitorios que el CSV de excluidos ubica en este bloque: ya se quitaron
    de serie_completa pero siguen contados en 'ciclos', asi que se descuentan al comparar.
    Los timestamps tienen formato fijo AAAA-MM-DD HH:MM:SS, asi que la comparacion de strings respeta el orden.
    """
    inicio_str = bloque['inicio']
    fin_str = bloque['fin']
    longitud = bloque['longitud']
    ciclos_esperados = bloque['ciclos'] - excluidos_en_bloque

    window = [(ts, rtt, lon) for ts, rtt, lon in serie_completa if inicio_str <= ts <= fin_str]

    if not window:
        print("    [ADVERTENCIA] No hay ciclos entre {} y {}".format(inicio_str, fin_str))
        return []
    if len(window) != ciclos_esperados:
        print("    [ADVERTENCIA] El bloque tiene {} ciclos entre {} y {}, pero se esperaban {} ({} del CSV de bloques "
              "menos {} parpadeos/transitorios excluidos). Revisar que --crudo, --bloques-csv y --ciclos-excluidos "
              "sean de la misma corrida de verificar_caminos."
              .format(len(window), inicio_str, fin_str, ciclos_esperados, bloque['ciclos'], excluidos_en_bloque))

    # Filtrar solo las mediciones que coinciden con la longitud del bloque (el target real)
    sub_serie = [(ts, rtt) for ts, rtt, lon in window if lon == longitud]

    return sub_serie


# ============================================================================
# FASE 2: SEGMENTACIÓN TEMPORAL POR HUECOS
# ============================================================================
def fmt_min(x):
    """Minutos sin ceros sobrantes: 60.0 -> '60', 12.5 -> '12.5'."""
    return "{:.2f}".format(x).rstrip('0').rstrip('.')


def parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

def cortar_en_secuencias(serie, gap_minutes):
    if not serie: return []
    secuencias = []
    actual = [serie[0]]
    for i in range(1, len(serie)):
        ts_prev = parse_ts(serie[i - 1][0])
        ts_cur = parse_ts(serie[i][0])
        delta_min = (ts_cur - ts_prev).total_seconds() / 60.0
        if delta_min > gap_minutes:
            secuencias.append(actual)
            actual = [serie[i]]
        else:
            actual.append(serie[i])
    secuencias.append(actual)
    return secuencias


# ============================================================================
# FASE 3: ENTRENAMIENTO DEL HDP-HMM CON BNPY
# ============================================================================
def correr_hdphmm(secuencias, K, nlap, transAlpha, startAlpha, hmmKappa, sF, seed, bnpy_outdir, alg_name):
    print("[FASE 3] Entrenando HDP-HMM con bnpy...")
    print("  - Parámetros: K={}, nlap={}, transAlpha={}, startAlpha={}, hmmKappa={}, sF={}".format(
        K, nlap, transAlpha, startAlpha, hmmKappa, sF))
    
    if not os.environ.get("BNPYOUTDIR"):
        os.environ["BNPYOUTDIR"] = bnpy_outdir
    if not os.path.isdir(os.environ["BNPYOUTDIR"]):
        os.makedirs(os.environ["BNPYOUTDIR"])
    
    try:
        import bnpy
    except ImportError:
        print("  [ERROR] bnpy no está disponible en este intérprete.")
        return None, None, None
    
    X = np.concatenate(
        [np.array([rtt for (_, rtt) in seq], dtype=np.float64).reshape(-1, 1)
         for seq in secuencias], axis=0)
    doc_range = [0]
    for seq in secuencias:
        doc_range.append(doc_range[-1] + len(seq))
    doc_range = np.array(doc_range, dtype=np.int32)
    
    Data = bnpy.data.GroupXData(X=X, doc_range=doc_range)
    Data.name = "ripe_atlas_rtt"
    
    try:
        hmodel, RInfo = bnpy.run(
            Data, "HDPHMM", "Gauss", alg_name,
            jobname="hdphmm-segment", nLap=nlap, nTask=1, nBatch=1,
            K=K, initname="randexamples",
            transAlpha=transAlpha, startAlpha=startAlpha, hmmKappa=hmmKappa,
            sF=sF, ECovMat="eye", printEvery=25, saveEvery=-1, traceEvery=-1,
            seed=seed, doWriteStdOut=False)
    except Exception as e:
        print("  [ERROR] Entrenando HDP-HMM: {}".format(e))
        return None, None, None
    
    LP = hmodel.calc_local_params(Data)
    resp = LP["resp"]
    estados = resp.argmax(axis=1)
    
    estados_por_secuencia = []
    for i in range(len(secuencias)):
        ini, fin = doc_range[i], doc_range[i + 1]
        estados_por_secuencia.append(estados[ini:fin])
    
    n_estados_usados = len(set(int(e) for arr in estados_por_secuencia for e in arr))
    print("  - Estados activos usados: {} (de K={} truncados)".format(n_estados_usados, K))
    
    todos_estados = [int(e) for arr in estados_por_secuencia for e in arr]
    todos_rtts = [rtt for seq in secuencias for (_, rtt) in seq]
    
    for est in sorted(set(todos_estados)):
        mask = [i for i, e in enumerate(todos_estados) if e == est]
        rtt_mean = np.mean([todos_rtts[i] for i in mask])
        rtt_std = np.std([todos_rtts[i] for i in mask])
        count = len(mask)
        pct = 100.0 * count / len(todos_estados)
        print("      Estado {}: {} puntos ({:.1f}%) | RTT={:.2f}±{:.2f}ms".format(
            est, count, pct, rtt_mean, rtt_std))
    
    return hmodel, Data, estados_por_secuencia


# ============================================================================
# FASE 3b: FUSIÓN DE ESTADOS (BHATTACHARYYA + COMPLETE LINKAGE)
# ============================================================================
def _distancia_bhattacharyya(m1, s1, m2, s2):
    s1 = max(s1, 1e-6)
    s2 = max(s2, 1e-6)
    var_sum = (s1 * s1) + (s2 * s2)
    term1 = 0.25 * ((m1 - m2) ** 2) / var_sum
    term2 = 0.5 * np.log(var_sum / (2.0 * s1 * s2))
    return float(term1 + term2)


def fusionar_estados_automatico(estados_por_secuencia, todos_rtts, umbral_db):
    print("\n[FASE 3b] Fusionando estados automáticamente (Bhattacharyya + Complete Linkage)...")
    print("  - Umbral D_B: {:.2f}".format(umbral_db))
    
    flat_states = [int(e) for arr in estados_por_secuencia for e in arr]
    unique_states = sorted(set(flat_states))
    
    stats = {}
    for s in unique_states:
        indices = [i for i, x in enumerate(flat_states) if x == s]
        rtts_s = [todos_rtts[i] for i in indices]
        stats[s] = {'mean': float(np.mean(rtts_s)), 'std': float(np.std(rtts_s)), 'count': len(rtts_s)}
    
    n = len(unique_states)
    db_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = unique_states[i], unique_states[j]
            d = _distancia_bhattacharyya(stats[s1]['mean'], stats[s1]['std'], stats[s2]['mean'], stats[s2]['std'])
            db_matrix[i, j] = d
            db_matrix[j, i] = d
    
    clusters = [[s] for s in unique_states]
    merged = True
    while merged:
        merged = False
        min_dist = float('inf')
        best_i, best_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_db = max(db_matrix[unique_states.index(s_i), unique_states.index(s_j)] 
                             for s_i in clusters[i] for s_j in clusters[j])
                if max_db < min_dist:
                    min_dist = max_db
                    best_i, best_j = i, j
        
        if min_dist <= umbral_db:
            clusters[best_i] = clusters[best_i] + clusters[best_j]
            clusters.pop(best_j)
            merged = True
    
    mapa_fusion = {}
    sorted_clusters = sorted(clusters, key=lambda cluster: min(stats[s]['mean'] for s in cluster))
    
    print("  - Mapa de fusión generado ({} macro-estados):".format(len(sorted_clusters)))
    for macro_id, cluster in enumerate(sorted_clusters):
        all_rtts = [todos_rtts[i] for s in cluster for i, x in enumerate(flat_states) if x == s]
        print("      Macro-Estado {}: estados {} | RTT={:.2f}±{:.2f}ms ({} pts)".format(
            macro_id, sorted(cluster), np.mean(all_rtts), np.std(all_rtts), len(all_rtts)))
        for s in cluster:
            mapa_fusion[s] = macro_id
    
    return [[mapa_fusion[int(e)] for e in seq] for seq in estados_por_secuencia], mapa_fusion


# ============================================================================
# FASE 4: SUAVIZADO POST-HOC
# ============================================================================
def suavizar_estados(estados, min_dwell):
    estados = list(estados)
    n = len(estados)
    if n == 0 or min_dwell <= 1: return estados
    
    for _ in range(20):
        runs = []
        i = 0
        while i < n:
            j = i
            while j < n and estados[j] == estados[i]: j += 1
            runs.append([i, j, estados[i]])
            i = j
        if len(runs) <= 1: break
        
        hubo_cambio = False
        for idx in range(len(runs)):
            ini, fin, val = runs[idx]
            if fin - ini >= min_dwell: continue
            vecino_izq = runs[idx - 1] if idx > 0 else None
            vecino_der = runs[idx + 1] if idx < len(runs) - 1 else None
            if vecino_izq is not None and vecino_der is not None:
                elegido = vecino_izq if (vecino_izq[1] - vecino_izq[0]) >= (vecino_der[1] - vecino_der[0]) else vecino_der
            elif vecino_izq is not None: elegido = vecino_izq
            elif vecino_der is not None: elegido = vecino_der
            else: continue
            
            for k in range(ini, fin): estados[k] = elegido[2]
            hubo_cambio = True
        if not hubo_cambio: break
    return estados


# ============================================================================
# FASE 5: DETECCIÓN Y CARACTERIZACIÓN
# ============================================================================
def detectar_change_points(secuencias, estados_por_secuencia):
    change_points = []
    for seq, estados in zip(secuencias, estados_por_secuencia):
        for i in range(1, len(estados)):
            if estados[i] != estados[i - 1]:
                change_points.append({
                    'timestamp': seq[i][0],
                    'estado_anterior': int(estados[i - 1]),
                    'estado_nuevo': int(estados[i])
                })
    return change_points


def caracterizar_segmentos(serie_completa, secuencias, estados_por_secuencia, estados_suaves_por_secuencia,
                           intervalo_min):
    segmentos = []
    for seq_id, (seq, estados_suaves) in enumerate(zip(secuencias, estados_suaves_por_secuencia)):
        start_idx = 0
        estado_actual = estados_suaves[0]
        
        for i in range(1, len(estados_suaves)):
            if estados_suaves[i] != estado_actual:
                segmentos.append(_procesar_segmento(serie_completa, seq, start_idx, i - 1, seq_id, int(estado_actual),
                                                    intervalo_min))
                start_idx = i
                estado_actual = estados_suaves[i]
        
        segmentos.append(_procesar_segmento(serie_completa, seq, start_idx, len(estados_suaves) - 1, seq_id,
                                            int(estado_actual), intervalo_min))
    
    return segmentos

def _procesar_segmento(serie_completa, seq, start_idx, end_idx, seq_id, estado, intervalo_min):
    rtts_seg = [serie_completa[idx][1] for idx in range(start_idx, end_idx + 1)]
    return {
        'seq_id': seq_id, 'estado': estado,
        'start_ts': seq[start_idx][0], 'end_ts': seq[end_idx][0],
        'n_ciclos': end_idx - start_idx + 1,
        'duracion_minutos': (end_idx - start_idx + 1) * intervalo_min,
        'media': float(np.mean(rtts_seg)), 'mediana': float(np.median(rtts_seg)),
        'std': float(np.std(rtts_seg)), 'min': float(np.min(rtts_seg)), 'max': float(np.max(rtts_seg))
    }

def calcular_calidad_segmentacion(serie_completa, segmentos):
    rtts = [rtt for _, rtt, _ in serie_completa]
    media_global = np.mean(rtts)
    n_total = len(rtts)
    
    var_intra = sum((seg['n_ciclos'] / n_total) * (seg['std'] ** 2) for seg in segmentos)
    var_inter = sum((seg['n_ciclos'] / n_total) * ((seg['media'] - media_global) ** 2) for seg in segmentos)
    
    return {
        'varianza_intra': var_intra, 'varianza_inter': var_inter,
        'ratio_calidad': var_inter / max(var_intra, 1e-10),
        'duracion_media_segmentos': np.mean([s['duracion_minutos'] for s in segmentos]),
        'n_segmentos': len(segmentos)
    }

def clasificar_tipo_degradacion(segmento, rtt_referencia):
    """
    Clasifica un segmento contra el RTT de referencia de su ruta (no contra un valor global).
    Usa la MEDIANA del segmento (robusta a spikes) y conserva el SIGNO de la desviacion:
    por encima de la referencia -> 'degradacion_*'; por debajo -> 'mejora_*'.
    """
    desviacion = segmento['mediana'] - rtt_referencia
    magnitud = abs(desviacion)
    duracion = segmento['duracion_minutos']
    prefijo = 'degradacion' if desviacion > 0 else 'mejora'
    
    if duracion > 60 and magnitud > UMBRAL_SEVERA_MS: return prefijo + '_severa_sostenida'
    elif duracion > 30 and magnitud > UMBRAL_MODERADA_MS: return prefijo + '_moderada_sostenida'
    elif duracion < 10 and magnitud > 10: return 'pico_transitorio'
    elif magnitud < 1: return 'regimen_base'
    else: return 'variacion_normal'


def calcular_referencias_por_ruta(rtts_por_longitud):
    """RTT de referencia (P25) de cada ruta, usando los ciclos de todos los bloques con esa longitud."""
    return dict((lon, float(np.percentile(rtts, 25))) for lon, rtts in rtts_por_longitud.items() if rtts)


def clasificar_cambio_rtt(delta):
    """Clasifica con signo el salto de RTT entre dos bloques consecutivos."""
    magnitud = abs(delta)
    if magnitud > UMBRAL_SEVERA_MS: nivel = 'severa'
    elif magnitud > UMBRAL_MODERADA_MS: nivel = 'moderada'
    else: return 'sin_cambio_relevante'
    return ('degradacion_' if delta > 0 else 'mejora_') + nivel


def detectar_cambios_de_ruta(info_bloques, ventana):
    """
    Un cambio de ruta es la frontera entre dos bloques procesados y contiguos. Se mide el RTT justo antes y justo
    despues: mediana de los ultimos / primeros 'ventana' ciclos de cada bloque (o menos, si un bloque es mas corto).
    Se usan los bordes y no la mediana de todo el bloque, porque un bloque largo puede haber cambiado de regimen
    antes de terminar (p. ej. B1 terminaba en ~34 ms aunque su mediana global sea ~29 ms).
    """
    cambios = []
    for ant, sig in zip(info_bloques[:-1], info_bloques[1:]):
        if sig['idx'] != ant['idx'] + 1:
            continue  # hay un bloque omitido entre ambos: no son contiguos
        w = min(ventana, ant['n'], sig['n'])
        antes = float(np.median(ant['rtts_fin'][-w:]))
        despues = float(np.median(sig['rtts_ini'][:w]))
        delta = despues - antes
        cambios.append({
            'timestamp': sig['start_ts'], 'bloque_anterior': ant['bloque_id'], 'bloque_siguiente': sig['bloque_id'],
            'longitud_anterior': ant['longitud'], 'longitud_siguiente': sig['longitud'],
            'rtt_antes': antes, 'rtt_despues': despues, 'delta_rtt': delta,
            'tipo_cambio': clasificar_cambio_rtt(delta), 'ventana': w})
    return cambios


# ============================================================================
# FASE 5b: AGRUPAMIENTO DE SEGMENTOS CONSECUTIVOS
# ============================================================================
def agrupar_segmentos_consecutivos(segmentos):
    """
    Agrupa segmentos consecutivos que comparten el mismo 'tipo_degradacion',
    manteniendo la continuidad temporal y calculando estadísticas agregadas.
    
    Fórmulas aplicadas:
    - Media: Media ponderada por n_ciclos
    - Std: Desviación estándar agrupada (Pooled Std Dev)
      σ_pooled = sqrt( (Σ(n_i * (σ_i² + μ_i²)) / N) - μ_global² )
    - Mediana: Proxy = mediana del segmento con mayor n_ciclos
    - Min/Max: Valores extremos absolutos del grupo
    """
    if not segmentos:
        return []
    
    grupos_raw = []
    current_group = []
    
    for seg in segmentos:
        # Solo se agrupan segmentos del mismo tipo Y de la misma ruta (longitud): la referencia es por ruta
        if (not current_group or (current_group[-1]['tipo_degradacion'] == seg['tipo_degradacion']
                                  and current_group[-1].get('longitud') == seg.get('longitud'))):
            current_group.append(seg)
        else:
            grupos_raw.append(current_group)
            current_group = [seg]
    if current_group:
        grupos_raw.append(current_group)
        
    grupos_finales = []
    for idx, group in enumerate(grupos_raw):
        total_n = sum(s['n_ciclos'] for s in group)
        total_dur = sum(s['duracion_minutos'] for s in group)
        
        # Media ponderada por cantidad de ciclos
        weighted_mean = sum(s['media'] * s['n_ciclos'] for s in group) / total_n
        
        # Desviación estándar agrupada (Pooled Std Dev)
        # Fórmula: sqrt( (Σ(n_i * (σ_i² + μ_i²)) / N) - μ_global² )
        sum_sq = sum(s['n_ciclos'] * (s['std']**2 + s['media']**2) for s in group)
        pooled_var = (sum_sq / total_n) - (weighted_mean**2)
        pooled_std = np.sqrt(max(0, pooled_var)) # max(0) para evitar errores de precisión flotante
        
        # Min y Max globales del grupo
        min_val = min(s['min'] for s in group)
        max_val = max(s['max'] for s in group)
        
        # Mediana proxy: usamos la mediana del segmento con mayor peso (más ciclos)
        median_proxy = max(group, key=lambda s: s['n_ciclos'])['mediana']
        
        grupos_finales.append({
            'grupo_id': idx + 1,
            'start_ts': group[0]['start_ts'],
            'end_ts': group[-1]['end_ts'],
            'n_ciclos': total_n,
            'duracion_minutos': total_dur,
            'media': weighted_mean,
            'mediana': median_proxy,
            'std': pooled_std,
            'min': min_val,
            'max': max_val,
            'tipo_degradacion': group[0]['tipo_degradacion'],
            'n_segmentos_originales': len(group)
        })
        
    return grupos_finales


# ============================================================================
# FASE 6: EXPORTACIÓN
# ============================================================================
def exportar_resultados(serie_completa, secuencias, estados_por_secuencia, estados_suaves_por_secuencia,
                        change_points, segmentos, metricas_calidad, output_dir, measurement_id, probe_id,
                        longitudes=None, cambios_longitud=None, tipo_fusion=None, mapa_fusion=None,
                        cambios_ruta=None, referencias=None):
    print("[FASE 6] Exportando resultados a: {}".format(output_dir))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    ids_suffix = "_measurement{}_probe{}".format(measurement_id, probe_id) if measurement_id and probe_id else ""
    fusion_suffix = "_macro" if tipo_fusion == 'manual' else ("_auto" if tipo_fusion == 'auto' else "")
    
    # 1. Estados
    with open(os.path.join(output_dir, 'estados_hdphmm{}{}.csv'.format(ids_suffix, fusion_suffix)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'rtt_ms', 'seq_id', 'state_raw', 'state'])
        for seq_id, (seq, crudos, suaves) in enumerate(zip(secuencias, estados_por_secuencia, estados_suaves_por_secuencia)):
            for (ts, rtt), e_crudo, e_suave in zip(seq, crudos, suaves):
                writer.writerow([ts, "{:.3f}".format(rtt), seq_id, int(e_crudo), int(e_suave)])
    
    # 2. Segmentos caracterizados
    path_segmentos = os.path.join(output_dir, 'segmentos_caracterizados{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_segmentos, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_id', 'estado', 'start_ts', 'end_ts', 'n_ciclos', 'duracion_minutos',
                        'media', 'mediana', 'std', 'min', 'max', 'tipo_degradacion',
                        'bloque_id', 'longitud', 'rtt_referencia', 'desviacion_ms'])
        for seg in segmentos:
            writer.writerow([seg['seq_id'], seg['estado'], seg['start_ts'], seg['end_ts'],
                           seg['n_ciclos'], fmt_min(seg['duracion_minutos']), "{:.3f}".format(seg['media']),
                           "{:.3f}".format(seg['mediana']), "{:.3f}".format(seg['std']),
                           "{:.3f}".format(seg['min']), "{:.3f}".format(seg['max']), seg['tipo_degradacion'],
                           seg.get('bloque_id', ''), seg.get('longitud', ''),
                           "{:.3f}".format(seg['rtt_referencia']), "{:.3f}".format(seg['desviacion_ms'])])
    print("  - Segmentos caracterizados: {}".format(path_segmentos))
    
    # 2b. Segmentos caracterizados AGRUPADOS
    segmentos_agrupados = agrupar_segmentos_consecutivos(segmentos)
    path_agrupados = os.path.join(output_dir, 'segmentos_caracterizados_agrupados{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_agrupados, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['grupo_id', 'start_ts', 'end_ts', 'n_ciclos', 'duracion_minutos',
                        'media', 'mediana', 'std', 'min', 'max', 'tipo_degradacion', 'n_segmentos_originales'])
        for grp in segmentos_agrupados:
            writer.writerow([grp['grupo_id'], grp['start_ts'], grp['end_ts'],
                           grp['n_ciclos'], fmt_min(grp['duracion_minutos']), 
                           "{:.3f}".format(grp['media']), "{:.3f}".format(grp['mediana']),
                           "{:.3f}".format(grp['std']), "{:.3f}".format(grp['min']),
                           "{:.3f}".format(grp['max']), grp['tipo_degradacion'], 
                           grp['n_segmentos_originales']])
    print("  - Segmentos agrupados: {} ({} grupos)".format(path_agrupados, len(segmentos_agrupados)))
    
    # 2c. Cambios de ruta (frontera entre bloques contiguos, con signo)
    cambios_ruta = cambios_ruta or []
    path_cr = os.path.join(output_dir, 'cambios_de_ruta{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_cr, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'bloque_anterior', 'bloque_siguiente', 'longitud_anterior', 'longitud_siguiente',
                        'delta_saltos', 'rtt_antes_ms', 'rtt_despues_ms', 'delta_rtt_ms', 'tipo_cambio', 'ciclos_ventana'])
        for c in cambios_ruta:
            writer.writerow([c['timestamp'], c['bloque_anterior'], c['bloque_siguiente'], c['longitud_anterior'],
                           c['longitud_siguiente'], c['longitud_siguiente'] - c['longitud_anterior'],
                           "{:.3f}".format(c['rtt_antes']), "{:.3f}".format(c['rtt_despues']),
                           "{:.3f}".format(c['delta_rtt']), c['tipo_cambio'], c['ventana']])
    print("  - Cambios de ruta: {} ({} cambios)".format(path_cr, len(cambios_ruta)))
    
    # 3. Puntos de cambio
    with open(os.path.join(output_dir, 'change_points{}{}.csv'.format(ids_suffix, fusion_suffix)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estado_anterior', 'estado_nuevo'])
        for cp in change_points:
            writer.writerow([cp['timestamp'], cp['estado_anterior'], cp['estado_nuevo']])
    
    # 4. Resumen
    with open(os.path.join(output_dir, 'resumen_ejecucion{}{}.txt'.format(ids_suffix, fusion_suffix)), 'w') as f:
        f.write("=" * 70 + "\nRESUMEN DE EJECUCIÓN HDP-HMM (ANÁLISIS POR BLOQUES)\n" + "=" * 70 + "\n\n")
        if measurement_id and probe_id:
            f.write("Measurement ID: {}\nProbe ID: {}\n\n".format(measurement_id, probe_id))
        f.write("Fecha: {}\nDataset: {} puntos temporales\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(serie_completa)))
        f.write("Rango temporal: {} a {}\nTipo de fusión: {}\n\n".format(serie_completa[0][0], serie_completa[-1][0], tipo_fusion or 'Ninguna'))
        
        rtts = [rtt for _, rtt, _ in serie_completa]
        f.write("--- ESTADÍSTICAS DE RTT ---\nMin: {:.2f} ms\nMax: {:.2f} ms\nMedia: {:.2f} ms\nDesv. Est.: {:.2f} ms\n\n".format(
            min(rtts), max(rtts), np.mean(rtts), np.std(rtts)))
        
        f.write("--- MODELO HDP-HMM ---\nEstados activos: {}\nChange-points detectados: {}\n\n".format(
            len(set(int(e) for arr in estados_por_secuencia for e in arr)), len(change_points)))
        
        f.write("--- CARACTERIZACIÓN DE SEGMENTOS ---\n")
        f.write("Total de segmentos originales: {}\n".format(len(segmentos)))
        f.write("Total de grupos agrupados: {}\n".format(len(segmentos_agrupados)))
        f.write("Reducción: {:.1f}%\n\n".format(100.0 * (1 - len(segmentos_agrupados) / max(len(segmentos), 1))))
        
        tipo_counts = defaultdict(int)
        for seg in segmentos: tipo_counts[seg['tipo_degradacion']] += 1
        f.write("Distribución de segmentos originales:\n")
        for tipo, count in sorted(tipo_counts.items(), key=lambda x: -x[1]):
            f.write("  {}: {} segmentos\n".format(tipo, count))
        
        tipo_counts_agrup = defaultdict(int)
        for grp in segmentos_agrupados: tipo_counts_agrup[grp['tipo_degradacion']] += 1
        f.write("\nDistribución de grupos agrupados:\n")
        for tipo, count in sorted(tipo_counts_agrup.items(), key=lambda x: -x[1]):
            f.write("  {}: {} grupos\n".format(tipo, count))
        
        f.write("\n--- MÉTRICAS DE CALIDAD ---\nVarianza intra-segmento: {:.4f}\nVarianza inter-segmento: {:.4f}\n".format(
            metricas_calidad['varianza_intra'], metricas_calidad['varianza_inter']))
        f.write("Ratio de calidad: {:.4f}\nDuración media de segmentos: {:.2f} minutos\n\n".format(
            metricas_calidad['ratio_calidad'], metricas_calidad['duracion_media_segmentos']))
        
        f.write("--- RTT DE REFERENCIA POR RUTA (P25 de los bloques con la misma longitud) ---\n")
        for lon in sorted((referencias or {}).keys()):
            f.write("  {} saltos: {:.2f} ms\n".format(lon, referencias[lon]))
        f.write("\n--- CAMBIOS DE RUTA ---\nCambios de ruta clasificados: {}\n".format(len(cambios_ruta)))
        for c in cambios_ruta:
            f.write("  B{}->B{} ({}->{} saltos): {:.2f} -> {:.2f} ms (delta {:+.2f}) {}\n".format(
                c['bloque_anterior'], c['bloque_siguiente'], c['longitud_anterior'], c['longitud_siguiente'],
                c['rtt_antes'], c['rtt_despues'], c['delta_rtt'], c['tipo_cambio']))
        f.write("\n--- PUNTOS DE CAMBIO ---\nPuntos de cambio detectados: {}\n".format(len(change_points)))
        f.write("\n" + "=" * 70 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="HDP-HMM con bnpy para segmentación de latencia en RIPE Atlas")
    parser.add_argument('--crudo', required=True, help='CSV crudo de traceroute')
    parser.add_argument('--bloques-csv', required=True, help='CSV de bloques generado por verificar_caminos.py')
    parser.add_argument('--ciclos-excluidos', required=True,
                        help='CSV ciclos_excluidos_measurement<M>_probe<P>.csv generado por verificar_caminos.py')
    parser.add_argument('--interval', type=float, required=True,
                        help='Intervalo de medicion de la sonda en segundos (p. ej. 300 o 900); '
                             'duracion_minutos = n_ciclos * interval / 60')
    parser.add_argument('--output-dir', default='resultados_hdphmm', help='Directorio de salida')
    parser.add_argument('--dest-hop', type=int, default=None, help='Hop fijo de destino')
    parser.add_argument('--gap-minutes', type=float, default=90.0, help='Hueco temporal (min)')
    parser.add_argument('--K', type=int, default=15, help='Truncamiento de estados')
    parser.add_argument('--nlap', type=int, default=100, help='Pasadas de entrenamiento')
    parser.add_argument('--trans-alpha', type=float, default=0.5, help='Concentración transiciones')
    parser.add_argument('--start-alpha', type=float, default=10.0, help='Concentración inicial')
    parser.add_argument('--hmm-kappa', type=float, default=1000.0, help='Sticky-ness')
    parser.add_argument('--sF', type=float, default=1.0, help='Escala covarianza prior')
    parser.add_argument('--min-dwell', type=int, default=3, help='Dwell mínimo (suavizado)')
    parser.add_argument('--seed', type=int, default=0, help='Semilla aleatoria')
    parser.add_argument('--bnpy-outdir', default='./bnpy-output', help='Carpeta bnpy')
    parser.add_argument('--alg-name', default='moVB', help='Algoritmo bnpy')
    
    parser.add_argument('--macro-estados', action='store_true', help='Fusionar estados manualmente')
    parser.add_argument('--mapa-macro', type=str, default=None, help='Archivo CSV con mapa personalizado')
    parser.add_argument('--fusion-auto', action='store_true', help='Fusionar estados automáticamente por Bhattacharyya')
    parser.add_argument('--percentil-outliers', type=float, default=0.0,
                        help='Descarta por bloque los ciclos con RTT por encima de este percentil (p. ej. 99). '
                             'Default 0 = sin filtro (el filtro borra ciclos del propio evento degradado)')
    parser.add_argument('--ventana-borde', type=int, default=10,
                        help='Ciclos a cada lado de la frontera entre bloques para medir el cambio de RTT de un '
                             'cambio de ruta (mediana de los ultimos/primeros N ciclos; default 10)')
    parser.add_argument('--umbral-db', type=float, default=0.5, help='Distancia de Bhattacharyya máxima (default: 0.5)')
    
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval debe ser mayor que 0 (segundos entre ciclos)")
    if args.ventana_borde < 1:
        parser.error("--ventana-borde debe ser al menos 1")
    intervalo_min = args.interval / 60.0
    
    print("=" * 70)
    print("HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas (v5 - Bloques CSV)")
    print("=" * 70)
    print("Python: {}".format(sys.version))
    print("NumPy: {}".format(np.__version__))
    print("Intervalo de medición: {:g} s ({:g} min por ciclo)".format(args.interval, intervalo_min))
    print("=" * 70)
    
    measurement_id, probe_id = extraer_ids_de_archivo(args.crudo)
    if measurement_id and probe_id:
        print("Measurement ID: {}\nProbe ID: {}".format(measurement_id, probe_id))
    print("=" * 70)
    
    excluidos, excluidos_por_bloque = cargar_ciclos_excluidos(args.ciclos_excluidos)
    serie_completa, longitudes, cambios_longitud = cargar_csv_crudo(args.crudo, dest_hop=args.dest_hop,
                                                                    excluidos=excluidos)
    bloques = cargar_bloques_desde_csv(args.bloques_csv)
    
    if not serie_completa or not bloques:
        print("[ERROR] No se pudieron cargar los datos o bloques."); sys.exit(1)
    
    print("\n[FASE 2] Procesando {} bloques coherentes...".format(len(bloques)))
    
    todos_segmentos = []
    todos_change_points = []
    estados_por_secuencia_global = []
    estados_suaves_por_secuencia_global = []
    secuencias_global = []
    rtts_por_longitud = defaultdict(list)   # RTT de todos los bloques de cada ruta (longitud), para la referencia
    info_bloques = []                        # bordes de cada bloque procesado, para clasificar cambios de ruta
    
    for idx, bloque in enumerate(bloques):
        print("\n  [BLOQUE {}/{}] ID: {}, Longitud: {} hops, Ciclos totales: {}".format(
            idx + 1, len(bloques), bloque['bloque_id'], bloque['longitud'], bloque['ciclos']))
        
        sub_serie = extraer_sub_serie(serie_completa, bloque,
                                      excluidos_por_bloque.get(bloque['bloque_id'], 0))
        if len(sub_serie) < 10:
            print("    → Bloque demasiado pequeño, omitido.")
            continue
        
        # >>> PRE-FILTRADO OPCIONAL DE OUTLIERS POR PERCENTIL (--percentil-outliers, default 0 = desactivado) <<<
        # Cortar el percentil mas alto de cada bloque borra los ciclos de RTT mas alto, que en una degradacion
        # sostenida son los del propio evento. Por eso esta apagado por defecto.
        if args.percentil_outliers > 0:
            rtts_bloque = [rtt for _, rtt in sub_serie]
            umbral_p = np.percentile(rtts_bloque, args.percentil_outliers)
            sub_serie_filtrada = [(ts, rtt) for ts, rtt in sub_serie if rtt <= umbral_p]
            n_filtrados = len(sub_serie) - len(sub_serie_filtrada)
            print("    → Outliers filtrados (RTT > P{:g}={:.2f}ms): {}".format(
                args.percentil_outliers, umbral_p, n_filtrados))
            sub_serie = sub_serie_filtrada

            if len(sub_serie) < 10:
                print("    → Bloque demasiado pequeño tras filtrar outliers, omitido.")
                continue
        else:
            print("    → Sin filtro de outliers por percentil (RTT max del bloque: {:.2f}ms)".format(
                max(rtt for _, rtt in sub_serie)))

        print("    → Ciclos extraídos para análisis: {}".format(len(sub_serie)))
        
        rtts_sub = [rtt for _, rtt in sub_serie]
        rtts_por_longitud[bloque['longitud']].extend(rtts_sub)
        info_bloques.append({'idx': idx, 'bloque_id': bloque['bloque_id'], 'longitud': bloque['longitud'],
                             'start_ts': sub_serie[0][0], 'n': len(rtts_sub),
                             'rtts_ini': rtts_sub[:args.ventana_borde], 'rtts_fin': rtts_sub[-args.ventana_borde:]})
        
        # Cortar en secuencias por huecos temporales dentro del bloque
        secuencias = cortar_en_secuencias(sub_serie, args.gap_minutes)
        secuencias = [s for s in secuencias if len(s) >= 2]
        
        if not secuencias:
            print("    → No quedaron secuencias utilizables.")
            continue
            
        # Entrenar HDP-HMM
        hmodel, Data, estados_por_secuencia = correr_hdphmm(
            secuencias, K=args.K, nlap=args.nlap, transAlpha=args.trans_alpha,
            startAlpha=args.start_alpha, hmmKappa=args.hmm_kappa, sF=args.sF,
            seed=args.seed, bnpy_outdir=args.bnpy_outdir, alg_name=args.alg_name)
        
        if estados_por_secuencia is None:
            continue
            
        # Fusión de estados
        estados_trabajo = estados_por_secuencia
        tipo_fusion = None
        mapa_fusion = None
        
        if args.fusion_auto:
            todos_rtts = [rtt for seq in secuencias for (_, rtt) in seq]
            estados_trabajo, mapa_fusion = fusionar_estados_automatico(estados_por_secuencia, todos_rtts, args.umbral_db)
            tipo_fusion = 'auto'
        elif args.macro_estados:
            mapa_fusion = cargar_mapa_macro_estados(args.mapa_macro)
            estados_trabajo = [[mapa_fusion.get(int(e), int(e)) for e in seq] for seq in estados_por_secuencia]
            tipo_fusion = 'manual'
        
        # Suavizado
        estados_suaves = [suavizar_estados(arr, args.min_dwell) for arr in estados_trabajo]
        
        # Caracterización
        segmentos = caracterizar_segmentos(sub_serie, secuencias, estados_trabajo, estados_suaves, intervalo_min)
        for seg in segmentos:
            seg['bloque_id'] = bloque['bloque_id']
            seg['longitud'] = bloque['longitud']
        todos_segmentos.extend(segmentos)
        
        # Change-points
        cps = detectar_change_points(secuencias, estados_suaves)
        todos_change_points.extend(cps)
        
        # Acumular para exportación global
        estados_por_secuencia_global.extend(estados_por_secuencia)
        estados_suaves_por_secuencia_global.extend(estados_suaves)
        secuencias_global.extend(secuencias)
    
    # RTT de referencia POR RUTA y clasificacion con signo
    rtts = [rtt for _, rtt, _ in serie_completa]
    referencias = calcular_referencias_por_ruta(rtts_por_longitud)
    print("\n  - RTT de referencia por ruta (P25 de los ciclos de los bloques con la misma longitud):")
    for lon in sorted(referencias):
        print("      {} saltos: {:.2f} ms ({} ciclos)".format(lon, referencias[lon], len(rtts_por_longitud[lon])))
    print("  - (Informativo) P25 global de la serie, que ya no se usa para clasificar: {:.2f}ms".format(
        np.percentile(rtts, 25)))
    
    for seg in todos_segmentos:
        ref = referencias[seg['longitud']]
        seg['rtt_referencia'] = ref
        seg['desviacion_ms'] = seg['mediana'] - ref
        seg['tipo_degradacion'] = clasificar_tipo_degradacion(seg, ref)
    
    cambios_ruta = detectar_cambios_de_ruta(info_bloques, args.ventana_borde)
    print("  - Cambios de ruta clasificados: {}".format(len(cambios_ruta)))
    for c in cambios_ruta:
        print("      B{}->B{} ({}->{} saltos): {:.2f} -> {:.2f} ms (delta {:+.2f}) {}".format(
            c['bloque_anterior'], c['bloque_siguiente'], c['longitud_anterior'], c['longitud_siguiente'],
            c['rtt_antes'], c['rtt_despues'], c['delta_rtt'], c['tipo_cambio']))
        
    metricas_calidad = calcular_calidad_segmentacion(serie_completa, todos_segmentos)
    print("  - Ratio de calidad global: {:.2f}".format(metricas_calidad['ratio_calidad']))
    
    print("  - Puntos de cambio detectados: {}".format(len(todos_change_points)))
    
    # Exportar
    exportar_resultados(
        serie_completa, secuencias_global, estados_por_secuencia_global, estados_suaves_por_secuencia_global,
        todos_change_points, todos_segmentos, metricas_calidad, args.output_dir,
        measurement_id, probe_id, longitudes=longitudes, cambios_longitud=cambios_longitud,
        tipo_fusion=tipo_fusion, mapa_fusion=mapa_fusion, cambios_ruta=cambios_ruta, referencias=referencias)
    
    print("\n" + "=" * 70)
    print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("Resultados disponibles en: {}".format(args.output_dir))

if __name__ == '__main__':
    main()
