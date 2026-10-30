# -*- coding: utf-8 -*-
"""
=============================================================================
HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas (v5)
+ Análisis por BLOQUES (Lectura directa de CSV de bloques)
+ Caracterización de Segmentos y Detección de Cambios Topológicos
+ Fusión de Estados (Automática por Bhattacharyya)
=============================================================================
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
TOLERANCIA_MINUTOS = 15

MAPA_MACRO_ESTADOS_DEFAULT = {
    10: 0, 13: 0, 14: 0,  # Macro 0: Base Óptimo
    0: 1, 1: 1,            # Macro 1: Degradación Leve
    2: 2,                  # Macro 2: Congestión
    5: 3,                  # Macro 3: Outlier
}


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
def cargar_csv_crudo(input_path, dest_hop=None):
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
    
    for ts in sorted(por_ciclo.keys()):
        hops = por_ciclo[ts]
        if not hops:
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


def extraer_sub_serie(serie_completa, bloque):
    """
    Extrae los ciclos del bloque buscando su timestamp de inicio y tomando 
    una ventana del tamaño de 'ciclos', filtrando solo la longitud del bloque.
    """
    inicio_str = bloque['inicio']
    longitud = bloque['longitud']
    ciclos_esperados = bloque['ciclos']
    
    start_idx = -1
    for i, (ts, rtt, lon) in enumerate(serie_completa):
        if ts == inicio_str:
            start_idx = i
            break
            
    if start_idx == -1:
        print("    [ADVERTENCIA] No se encontró el inicio del bloque: {}".format(inicio_str))
        return []
        
    end_idx = min(start_idx + ciclos_esperados, len(serie_completa))
    window = serie_completa[start_idx:end_idx]
    
    # Filtrar solo las mediciones que coinciden con la longitud del bloque (el target real)
    sub_serie = [(ts, rtt) for ts, rtt, lon in window if lon == longitud]
    
    return sub_serie


def cargar_categorias(path_csv):
    print("[FASE 1c] Cargando categorías Path Analysis (solo cambios topológicos): {}".format(path_csv))
    dict_categorias = defaultdict(lambda: {'categorias': set(), 'conteo': 0})
    
    with open(path_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row['timestamp'].strip()
            cat = row['categoria'].strip()
            if cat in ['cambio_ip_asn', 'cambio_longitud']:
                dict_categorias[ts]['categorias'].add(cat)
                dict_categorias[ts]['conteo'] += 1
    
    print("  - Eventos topológicos: {}".format(len(dict_categorias)))
    return dict(dict_categorias)


# ============================================================================
# FASE 2: SEGMENTACIÓN TEMPORAL POR HUECOS
# ============================================================================
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


def caracterizar_segmentos(serie_completa, secuencias, estados_por_secuencia, estados_suaves_por_secuencia):
    segmentos = []
    for seq_id, (seq, estados_suaves) in enumerate(zip(secuencias, estados_suaves_por_secuencia)):
        start_idx = 0
        estado_actual = estados_suaves[0]
        
        for i in range(1, len(estados_suaves)):
            if estados_suaves[i] != estado_actual:
                segmentos.append(_procesar_segmento(serie_completa, seq, start_idx, i - 1, seq_id, int(estado_actual)))
                start_idx = i
                estado_actual = estados_suaves[i]
        
        segmentos.append(_procesar_segmento(serie_completa, seq, start_idx, len(estados_suaves) - 1, seq_id, int(estado_actual)))
    
    return segmentos

def _procesar_segmento(serie_completa, seq, start_idx, end_idx, seq_id, estado):
    rtts_seg = [serie_completa[idx][1] for idx in range(start_idx, end_idx + 1)]
    return {
        'seq_id': seq_id, 'estado': estado,
        'start_ts': seq[start_idx][0], 'end_ts': seq[end_idx][0],
        'n_ciclos': end_idx - start_idx + 1,
        'duracion_minutos': (end_idx - start_idx + 1) * 10,
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

def clasificar_tipo_degradacion(segmento, rtt_normal):
    desviacion = segmento['media'] - rtt_normal
    duracion = segmento['duracion_minutos']
    
    if duracion > 60 and abs(desviacion) > 5: return 'degradacion_severa_sostenida'
    elif duracion > 30 and abs(desviacion) > 2: return 'degradacion_moderada_sostenida'
    elif duracion < 10 and abs(desviacion) > 10: return 'pico_transitorio'
    elif abs(desviacion) < 1: return 'regimen_base'
    else: return 'variacion_normal'


def detectar_cambios_topologicos(change_points, dict_categorias):
    cp_dt = {cp['timestamp']: datetime.strptime(cp['timestamp'], "%Y-%m-%d %H:%M:%S") for cp in change_points}
    cat_dt = {ts: datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in dict_categorias.keys()}
    
    n_coincidencias = 0
    for cp in change_points:
        ts_dt = cp_dt.get(cp['timestamp'])
        if not ts_dt: continue
        
        for ts_str, dt_cat in cat_dt.items():
            if abs((ts_dt - dt_cat).total_seconds()) / 60.0 <= TOLERANCIA_MINUTOS:
                cp['evento_topologico'] = True
                cp['categorias_topologicas'] = dict_categorias[ts_str]['categorias']
                n_coincidencias += 1
                break
        else:
            cp['evento_topologico'] = False
            cp['categorias_topologicas'] = set()
            
    return change_points, n_coincidencias


# ============================================================================
# FASE 6: EXPORTACIÓN
# ============================================================================
def exportar_resultados(serie_completa, secuencias, estados_por_secuencia, estados_suaves_por_secuencia,
                        change_points, segmentos, metricas_calidad, output_dir, measurement_id, probe_id,
                        longitudes=None, cambios_longitud=None, tipo_fusion=None, mapa_fusion=None):
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
    with open(os.path.join(output_dir, 'segmentos_caracterizados{}{}.csv'.format(ids_suffix, fusion_suffix)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_id', 'estado', 'start_ts', 'end_ts', 'n_ciclos', 'duracion_minutos',
                        'media', 'mediana', 'std', 'min', 'max', 'tipo_degradacion'])
        for seg in segmentos:
            writer.writerow([seg['seq_id'], seg['estado'], seg['start_ts'], seg['end_ts'],
                           seg['n_ciclos'], seg['duracion_minutos'], "{:.3f}".format(seg['media']),
                           "{:.3f}".format(seg['mediana']), "{:.3f}".format(seg['std']),
                           "{:.3f}".format(seg['min']), "{:.3f}".format(seg['max']), seg['tipo_degradacion']])
    
    # 3. Change-points topológicos
    with open(os.path.join(output_dir, 'change_points_topologicos{}{}.csv'.format(ids_suffix, fusion_suffix)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estado_anterior', 'estado_nuevo', 'evento_topologico', 'categorias_topologicas'])
        for cp in change_points:
            cats_str = "|".join(cp['categorias_topologicas']) if cp['categorias_topologicas'] else 'NINGUNA'
            writer.writerow([cp['timestamp'], cp['estado_anterior'], cp['estado_nuevo'], cp['evento_topologico'], cats_str])
    
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
        
        f.write("--- CARACTERIZACIÓN DE SEGMENTOS ---\nTotal de segmentos: {}\n".format(len(segmentos)))
        tipo_counts = defaultdict(int)
        for seg in segmentos: tipo_counts[seg['tipo_degradacion']] += 1
        for tipo, count in sorted(tipo_counts.items(), key=lambda x: -x[1]):
            f.write("  {}: {} segmentos\n".format(tipo, count))
        
        f.write("\n--- MÉTRICAS DE CALIDAD ---\nVarianza intra-segmento: {:.4f}\nVarianza inter-segmento: {:.4f}\n".format(
            metricas_calidad['varianza_intra'], metricas_calidad['varianza_inter']))
        f.write("Ratio de calidad: {:.4f}\nDuración media de segmentos: {:.2f} minutos\n\n".format(
            metricas_calidad['ratio_calidad'], metricas_calidad['duracion_media_segmentos']))
        
        n_topo = sum(1 for cp in change_points if cp['evento_topologico'])
        f.write("--- CAMBIOS TOPOLÓGICOS ---\nChange-points con evento topológico: {}/{} ({:.1f}%)\n".format(
            n_topo, len(change_points), 100.0 * n_topo / max(len(change_points), 1)))
        f.write("\n" + "=" * 70 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="HDP-HMM con bnpy para segmentación de latencia en RIPE Atlas")
    parser.add_argument('--crudo', required=True, help='CSV crudo de traceroute')
    parser.add_argument('--categorias', required=True, help='CSV de categorías Path Analysis')
    parser.add_argument('--bloques-csv', required=True, help='CSV de bloques generado por verificar_caminos.py')
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
    parser.add_argument('--umbral-db', type=float, default=0.5, help='Distancia de Bhattacharyya máxima (default: 0.5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas (v5 - Bloques CSV)")
    print("=" * 70)
    print("Python: {}".format(sys.version))
    print("NumPy: {}".format(np.__version__))
    print("=" * 70)
    
    measurement_id, probe_id = extraer_ids_de_archivo(args.crudo)
    if measurement_id and probe_id:
        print("Measurement ID: {}\nProbe ID: {}".format(measurement_id, probe_id))
    print("=" * 70)
    
    serie_completa, longitudes, cambios_longitud = cargar_csv_crudo(args.crudo, dest_hop=args.dest_hop)
    bloques = cargar_bloques_desde_csv(args.bloques_csv)
    dict_categorias = cargar_categorias(args.categorias)
    
    if not serie_completa or not bloques:
        print("[ERROR] No se pudieron cargar los datos o bloques."); sys.exit(1)
    
    print("\n[FASE 2] Procesando {} bloques coherentes...".format(len(bloques)))
    
    todos_segmentos = []
    todos_change_points = []
    estados_por_secuencia_global = []
    estados_suaves_por_secuencia_global = []
    secuencias_global = []
    
    for idx, bloque in enumerate(bloques):
        print("\n  [BLOQUE {}/{}] ID: {}, Longitud: {} hops, Ciclos totales: {}".format(
            idx + 1, len(bloques), bloque['bloque_id'], bloque['longitud'], bloque['ciclos']))
        
        sub_serie = extraer_sub_serie(serie_completa, bloque)
        if len(sub_serie) < 10:
            print("    → Bloque demasiado pequeño, omitido.")
            continue
        
        # >>> PRE-FILTRADO DE OUTLIERS (Percentil 99) <<<
        # Se filtran los valores extremos para cumplir el supuesto Gaussiano del HDP-HMM
        rtts_bloque = [rtt for _, rtt in sub_serie]
        umbral_p99 = np.percentile(rtts_bloque, 99.0)
        sub_serie_filtrada = [(ts, rtt) for ts, rtt in sub_serie if rtt <= umbral_p99]
        n_filtrados = len(sub_serie) - len(sub_serie_filtrada)
        print("    → Outliers filtrados (RTT > P99={:.2f}ms): {}".format(umbral_p99, n_filtrados))
        sub_serie = sub_serie_filtrada
        
        if len(sub_serie) < 10:
            print("    → Bloque demasiado pequeño tras filtrar outliers, omitido.")
            continue

        print("    → Ciclos extraídos para análisis: {}".format(len(sub_serie)))
        
        # Cortar en secuencias por huecos temporales dentro del bloque
        
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
        segmentos = caracterizar_segmentos(sub_serie, secuencias, estados_trabajo, estados_suaves)
        todos_segmentos.extend(segmentos)
        
        # Change-points
        cps = detectar_change_points(secuencias, estados_suaves)
        todos_change_points.extend(cps)
        
        # Acumular para exportación global
        estados_por_secuencia_global.extend(estados_por_secuencia)
        estados_suaves_por_secuencia_global.extend(estados_suaves)
        secuencias_global.extend(secuencias)
    
    # Calcular RTT normal y clasificar segmentos
    rtts = [rtt for _, rtt, _ in serie_completa]
    rtt_normal = np.percentile(rtts, 25)
    print("\n  - RTT Normal (P25 global): {:.2f}ms".format(rtt_normal))
    
    for seg in todos_segmentos:
        seg['tipo_degradacion'] = clasificar_tipo_degradacion(seg, rtt_normal)
        
    metricas_calidad = calcular_calidad_segmentacion(serie_completa, todos_segmentos)
    print("  - Ratio de calidad global: {:.2f}".format(metricas_calidad['ratio_calidad']))
    
    # Cambios topológicos
    todos_change_points, n_topo = detectar_cambios_topologicos(todos_change_points, dict_categorias)
    print("  - Change-points con evento topológico: {}/{} ({:.1f}%)".format(
        n_topo, len(todos_change_points), 100.0 * n_topo / max(len(todos_change_points), 1)))
    
    # Exportar
    exportar_resultados(
        serie_completa, secuencias_global, estados_por_secuencia_global, estados_suaves_por_secuencia_global,
        todos_change_points, todos_segmentos, metricas_calidad, args.output_dir,
        measurement_id, probe_id, longitudes=longitudes, cambios_longitud=cambios_longitud,
        tipo_fusion=tipo_fusion, mapa_fusion=mapa_fusion)
    
    print("\n" + "=" * 70)
    print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("Resultados disponibles en: {}".format(args.output_dir))

if __name__ == '__main__':
    main()
