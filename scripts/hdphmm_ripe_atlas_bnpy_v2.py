# -*- coding: utf-8 -*-
"""
=============================================================================
HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas
+ Caracterización de Segmentos y Detección de Cambios Topológicos
+ Fusión de Estados (Manual y Automática por Bhattacharyya)
+ Detección de Longitud del Path
=============================================================================
Basado en: Mouchet et al., "Large-Scale Characterization and Segmentation of
           Internet Path Delays with Infinite HMMs" (arXiv:1910.12714)

Entorno: Python 2.7 + bnpy (build 8019474-py27_1) + numpy 1.16.6 + scipy 1.2.1

CAMBIOS EN ESTA VERSIÓN:
- Eliminada la correlación con Path Analysis como métrica principal
- Path Analysis se usa solo para detectar cambios topológicos (cambio_ip_asn, cambio_longitud)
- Agregada caracterización detallada de segmentos (duración, media, std, tipo de degradación)
- Agregadas métricas de calidad de segmentación (varianza intra/inter, ratio de calidad)
- Nuevos archivos de salida con estadísticas completas
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import argparse
import csv
import re
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime

try:
    import numpy as np
except ImportError:
    sys.stderr.write("Este script necesita numpy instalado.\n")
    sys.exit(1)


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
TOLERANCIA_MINUTOS = 15  # Ventana temporal para correlacionar eventos

# Mapeo por defecto para fusión manual
MAPA_MACRO_ESTADOS_DEFAULT = {
    10: 0, 13: 0, 14: 0,  # Macro 0: Base Óptimo
    0: 1, 1: 1,            # Macro 1: Degradación Leve
    2: 2,                  # Macro 2: Congestión
    5: 3,                  # Macro 3: Outlier
}

DESCRIPCION_MACRO_ESTADOS = {
    0: "Base Optimo", 1: "Degradacion Leve", 2: "Congestion", 3: "Outlier",
}


# ============================================================================
# UTILIDADES Y ESTRUCTURAS DE DATOS
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
                    orig = int(row[0].strip())
                    macro = int(row[1].strip())
                    mapa[orig] = macro
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
                serie.append((ts, hops[dest_hop]))
        else:
            hop_maximo = max(hops.keys())
            serie.append((ts, hops[hop_maximo]))
    
    print("  - Líneas procesadas: {}".format(n_lineas))
    print("  - Ciclos extraídos: {}".format(len(serie)))
    
    if longitudes:
        long_vals = [l for (_, l) in longitudes]
        long_unicas = sorted(set(long_vals))
        long_counts = defaultdict(int)
        for l in long_vals:
            long_counts[l] += 1
        print("  - Longitudes de path detectadas: {}".format(long_unicas))
        for l in long_unicas:
            print("      {} hops: {} ciclos ({:.1f}%)".format(
                l, long_counts[l], 100.0 * long_counts[l] / len(longitudes)))
    print("  - Cambios de longitud detectados: {}".format(len(cambios_longitud)))
    
    return serie, longitudes, cambios_longitud


def cargar_categorias(path_csv):
    """
    Carga el CSV de categorías del Path Analysis.
    Solo se utilizan para detectar cambios topológicos (cambio_ip_asn, cambio_longitud).
    """
    print("[FASE 1] Cargando categorías Path Analysis (solo cambios topológicos): {}".format(path_csv))
    dict_categorias = defaultdict(lambda: {'categorias': set(), 'conteo': 0})
    
    with open(path_csv, 'r') as f:
        reader = csv.DictReader(f)
        n_lineas = 0
        for row in reader:
            n_lineas += 1
            ts = row['timestamp'].strip()
            cat = row['categoria'].strip()
            # Solo nos interesan cambios topológicos
            if cat in ['cambio_ip_asn', 'cambio_longitud']:
                dict_categorias[ts]['categorias'].add(cat)
                dict_categorias[ts]['conteo'] += 1
    
    print("  - Líneas procesadas: {}".format(n_lineas))
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
# FASE 3b: FUSIÓN DE ESTADOS (MANUAL O AUTOMÁTICA POR BHATTACHARYYA)
# ============================================================================
def fusionar_estados_manual(estados_por_secuencia, mapa_macro):
    print("\n[FASE 3b] Fusionando estados manualmente...")
    print("  - Mapa de fusión: {}".format(mapa_macro))
    estados_fusionados = [[mapa_macro.get(int(e), int(e)) for e in seq] for seq in estados_por_secuencia]
    return estados_fusionados


def _distancia_bhattacharyya(m1, s1, m2, s2):
    """
    Calcula la Distancia de Bhattacharyya entre dos distribuciones Gaussianas.
    """
    s1 = max(s1, 1e-6)
    s2 = max(s2, 1e-6)
    
    var1 = s1 * s1
    var2 = s2 * s2
    var_sum = var1 + var2
    
    term1 = 0.25 * ((m1 - m2) ** 2) / var_sum
    term2 = 0.5 * np.log(var_sum / (2.0 * s1 * s2))
    return float(term1 + term2)


def fusionar_estados_automatico(estados_por_secuencia, todos_rtts, umbral_db):
    """
    Fusiona estados automáticamente usando la Distancia de Bhattacharyya
    y Agrupamiento Jerárquico Aglomerativo con Complete Linkage.
    """
    print("\n[FASE 3b] Fusionando estados automáticamente (Bhattacharyya + Complete Linkage)...")
    print("  - Umbral D_B: {:.2f} (menor = más estricto)".format(umbral_db))
    
    flat_states = [int(e) for arr in estados_por_secuencia for e in arr]
    unique_states = sorted(set(flat_states))
    
    stats = {}
    for s in unique_states:
        indices = [i for i, x in enumerate(flat_states) if x == s]
        rtts_s = [todos_rtts[i] for i in indices]
        stats[s] = {
            'mean': float(np.mean(rtts_s)),
            'std': float(np.std(rtts_s)),
            'count': len(rtts_s)
        }
    
    n = len(unique_states)
    db_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = unique_states[i], unique_states[j]
            d = _distancia_bhattacharyya(stats[s1]['mean'], stats[s1]['std'],
                                         stats[s2]['mean'], stats[s2]['std'])
            db_matrix[i, j] = d
            db_matrix[j, i] = d
    
    print("  - Matriz de distancias D_B calculada ({} estados)".format(n))
    
    clusters = [[s] for s in unique_states]
    
    iteracion = 0
    merged = True
    while merged:
        merged = False
        iteracion += 1
        min_dist = float('inf')
        best_i, best_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_db = 0.0
                for s_i in clusters[i]:
                    idx_i = unique_states.index(s_i)
                    for s_j in clusters[j]:
                        idx_j = unique_states.index(s_j)
                        db = db_matrix[idx_i, idx_j]
                        if db > max_db:
                            max_db = db
                
                if max_db < min_dist:
                    min_dist = max_db
                    best_i, best_j = i, j
        
        if min_dist <= umbral_db:
            clusters[best_i] = clusters[best_i] + clusters[best_j]
            clusters.pop(best_j)
            merged = True
            print("    [Iter {}] Fusionados clústeres (D_B máx = {:.3f} <= {:.2f})".format(
                iteracion, min_dist, umbral_db))
        else:
            print("    [Iter {}] Distancia mínima entre clústeres = {:.3f} > umbral {:.2f}. STOP.".format(
                iteracion, min_dist, umbral_db))
    
    mapa_fusion = {}
    sorted_clusters = sorted(clusters, key=lambda cluster: min(stats[s]['mean'] for s in cluster))
    
    print("  - Mapa de fusión generado ({} macro-estados):".format(len(sorted_clusters)))
    for macro_id, cluster in enumerate(sorted_clusters):
        all_rtts = []
        for s in cluster:
            indices = [i for i, x in enumerate(flat_states) if x == s]
            all_rtts.extend([todos_rtts[i] for i in indices])
        
        macro_mean = np.mean(all_rtts)
        macro_std = np.std(all_rtts)
        macro_count = len(all_rtts)
        macro_pct = 100.0 * macro_count / len(flat_states)
        
        print("      Macro-Estado {}: estados {} | RTT={:.2f}±{:.2f}ms ({} pts, {:.1f}%)".format(
            macro_id, sorted(cluster), macro_mean, macro_std, macro_count, macro_pct))
        
        for s in cluster:
            mapa_fusion[s] = macro_id
    
    estados_fusionados = [[mapa_fusion[int(e)] for e in seq] for seq in estados_por_secuencia]
    
    return estados_fusionados, mapa_fusion


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
            largo = fin - ini
            if largo >= min_dwell: continue
            vecino_izq = runs[idx - 1] if idx > 0 else None
            vecino_der = runs[idx + 1] if idx < len(runs) - 1 else None
            if vecino_izq is not None and vecino_der is not None:
                largo_izq = vecino_izq[1] - vecino_izq[0]
                largo_der = vecino_der[1] - vecino_der[0]
                elegido = vecino_izq if largo_izq >= largo_der else vecino_der
            elif vecino_izq is not None: elegido = vecino_izq
            elif vecino_der is not None: elegido = vecino_der
            else: continue
            nuevo_valor = elegido[2]
            for k in range(ini, fin): estados[k] = nuevo_valor
            hubo_cambio = True
        if not hubo_cambio: break
    return estados


# ============================================================================
# FASE 5: DETECCIÓN DE CHANGE-POINTS
# ============================================================================
def detectar_change_points(secuencias, estados_por_secuencia):
    print("[FASE 5] Detectando change-points...")
    change_points = []
    for seq, estados in zip(secuencias, estados_por_secuencia):
        for i in range(1, len(estados)):
            if estados[i] != estados[i - 1]:
                change_points.append({
                    'timestamp': seq[i][0],
                    'estado_anterior': int(estados[i - 1]),
                    'estado_nuevo': int(estados[i])
                })
    print("  - Change-points detectados: {}".format(len(change_points)))
    return change_points


# ============================================================================
# FASE 5b: INCORPORAR CAMBIOS DE LONGITUD
# ============================================================================
def incorporar_cambios_longitud(cambios_longitud, dict_categorias):
    if not cambios_longitud: return dict_categorias
    print("[FASE 5b] Incorporando {} cambios de longitud como eventos topológicos...".format(len(cambios_longitud)))
    dict_extendido = defaultdict(lambda: {'categorias': set(), 'conteo': 0})
    for ts, info in dict_categorias.items():
        dict_extendido[ts] = {'categorias': set(info['categorias']), 'conteo': info['conteo']}
    for cambio in cambios_longitud:
        ts = cambio['timestamp']
        dict_extendido[ts]['categorias'].add('cambio_longitud_detectado')
        dict_extendido[ts]['conteo'] += 1
    return dict(dict_extendido)


# ============================================================================
# FASE 6: CARACTERIZACIÓN DETALLADA DE SEGMENTOS
# ============================================================================
def caracterizar_segmentos(serie_completa, secuencias, estados_por_secuencia, estados_suaves_por_secuencia):
    """
    Caracteriza cada segmento con estadísticas completas y clasifica el tipo de degradación.
    """
    print("\n[FASE 6] Caracterizando segmentos...")
    
    segmentos = []
    timestamps_ordenados = [ts for ts, _ in serie_completa]
    
    for seq_id, (seq, estados_crudos, estados_suaves) in enumerate(zip(secuencias, estados_por_secuencia, estados_suaves_por_secuencia)):
        # Identificar segmentos dentro de esta secuencia
        segmentos_seq = []
        start_idx = 0
        estado_actual = estados_suaves[0]
        
        for i in range(1, len(estados_suaves)):
            if estados_suaves[i] != estado_actual:
                # Fin del segmento actual
                segmentos_seq.append({
                    'seq_id': seq_id,
                    'estado': int(estado_actual),
                    'start_idx_global': start_idx,
                    'end_idx_global': i - 1,
                    'start_ts': seq[start_idx][0],
                    'end_ts': seq[i - 1][0],
                    'n_ciclos': i - start_idx
                })
                start_idx = i
                estado_actual = estados_suaves[i]
        
        # Último segmento
        segmentos_seq.append({
            'seq_id': seq_id,
            'estado': int(estado_actual),
            'start_idx_global': start_idx,
            'end_idx_global': len(estados_suaves) - 1,
            'start_ts': seq[start_idx][0],
            'end_ts': seq[-1][0],
            'n_ciclos': len(estados_suaves) - start_idx
        })
        
        # Calcular estadísticas para cada segmento
        for seg in segmentos_seq:
            rtts_seg = [serie_completa[idx][1] for idx in range(seg['start_idx_global'], seg['end_idx_global'] + 1)]
            seg['media'] = float(np.mean(rtts_seg))
            seg['mediana'] = float(np.median(rtts_seg))
            seg['std'] = float(np.std(rtts_seg))
            seg['min'] = float(np.min(rtts_seg))
            seg['max'] = float(np.max(rtts_seg))
            seg['duracion_minutos'] = seg['n_ciclos'] * 10  # Asumiendo cadencia de 10 min
        
        segmentos.extend(segmentos_seq)
    
    print("  - Total de segmentos caracterizados: {}".format(len(segmentos)))
    return segmentos


def calcular_calidad_segmentacion(serie_completa, segmentos):
    """
    Calcula métricas de calidad de la segmentación:
    - Varianza intra-segmento (homogeneidad interna)
    - Varianza inter-segmento (separación entre segmentos)
    - Ratio de calidad (inter/intra)
    """
    rtts = [rtt for _, rtt in serie_completa]
    media_global = np.mean(rtts)
    n_total = len(rtts)
    
    # Varianza intra-segmento (promedio ponderado)
    var_intra_total = 0.0
    for seg in segmentos:
        n_seg = seg['n_ciclos']
        var_intra_total += (n_seg / n_total) * (seg['std'] ** 2)
    
    # Varianza inter-segmento
    var_inter = 0.0
    for seg in segmentos:
        n_seg = seg['n_ciclos']
        var_inter += (n_seg / n_total) * ((seg['media'] - media_global) ** 2)
    
    # Ratio de calidad (mayor es mejor)
    ratio_calidad = var_inter / max(var_intra_total, 1e-10)
    
    return {
        'varianza_intra': var_intra_total,
        'varianza_inter': var_inter,
        'ratio_calidad': ratio_calidad,
        'duracion_media_segmentos': np.mean([s['duracion_minutos'] for s in segmentos]),
        'n_segmentos': len(segmentos)
    }


def clasificar_tipo_degradacion(segmento, media_global):
    """
    Clasifica un segmento según su tipo de degradación basado en magnitud y duración.
    """
    desviacion = segmento['media'] - media_global
    duracion = segmento['duracion_minutos']
    
    if duracion > 60 and abs(desviacion) > 5:
        return 'degradacion_severa_sostenida'
    elif duracion > 30 and abs(desviacion) > 2:
        return 'degradacion_moderada_sostenida'
    elif duracion < 10 and abs(desviacion) > 10:
        return 'pico_transitorio'
    elif abs(desviacion) < 1:
        return 'regimen_base'
    else:
        return 'variacion_normal'


# ============================================================================
# FASE 7: DETECCIÓN DE CAMBIOS TOPOLÓGICOS
# ============================================================================
def detectar_cambios_topologicos(change_points, dict_categorias):
    """
    Detecta qué change-points coinciden con eventos topológicos del Path Analysis.
    """
    print("\n[FASE 7] Detectando cambios topológicos...")
    
    cp_dt = {}
    for cp in change_points:
        try:
            cp_dt[cp['timestamp']] = datetime.strptime(cp['timestamp'], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
    
    cat_dt = {}
    for ts_str in dict_categorias.keys():
        try:
            cat_dt[ts_str] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    
    n_coincidencias = 0
    for cp in change_points:
        ts = cp['timestamp']
        if ts not in cp_dt:
            continue
        ts_dt = cp_dt[ts]
        
        for ts_str, dt_cat in cat_dt.items():
            diff_min = abs((ts_dt - dt_cat).total_seconds()) / 60.0
            if diff_min <= TOLERANCIA_MINUTOS:
                cp['evento_topologico'] = True
                cp['categorias_topologicas'] = dict_categorias[ts_str]['categorias']
                n_coincidencias += 1
                break
        else:
            cp['evento_topologico'] = False
            cp['categorias_topologicas'] = set()
    
    print("  - Change-points con evento topológico: {}/{} ({:.1f}%)".format(
        n_coincidencias, len(change_points), 100.0 * n_coincidencias / max(len(change_points), 1)))
    
    return change_points


# ============================================================================
# FASE 8: EXPORTACIÓN DE RESULTADOS
# ============================================================================
def exportar_resultados(serie_completa, secuencias, estados_por_secuencia,
                        estados_suaves_por_secuencia, change_points,
                        segmentos, metricas_calidad, output_dir,
                        measurement_id, probe_id,
                        longitudes=None, cambios_longitud=None,
                        tipo_fusion=None, mapa_fusion=None):
    print("[FASE 8] Exportando resultados a: {}".format(output_dir))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    if measurement_id and probe_id:
        ids_suffix = "_measurement{}_probe{}".format(measurement_id, probe_id)
    else:
        ids_suffix = ""
    
    fusion_suffix = ""
    if tipo_fusion == 'manual': fusion_suffix = "_macro"
    elif tipo_fusion == 'auto': fusion_suffix = "_auto"
    
    # 1. Estados
    path_estados = os.path.join(output_dir, 'estados_hdphmm{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_estados, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'rtt_ms', 'seq_id', 'state_raw', 'state'])
        for seq_id, (seq, crudos, suaves) in enumerate(zip(secuencias, estados_por_secuencia, estados_suaves_por_secuencia)):
            for (ts, rtt), e_crudo, e_suave in zip(seq, crudos, suaves):
                writer.writerow([ts, "{:.3f}".format(rtt), seq_id, int(e_crudo), int(e_suave)])
    print("  - Estados: {}".format(path_estados))
    
    # 2. Longitudes
    if longitudes:
        path_long = os.path.join(output_dir, 'longitudes_path{}{}.csv'.format(ids_suffix, fusion_suffix))
        with open(path_long, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'longitud_path'])
            for ts, lon in longitudes: writer.writerow([ts, lon])
        print("  - Longitudes: {}".format(path_long))
    
    if cambios_longitud:
        path_cambios = os.path.join(output_dir, 'cambios_longitud{}{}.csv'.format(ids_suffix, fusion_suffix))
        with open(path_cambios, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'longitud_anterior', 'longitud_nueva', 'diferencia'])
            for c in cambios_longitud:
                writer.writerow([c['timestamp'], c['longitud_anterior'], c['longitud_nueva'], c['diferencia']])
        print("  - Cambios de longitud: {}".format(path_cambios))
    
    # 3. Segmentos caracterizados
    path_segmentos = os.path.join(output_dir, 'segmentos_caracterizados{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_segmentos, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_id', 'estado', 'start_ts', 'end_ts', 'n_ciclos', 'duracion_minutos',
                        'media', 'mediana', 'std', 'min', 'max', 'tipo_degradacion'])
        for seg in segmentos:
            writer.writerow([seg['seq_id'], seg['estado'], seg['start_ts'], seg['end_ts'],
                           seg['n_ciclos'], seg['duracion_minutos'],
                           "{:.3f}".format(seg['media']), "{:.3f}".format(seg['mediana']),
                           "{:.3f}".format(seg['std']), "{:.3f}".format(seg['min']),
                           "{:.3f}".format(seg['max']), seg['tipo_degradacion']])
    print("  - Segmentos caracterizados: {}".format(path_segmentos))
    
    # 4. Change-points con eventos topológicos
    path_cp = os.path.join(output_dir, 'change_points_topologicos{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_cp, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estado_anterior', 'estado_nuevo', 'evento_topologico', 'categorias_topologicas'])
        for cp in change_points:
            cats_str = "|".join(cp['categorias_topologicas']) if cp['categorias_topologicas'] else 'NINGUNA'
            writer.writerow([cp['timestamp'], cp['estado_anterior'], cp['estado_nuevo'],
                           cp['evento_topologico'], cats_str])
    print("  - Change-points topológicos: {}".format(path_cp))
    
    # 5. Resumen
    path_resumen = os.path.join(output_dir, 'resumen_ejecucion{}{}.txt'.format(ids_suffix, fusion_suffix))
    with open(path_resumen, 'w') as f:
        f.write("=" * 70 + "\nRESUMEN DE EJECUCIÓN HDP-HMM\n" + "=" * 70 + "\n\n")
        if measurement_id and probe_id:
            f.write("Measurement ID: {}\nProbe ID: {}\n\n".format(measurement_id, probe_id))
        f.write("Fecha: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Dataset: {} puntos temporales\n".format(len(serie_completa)))
        f.write("Rango temporal: {} a {}\n".format(serie_completa[0][0], serie_completa[-1][0]))
        f.write("Tipo de fusión: {}\n".format(tipo_fusion or 'Ninguna'))
        if mapa_fusion:
            f.write("Mapa de fusión: {}\n".format(mapa_fusion))
        
        f.write("\n--- ESTADÍSTICAS DE RTT ---\n")
        rtts = [rtt for _, rtt in serie_completa]
        f.write("Min: {:.2f} ms\nMax: {:.2f} ms\nMedia: {:.2f} ms\nDesv. Est.: {:.2f} ms\n".format(
            min(rtts), max(rtts), np.mean(rtts), np.std(rtts)))
        
        if longitudes:
            f.write("\n--- LONGITUD DEL PATH ---\n")
            long_vals = [l for (_, l) in longitudes]
            f.write("Longitudes detectadas: {}\n".format(sorted(set(long_vals))))
            f.write("Longitud más frecuente: {}\n".format(max(set(long_vals), key=long_vals.count)))
            if cambios_longitud: f.write("Cambios de longitud: {}\n".format(len(cambios_longitud)))
        
        f.write("\n--- MODELO HDP-HMM ---\n")
        n_estados = len(set(int(e) for arr in estados_por_secuencia for e in arr))
        f.write("Estados activos: {}\nChange-points detectados: {}\n".format(n_estados, len(change_points)))
        
        f.write("\n--- CARACTERIZACIÓN DE SEGMENTOS ---\n")
        f.write("Total de segmentos: {}\n".format(len(segmentos)))
        tipo_counts = defaultdict(int)
        for seg in segmentos:
            tipo_counts[seg['tipo_degradacion']] += 1
        for tipo, count in sorted(tipo_counts.items(), key=lambda x: -x[1]):
            f.write("  {}: {} segmentos\n".format(tipo, count))
        
        f.write("\n--- MÉTRICAS DE CALIDAD ---\n")
        f.write("Varianza intra-segmento: {:.4f}\n".format(metricas_calidad['varianza_intra']))
        f.write("Varianza inter-segmento: {:.4f}\n".format(metricas_calidad['varianza_inter']))
        f.write("Ratio de calidad: {:.4f}\n".format(metricas_calidad['ratio_calidad']))
        f.write("Duración media de segmentos: {:.2f} minutos\n".format(metricas_calidad['duracion_media_segmentos']))
        
        f.write("\n--- CAMBIOS TOPOLÓGICOS ---\n")
        n_topo = sum(1 for cp in change_points if cp['evento_topologico'])
        f.write("Change-points con evento topológico: {}/{} ({:.1f}%)\n".format(
            n_topo, len(change_points), 100.0 * n_topo / max(len(change_points), 1)))
        
        f.write("\n" + "=" * 70 + "\n")
    print("  - Resumen: {}".format(path_resumen))


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="HDP-HMM con bnpy para segmentación de latencia en RIPE Atlas")
    parser.add_argument('--crudo', required=True, help='CSV crudo de traceroute')
    parser.add_argument('--categorias', required=True, help='CSV de categorías Path Analysis')
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
    
    # Flags de fusión
    parser.add_argument('--macro-estados', action='store_true', help='Fusionar estados manualmente (mapa por defecto o --mapa-macro)')
    parser.add_argument('--mapa-macro', type=str, default=None, help='Archivo CSV con mapa personalizado de macro-estados')
    parser.add_argument('--fusion-auto', action='store_true', help='Fusionar estados automáticamente por Distancia de Bhattacharyya (Complete Linkage)')
    parser.add_argument('--umbral-db', type=float, default=0.5, help='Distancia de Bhattacharyya máxima para fusionar (default: 0.5; rango típico 0.3-1.0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas")
    print("=" * 70)
    print("Python: {}".format(sys.version))
    print("NumPy: {}".format(np.__version__))
    print("=" * 70)
    
    measurement_id, probe_id = extraer_ids_de_archivo(args.crudo)
    if measurement_id and probe_id:
        print("Measurement ID: {}\nProbe ID: {}".format(measurement_id, probe_id))
    else:
        print("[ADVERTENCIA] No se pudo extraer measurement_id/probe_id.")
    print("=" * 70)
    
    serie, longitudes, cambios_longitud = cargar_csv_crudo(args.crudo, dest_hop=args.dest_hop)
    dict_categorias = cargar_categorias(args.categorias)
    if not serie:
        print("[ERROR] No se pudo extraer ninguna muestra de RTT."); sys.exit(1)
    
    secuencias = cortar_en_secuencias(serie, args.gap_minutes)
    secuencias = [s for s in secuencias if len(s) >= 2]
    total_ciclos = sum(len(s) for s in secuencias)
    print("\n[FASE 2] Segmentación temporal:")
    print("  - Serie cargada: {} ciclos totales -> {} secuencias (gap > {} min) -> {} ciclos utilizables".format(
        len(serie), len(secuencias), args.gap_minutes, total_ciclos))
    if not secuencias:
        print("[ERROR] No quedó ninguna secuencia utilizable."); sys.exit(1)
    
    muestra_rtt = np.array([rtt for seq in secuencias for (_, rtt) in seq])
    print("  - RTT observado: media={:.2f}ms, var={:.2f}, desvio={:.2f}ms".format(
        muestra_rtt.mean(), muestra_rtt.var(), muestra_rtt.std()))
    
    hmodel, Data, estados_por_secuencia = correr_hdphmm(
        secuencias, K=args.K, nlap=args.nlap,
        transAlpha=args.trans_alpha, startAlpha=args.start_alpha,
        hmmKappa=args.hmm_kappa, sF=args.sF, seed=args.seed,
        bnpy_outdir=args.bnpy_outdir, alg_name=args.alg_name)
    
    if estados_por_secuencia is None:
        print("[ERROR] No se pudo entrenar el modelo."); sys.exit(1)
    
    # FASE 3b: Fusión
    estados_trabajo = estados_por_secuencia
    tipo_fusion = None
    mapa_fusion = None
    
    if args.fusion_auto:
        todos_rtts = [rtt for seq in secuencias for (_, rtt) in seq]
        estados_trabajo, mapa_fusion = fusionar_estados_automatico(
            estados_por_secuencia, todos_rtts, args.umbral_db)
        tipo_fusion = 'auto'
    elif args.macro_estados:
        mapa_fusion = cargar_mapa_macro_estados(args.mapa_macro)
        estados_trabajo = fusionar_estados_manual(estados_por_secuencia, mapa_fusion)
        tipo_fusion = 'manual'
    
    # FASE 4: Suavizado
    print("\n[FASE 4] Suavizado post-hoc (--min-dwell {})...".format(args.min_dwell))
    n_cambios_crudos = sum(1 for arr in estados_trabajo for i in range(1, len(arr)) if arr[i] != arr[i - 1])
    print("  - Cambios sin suavizar: {}".format(n_cambios_crudos))
    estados_suaves_por_secuencia = [suavizar_estados(arr, args.min_dwell) for arr in estados_trabajo]
    n_cambios_suaves = sum(1 for arr in estados_suaves_por_secuencia for i in range(1, len(arr)) if arr[i] != arr[i - 1])
    print("  - Cambios tras suavizado: {}".format(n_cambios_suaves))
    
    change_points = detectar_change_points(secuencias, estados_suaves_por_secuencia)
    dict_categorias_extendido = incorporar_cambios_longitud(cambios_longitud, dict_categorias)
    
    # FASE 6: Caracterización de segmentos
    segmentos = caracterizar_segmentos(serie, secuencias, estados_por_secuencia, estados_suaves_por_secuencia)
    
    # Calcular RTT normal (Percentil 25) para clasificación
    rtts = [rtt for _, rtt in serie]
    rtt_normal = np.percentile(rtts, 25)
    print("  - RTT Normal (P25): {:.2f}ms".format(rtt_normal))
    
    # Clasificar tipo de degradación para cada segmento
    for seg in segmentos:
        seg['tipo_degradacion'] = clasificar_tipo_degradacion(seg, rtt_normal)
    
    # Calcular métricas de calidad
    metricas_calidad = calcular_calidad_segmentacion(serie, segmentos)
    print("  - Ratio de calidad: {:.2f}".format(metricas_calidad['ratio_calidad']))
    
    # FASE 7: Detección de cambios topológicos
    change_points = detectar_cambios_topologicos(change_points, dict_categorias_extendido)
    
    # FASE 8: Exportación
    exportar_resultados(
        serie, secuencias, estados_por_secuencia, estados_suaves_por_secuencia,
        change_points, segmentos, metricas_calidad, args.output_dir,
        measurement_id, probe_id, longitudes=longitudes, cambios_longitud=cambios_longitud,
        tipo_fusion=tipo_fusion, mapa_fusion=mapa_fusion)
    
    print("\n" + "=" * 70)
    print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("Resultados disponibles en: {}".format(args.output_dir))

if __name__ == '__main__':
    main()
