# -*- coding: utf-8 -*-
"""
=============================================================================
HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas
+ Correlación con Path Analysis, Análisis de Residuos,
  Fusión de Estados (Manual y Automática) y Detección de Longitud del Path
=============================================================================
Basado en: Mouchet et al., "Large-Scale Characterization and Segmentation of
           Internet Path Delays with Infinite HMMs" (arXiv:1910.12714)

Entorno: Python 2.7 + bnpy (build 8019474-py27_1) + numpy 1.16.6 + scipy 1.2.1
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
class UnionFind:
    """Estructura de datos para agrupamiento eficiente (Componentes Conectados)."""
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


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
    print("[FASE 1] Cargando categorías: {}".format(path_csv))
    dict_categorias = defaultdict(lambda: {'categorias': set(), 'conteo': 0})
    
    with open(path_csv, 'r') as f:
        reader = csv.DictReader(f)
        n_lineas = 0
        for row in reader:
            n_lineas += 1
            ts = row['timestamp'].strip()
            cat = row['categoria'].strip()
            dict_categorias[ts]['categorias'].add(cat)
            dict_categorias[ts]['conteo'] += 1
    
    print("  - Líneas procesadas: {}".format(n_lineas))
    print("  - Timestamps con eventos: {}".format(len(dict_categorias)))
    todas_cats = defaultdict(int)
    for info in dict_categorias.values():
        for c in info['categorias']:
            todas_cats[c] += 1
    print("  - Distribución de categorías:")
    for cat, count in sorted(todas_cats.items(), key=lambda x: -x[1]):
        print("      {}: {} timestamps".format(cat, count))
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
# FASE 3b: FUSIÓN DE ESTADOS (MANUAL O AUTOMÁTICA)
# ============================================================================
def fusionar_estados_manual(estados_por_secuencia, mapa_macro):
    print("\n[FASE 3b] Fusionando estados manualmente...")
    print("  - Mapa de fusión: {}".format(mapa_macro))
    estados_fusionados = [[mapa_macro.get(int(e), int(e)) for e in seq] for seq in estados_por_secuencia]
    return estados_fusionados


def fusionar_estados_automatico(estados_por_secuencia, todos_rtts, umbral_media, umbral_std):
    """
    Fusiona estados automáticamente usando componentes conectados (Union-Find).
    Dos estados se fusionan si:
      |media_i - media_j| < umbral_media  Y  |std_i - std_j| < umbral_std
    """
    print("\n[FASE 3b] Fusionando estados automáticamente...")
    print("  - Umbrales: media={:.1f}ms, std={:.1f}ms".format(umbral_media, umbral_std))
    
    flat_states = [int(e) for arr in estados_por_secuencia for e in arr]
    unique_states = sorted(set(flat_states))
    
    # Calcular estadísticas
    stats = {}
    for s in unique_states:
        indices = [i for i, x in enumerate(flat_states) if x == s]
        rtts_s = [todos_rtts[i] for i in indices]
        stats[s] = {
            'mean': float(np.mean(rtts_s)),
            'std': float(np.std(rtts_s)),
            'count': len(rtts_s)
        }
    
    # Agrupamiento por Union-Find
    uf = UnionFind(unique_states)
    for i in range(len(unique_states)):
        for j in range(i + 1, len(unique_states)):
            s1, s2 = unique_states[i], unique_states[j]
            diff_media = abs(stats[s1]['mean'] - stats[s2]['mean'])
            diff_std = abs(stats[s1]['std'] - stats[s2]['std'])
            
            if diff_media < umbral_media and diff_std < umbral_std:
                uf.union(s1, s2)
    
    # Construir clusters y mapa
    clusters = defaultdict(list)
    for s in unique_states:
        clusters[uf.find(s)].append(s)
    
    # Asignar IDs de macro-estados ordenados por el estado original más bajo
    mapa_fusion = {}
    sorted_clusters = sorted(clusters.items(), key=lambda x: min(x[1]))
    
    print("  - Mapa de fusión generado:")
    for macro_id, (root, members) in enumerate(sorted_clusters):
        # Calcular stats combinadas para el macro-estado
        all_rtts = []
        for m in members:
            indices = [i for i, x in enumerate(flat_states) if x == m]
            all_rtts.extend([todos_rtts[i] for i in indices])
        
        macro_mean = np.mean(all_rtts)
        macro_std = np.std(all_rtts)
        macro_count = len(all_rtts)
        macro_pct = 100.0 * macro_count / len(flat_states)
        
        print("      Macro-Estado {}: estados {} | RTT={:.2f}±{:.2f}ms ({} pts, {:.1f}%)".format(
            macro_id, members, macro_mean, macro_std, macro_count, macro_pct))
        
        for m in members:
            mapa_fusion[m] = macro_id
    
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
    print("[FASE 5b] Incorporando {} cambios de longitud como eventos...".format(len(cambios_longitud)))
    dict_extendido = defaultdict(lambda: {'categorias': set(), 'conteo': 0})
    for ts, info in dict_categorias.items():
        dict_extendido[ts] = {'categorias': set(info['categorias']), 'conteo': info['conteo']}
    for cambio in cambios_longitud:
        ts = cambio['timestamp']
        dict_extendido[ts]['categorias'].add('cambio_longitud_detectado')
        dict_extendido[ts]['conteo'] += 1
    return dict(dict_extendido)


# ============================================================================
# FASE 6: CORRELACIÓN CON PATH ANALYSIS
# ============================================================================
def correlacionar_con_categorias(change_points, dict_categorias):
    print("[FASE 6] Correlacionando con Path Analysis (tolerancia: {} min)...".format(TOLERANCIA_MINUTOS))
    cp_dt = {}
    for cp in change_points:
        try: cp_dt[cp['timestamp']] = datetime.strptime(cp['timestamp'], "%Y-%m-%d %H:%M:%S")
        except ValueError: continue
    
    cat_dt = {}
    for ts_str in dict_categorias.keys():
        try: cat_dt[ts_str] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError: continue
    
    resultados = []
    n_coincidencias = 0
    for cp in change_points:
        ts_cp_str = cp['timestamp']
        if ts_cp_str not in cp_dt: continue
        ts_cp = cp_dt[ts_cp_str]
        categorias_ventana = defaultdict(int)
        for ts_str, dt_cat in cat_dt.items():
            diff_min = abs((ts_cp - dt_cat).total_seconds()) / 60.0
            if diff_min <= TOLERANCIA_MINUTOS:
                for cat in dict_categorias[ts_str]['categorias']:
                    categorias_ventana[cat] += 1
        
        if categorias_ventana:
            cat_dominante = max(categorias_ventana.items(), key=lambda x: x[1])[0]
            coincide = True
            n_coincidencias += 1
        else:
            cat_dominante = None
            coincide = False
        
        resultados.append({
            'timestamp': ts_cp_str, 'estado_anterior': cp['estado_anterior'],
            'estado_nuevo': cp['estado_nuevo'], 'categoria_path_analysis': cat_dominante,
            'categorias_en_ventana': dict(categorias_ventana), 'coincide': coincide
        })
    print("  - Change-points con evento correspondiente: {}/{} ({:.1f}%)".format(
        n_coincidencias, len(change_points), 100.0 * n_coincidencias / max(len(change_points), 1)))
    return resultados


# ============================================================================
# FASE 7: ANÁLISIS DE RESIDUOS
# ============================================================================
def analizar_residuos(resultados_correlacion, serie_completa):
    print("[FASE 7] Analizando residuos (contribución original)...")
    rtt_dict = {ts: rtt for ts, rtt in serie_completa}
    timestamps_ordenados = [ts for ts, _ in serie_completa]
    residuos = []
    for res in resultados_correlacion:
        if not res['coincide']:
            ts = res['timestamp']
            try: idx = timestamps_ordenados.index(ts)
            except ValueError: continue
            if idx > 0 and idx < len(timestamps_ordenados):
                rtt_antes = rtt_dict[timestamps_ordenados[idx - 1]]
                rtt_despues = rtt_dict[timestamps_ordenados[idx]]
                delta_rtt = rtt_despues - rtt_antes
                pct_cambio = 100.0 * delta_rtt / max(rtt_antes, 0.001)
            else:
                rtt_antes = None; rtt_despues = rtt_dict.get(ts, 0); delta_rtt = 0.0; pct_cambio = 0.0
            residuos.append({
                'timestamp': ts, 'estado_anterior': res['estado_anterior'],
                'estado_nuevo': res['estado_nuevo'], 'rtt_antes_ms': rtt_antes,
                'rtt_despues_ms': rtt_despues, 'delta_rtt_ms': delta_rtt, 'pct_cambio': pct_cambio
            })
    print("  - Residuos identificados: {}".format(len(residuos)))
    return residuos


# ============================================================================
# FASE 8: EXPORTACIÓN DE RESULTADOS
# ============================================================================
def exportar_resultados(serie_completa, secuencias, estados_por_secuencia,
                        estados_suaves_por_secuencia, change_points,
                        resultados_correlacion, residuos, output_dir,
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
    
    # 3. Correlación
    path_cp = os.path.join(output_dir, 'change_points_correlacion{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_cp, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estado_anterior', 'estado_nuevo', 'categoria_path_analysis', 'categorias_en_ventana', 'coincide'])
        for res in resultados_correlacion:
            cats_str = "|".join("{}:{}".format(k, v) for k, v in res['categorias_en_ventana'].items())
            writer.writerow([res['timestamp'], res['estado_anterior'], res['estado_nuevo'],
                           res['categoria_path_analysis'] or 'NINGUNA', cats_str, res['coincide']])
    print("  - Correlación: {}".format(path_cp))
    
    # 4. Residuos
    path_res = os.path.join(output_dir, 'residuos_contribucion_original{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_res, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estado_anterior', 'estado_nuevo', 'rtt_antes_ms', 'rtt_despues_ms', 'delta_rtt_ms', 'pct_cambio'])
        for r in residuos:
            writer.writerow([r['timestamp'], r['estado_anterior'], r['estado_nuevo'],
                           "{:.3f}".format(r['rtt_antes_ms']) if r['rtt_antes_ms'] else '',
                           "{:.3f}".format(r['rtt_despues_ms']), "{:.3f}".format(r['delta_rtt_ms']), "{:.2f}".format(r['pct_cambio'])])
    print("  - Residuos: {}".format(path_res))
    
    # 5. Segmentos
    path_eventos = os.path.join(output_dir, 'segmentos_hdphmm{}{}.csv'.format(ids_suffix, fusion_suffix))
    with open(path_eventos, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'categoria', 'estado_anterior', 'estado_nuevo'])
        for cp in change_points:
            writer.writerow([cp['timestamp'], 'cambio_segmento_hdphmm', cp['estado_anterior'], cp['estado_nuevo']])
    print("  - Eventos de segmento: {}".format(path_eventos))
    
    # 6. Resumen
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
        
        f.write("\n--- CORRELACIÓN CON PATH ANALYSIS ---\n")
        n_coinc = sum(1 for r in resultados_correlacion if r['coincide'])
        f.write("Coincidencias: {}/{} ({:.1f}%)\n".format(n_coinc, len(resultados_correlacion), 100.0 * n_coinc / max(len(resultados_correlacion), 1)))
        f.write("\n--- RESIDUOS (CONTRIBUCIÓN ORIGINAL) ---\nTotal residuos: {}\n\n".format(len(residuos)))
        f.write("=" * 70 + "\n")
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
    parser.add_argument('--fusion-auto', action='store_true', help='Fusionar estados automáticamente por criterio estadístico')
    parser.add_argument('--umbral-media', type=float, default=2.0, help='Diferencia máxima de media para fusionar auto (ms, default: 2.0)')
    parser.add_argument('--umbral-std', type=float, default=1.0, help='Diferencia máxima de std para fusionar auto (ms, default: 1.0)')
    
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
            estados_por_secuencia, todos_rtts, args.umbral_media, args.umbral_std)
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
    resultados_correlacion = correlacionar_con_categorias(change_points, dict_categorias_extendido)
    residuos = analizar_residuos(resultados_correlacion, serie)
    
    exportar_resultados(
        serie, secuencias, estados_por_secuencia, estados_suaves_por_secuencia,
        change_points, resultados_correlacion, residuos, args.output_dir,
        measurement_id, probe_id, longitudes=longitudes, cambios_longitud=cambios_longitud,
        tipo_fusion=tipo_fusion, mapa_fusion=mapa_fusion)
    
    print("\n" + "=" * 70)
    print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("Resultados disponibles en: {}".format(args.output_dir))

if __name__ == '__main__':
    main()
