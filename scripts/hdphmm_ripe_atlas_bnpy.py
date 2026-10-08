# -*- coding: utf-8 -*-
"""
=============================================================================
HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas
+ Correlación con Path Analysis, Análisis de Residuos,
  Fusión de Macro-Estados y Detección de Cambios de Longitud del Path
=============================================================================
Basado en: Mouchet et al., "Large-Scale Characterization and Segmentation of
           Internet Path Delays with Infinite HMMs" (arXiv:1910.12714)

Entorno: Python 2.7 + bnpy (build 8019474-py27_1) + numpy 1.16.6 + scipy 1.2.1

OBJETIVO:
    1. Extraer serie temporal end-to-end de RTT desde CSV crudo de traceroute
    2. Detectar cambios en la longitud del path (número de hops)
    3. Segmentar en secuencias independientes por huecos temporales
    4. Entrenar HDP-HMM con bnpy para inferir estados latentes
    5. Opcionalmente fusionar estados en macro-estados (post-hoc)
    6. Suavizar estados para eliminar chattering
    7. Detectar change-points (transiciones entre estados)
    8. Correlacionar con categorías del Path Analysis de RIPE Atlas
    9. Identificar residuos (cambios detectados por HDP-HMM no explicados
       por el Path Analysis) -> contribución original

USO:
    python hdphmm_ripe_atlas_bnpy.py \
        --crudo historial_traceroute_measurement126502326_probe64883.csv \
        --categorias eventos_measurement126502326_probe64883.csv \
        --output-dir resultados_hdphmm \
        --gap-minutes 90 --K 15 --nlap 100 --sF 10 \
        --macro-estados --min-dwell 3
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

# Mapeo por defecto de estados a macro-estados
# Basado en la observación empírica de que:
#   - Estados 10, 13, 14 -> Macro 0 (Régimen Base Óptimo, ~33ms)
#   - Estados 0, 1       -> Macro 1 (Degradación Leve, ~34-35ms)
#   - Estado 2           -> Macro 2 (Congestión/Anomalía, ~45ms)
#   - Estado 5           -> Macro 3 (Outlier extremo, ~61ms)
# Los estados no listados se mantienen con su número original.
MAPA_MACRO_ESTADOS_DEFAULT = {
    10: 0, 13: 0, 14: 0,  # Macro 0: Base Óptimo
    0: 1, 1: 1,            # Macro 1: Degradación Leve
    2: 2,                  # Macro 2: Congestión
    5: 3,                  # Macro 3: Outlier
}

DESCRIPCION_MACRO_ESTADOS = {
    0: "Base Optimo",
    1: "Degradacion Leve",
    2: "Congestion",
    3: "Outlier",
}


# ============================================================================
# UTILIDADES
# ============================================================================
def extraer_ids_de_archivo(filename):
    """
    Extrae measurement_id y probe_id del nombre del archivo.
    
    Formato esperado: historial_traceroute_measurement{ID}_probe{ID}.csv
    o: eventos_measurement{ID}_probe{ID}.csv
    
    Returns:
        tuple: (measurement_id, probe_id) o (None, None) si no se encuentran
    """
    basename = os.path.basename(filename)
    
    # Buscar patrón measurement{digits}_probe{digits}
    match = re.search(r'measurement(\d+)_probe(\d+)', basename)
    if match:
        return match.group(1), match.group(2)
    
    return None, None


def cargar_mapa_macro_estados(ruta_archivo):
    """
    Carga un mapa de macro-estados desde un archivo CSV con formato:
        estado_original,macro_estado
    
    Si el archivo no existe o está vacío, retorna el mapa por defecto.
    """
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
    """
    Lee el CSV crudo y extrae el RTT del hop de destino por ciclo.
    Detecta cambios en la longitud del path (número de hops).
    
    Returns:
        tuple: (serie, longitudes, cambios_longitud)
            - serie: lista de (timestamp_str, rtt_ms)
            - longitudes: lista de (timestamp_str, longitud_path)
            - cambios_longitud: lista de dicts con info de cambios
    """
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
        
        # Detectar cambio de longitud del path
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
    
    # Resumen de longitudes
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
    """
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
    """
    Corta la serie en secuencias independientes cada vez que el hueco
    temporal supera gap_minutes.
    """
    if not serie:
        return []
    
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
    """
    Ejecuta el HDP-HMM con bnpy usando la API correcta para el build 8019474.
    """
    print("[FASE 3] Entrenando HDP-HMM con bnpy...")
    print("  - Parámetros: K={}, nlap={}, transAlpha={}, startAlpha={}, hmmKappa={}, sF={}".format(
        K, nlap, transAlpha, startAlpha, hmmKappa, sF))
    
    # Configurar BNPYOUTDIR antes del import
    if not os.environ.get("BNPYOUTDIR"):
        os.environ["BNPYOUTDIR"] = bnpy_outdir
    if not os.path.isdir(os.environ["BNPYOUTDIR"]):
        os.makedirs(os.environ["BNPYOUTDIR"])
    
    # Import tardío de bnpy
    try:
        import bnpy
    except ImportError:
        print("  [ERROR] bnpy no está disponible en este intérprete.")
        return None, None, None
    
    # Preparar datos para GroupXData
    X = np.concatenate(
        [np.array([rtt for (_, rtt) in seq], dtype=np.float64).reshape(-1, 1)
         for seq in secuencias],
        axis=0,
    )
    doc_range = [0]
    for seq in secuencias:
        doc_range.append(doc_range[-1] + len(seq))
    doc_range = np.array(doc_range, dtype=np.int32)
    
    Data = bnpy.data.GroupXData(X=X, doc_range=doc_range)
    Data.name = "ripe_atlas_rtt"
    
    # Ejecutar HDP-HMM
    try:
        hmodel, RInfo = bnpy.run(
            Data, "HDPHMM", "Gauss", alg_name,
            jobname="hdphmm-segment",
            nLap=nlap, nTask=1, nBatch=1,
            K=K, initname="randexamples",
            transAlpha=transAlpha, startAlpha=startAlpha, hmmKappa=hmmKappa,
            sF=sF, ECovMat="eye",
            printEvery=25, saveEvery=-1, traceEvery=-1,
            seed=seed,
            doWriteStdOut=False,
        )
    except Exception as e:
        print("  [ERROR] Entrenando HDP-HMM: {}".format(e))
        return None, None, None
    
    # Extraer estados por ciclo
    LP = hmodel.calc_local_params(Data)
    resp = LP["resp"]
    estados = resp.argmax(axis=1)
    
    estados_por_secuencia = []
    for i in range(len(secuencias)):
        ini, fin = doc_range[i], doc_range[i + 1]
        estados_por_secuencia.append(estados[ini:fin])
    
    # Estadísticas del modelo
    n_estados_usados = len(set(int(e) for arr in estados_por_secuencia for e in arr))
    print("  - Estados activos usados: {} (de K={} truncados)".format(n_estados_usados, K))
    
    # Estadísticas por estado
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
# FASE 3b: FUSIÓN DE MACRO-ESTADOS (POST-HOC)
# ============================================================================
def fusionar_estados(estados_por_secuencia, mapa_macro):
    """
    Fusiona estados en macro-estados según el mapa proporcionado.
    Los estados no listados en el mapa se mantienen con su número original.
    """
    if not mapa_macro:
        return estados_por_secuencia
    
    print("\n[FASE 3b] Fusionando estados en macro-estados...")
    print("  - Mapa de fusión: {}".format(mapa_macro))
    
    estados_fusionados = []
    for secuencia in estados_por_secuencia:
        estados_fusionados.append([mapa_macro.get(int(e), int(e)) for e in secuencia])
    
    # Estadísticas de macro-estados
    todos_macro = [int(e) for arr in estados_fusionados for e in arr]
    macro_unicos = sorted(set(todos_macro))
    print("  - Macro-estados resultantes: {}".format(macro_unicos))
    
    todos_rtts = []
    # Necesitamos reconstruir la lista plana de RTTs en el mismo orden
    # (esto se hace en la función que llama, pero aquí mostramos distribución)
    for macro in macro_unicos:
        count = sum(1 for e in todos_macro if e == macro)
        pct = 100.0 * count / len(todos_macro)
        desc = DESCRIPCION_MACRO_ESTADOS.get(macro, "Sin descripcion")
        print("      Macro-Estado {} ({}): {} puntos ({:.1f}%)".format(
            macro, desc, count, pct))
    
    return estados_fusionados


# ============================================================================
# FASE 4: SUAVIZADO POST-HOC
# ============================================================================
def suavizar_estados(estados, min_dwell):
    """
    Fusiona corridas de estados más cortas que min_dwell con la corrida vecina más larga.
    """
    estados = list(estados)
    n = len(estados)
    if n == 0 or min_dwell <= 1:
        return estados
    
    for _ in range(20):
        runs = []
        i = 0
        while i < n:
            j = i
            while j < n and estados[j] == estados[i]:
                j += 1
            runs.append([i, j, estados[i]])
            i = j
        
        if len(runs) <= 1:
            break
        
        hubo_cambio = False
        for idx in range(len(runs)):
            ini, fin, val = runs[idx]
            largo = fin - ini
            if largo >= min_dwell:
                continue
            vecino_izq = runs[idx - 1] if idx > 0 else None
            vecino_der = runs[idx + 1] if idx < len(runs) - 1 else None
            if vecino_izq is not None and vecino_der is not None:
                largo_izq = vecino_izq[1] - vecino_izq[0]
                largo_der = vecino_der[1] - vecino_der[0]
                elegido = vecino_izq if largo_izq >= largo_der else vecino_der
            elif vecino_izq is not None:
                elegido = vecino_izq
            elif vecino_der is not None:
                elegido = vecino_der
            else:
                continue
            nuevo_valor = elegido[2]
            for k in range(ini, fin):
                estados[k] = nuevo_valor
            hubo_cambio = True
        
        if not hubo_cambio:
            break
    
    return estados


# ============================================================================
# FASE 5: DETECCIÓN DE CHANGE-POINTS
# ============================================================================
def detectar_change_points(secuencias, estados_por_secuencia):
    """
    Detecta change-points como transiciones entre estados dentro de cada secuencia.
    """
    print("[FASE 5] Detectando change-points...")
    
    change_points = []
    
    for seq, estados in zip(secuencias, estados_por_secuencia):
        for i in range(1, len(estados)):
            if estados[i] != estados[i - 1]:
                ts = seq[i][0]
                estado_anterior = int(estados[i - 1])
                estado_nuevo = int(estados[i])
                change_points.append({
                    'timestamp': ts,
                    'estado_anterior': estado_anterior,
                    'estado_nuevo': estado_nuevo
                })
    
    print("  - Change-points detectados: {}".format(len(change_points)))
    
    return change_points


# ============================================================================
# FASE 5b: INCORPORAR CAMBIOS DE LONGITUD COMO EVENTOS
# ============================================================================
def incorporar_cambios_longitud(cambios_longitud, dict_categorias):
    """
    Agrega los cambios de longitud del path al diccionario de categorías
    como eventos adicionales con categoría 'cambio_longitud_detectado'.
    Esto permite que la correlación los considere al buscar coincidencias.
    """
    if not cambios_longitud:
        return dict_categorias
    
    print("[FASE 5b] Incorporando {} cambios de longitud como eventos...".format(
        len(cambios_longitud)))
    
    # Copiar para no modificar el original
    dict_extendido = defaultdict(lambda: {'categorias': set(), 'conteo': 0})
    for ts, info in dict_categorias.items():
        dict_extendido[ts] = {
            'categorias': set(info['categorias']),
            'conteo': info['conteo']
        }
    
    for cambio in cambios_longitud:
        ts = cambio['timestamp']
        dict_extendido[ts]['categorias'].add('cambio_longitud_detectado')
        dict_extendido[ts]['conteo'] += 1
    
    return dict(dict_extendido)


# ============================================================================
# FASE 6: CORRELACIÓN CON PATH ANALYSIS
# ============================================================================
def correlacionar_con_categorias(change_points, dict_categorias):
    """
    Cruza los change-points con los eventos del Path Analysis.
    """
    print("[FASE 6] Correlacionando con Path Analysis (tolerancia: {} min)...".format(
        TOLERANCIA_MINUTOS))
    
    # Parsear timestamps de change-points
    cp_dt = {}
    for cp in change_points:
        try:
            cp_dt[cp['timestamp']] = datetime.strptime(cp['timestamp'], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    
    # Parsear timestamps de categorías
    cat_dt = {}
    for ts_str in dict_categorias.keys():
        try:
            cat_dt[ts_str] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    
    resultados = []
    n_coincidencias = 0
    
    for cp in change_points:
        ts_cp_str = cp['timestamp']
        if ts_cp_str not in cp_dt:
            continue
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
            'timestamp': ts_cp_str,
            'estado_anterior': cp['estado_anterior'],
            'estado_nuevo': cp['estado_nuevo'],
            'categoria_path_analysis': cat_dominante,
            'categorias_en_ventana': dict(categorias_ventana),
            'coincide': coincide
        })
    
    print("  - Change-points con evento correspondiente: {}/{} ({:.1f}%)".format(
        n_coincidencias, len(change_points),
        100.0 * n_coincidencias / max(len(change_points), 1)))
    
    return resultados


# ============================================================================
# FASE 7: ANÁLISIS DE RESIDUOS (CONTRIBUCIÓN ORIGINAL)
# ============================================================================
def analizar_residuos(resultados_correlacion, serie_completa):
    """
    Identifica change-points sin evento correspondiente en Path Analysis.
    """
    print("[FASE 7] Analizando residuos (contribución original)...")
    
    # Crear diccionario de RTT por timestamp
    rtt_dict = {ts: rtt for ts, rtt in serie_completa}
    timestamps_ordenados = [ts for ts, _ in serie_completa]
    
    residuos = []
    
    for res in resultados_correlacion:
        if not res['coincide']:
            ts = res['timestamp']
            
            # Encontrar índice en serie completa
            try:
                idx = timestamps_ordenados.index(ts)
            except ValueError:
                continue
            
            if idx > 0 and idx < len(timestamps_ordenados):
                rtt_antes = rtt_dict[timestamps_ordenados[idx - 1]]
                rtt_despues = rtt_dict[timestamps_ordenados[idx]]
                delta_rtt = rtt_despues - rtt_antes
                pct_cambio = 100.0 * delta_rtt / max(rtt_antes, 0.001)
            else:
                rtt_antes = None
                rtt_despues = rtt_dict.get(ts, 0)
                delta_rtt = 0.0
                pct_cambio = 0.0
            
            residuos.append({
                'timestamp': ts,
                'estado_anterior': res['estado_anterior'],
                'estado_nuevo': res['estado_nuevo'],
                'rtt_antes_ms': rtt_antes,
                'rtt_despues_ms': rtt_despues,
                'delta_rtt_ms': delta_rtt,
                'pct_cambio': pct_cambio
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
                        mapa_macro=None, usar_macro_estados=False):
    """
    Exporta todos los resultados a archivos CSV con nombres que incluyen
    measurement_id y probe_id para evitar sobrescritura.
    """
    print("[FASE 8] Exportando resultados a: {}".format(output_dir))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construir sufijo de IDs
    if measurement_id and probe_id:
        ids_suffix = "_measurement{}_probe{}".format(measurement_id, probe_id)
    else:
        ids_suffix = ""
    
    # Sufijo adicional si se usaron macro-estados
    macro_suffix = "_macro" if usar_macro_estados else ""
    
    # 1. Estados por ciclo (completo)
    path_estados = os.path.join(output_dir, 'estados_hdphmm{}{}.csv'.format(ids_suffix, macro_suffix))
    with open(path_estados, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'rtt_ms', 'seq_id', 'state_raw', 'state'])
        for seq_id, (seq, crudos, suaves) in enumerate(
            zip(secuencias, estados_por_secuencia, estados_suaves_por_secuencia)
        ):
            for (ts, rtt), e_crudo, e_suave in zip(seq, crudos, suaves):
                writer.writerow([ts, "{:.3f}".format(rtt), seq_id, int(e_crudo), int(e_suave)])
    print("  - Estados: {}".format(path_estados))
    
    # 1b. Longitudes de path por ciclo (nuevo)
    if longitudes:
        path_long = os.path.join(output_dir, 'longitudes_path{}{}.csv'.format(ids_suffix, macro_suffix))
        with open(path_long, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'longitud_path'])
            for ts, lon in longitudes:
                writer.writerow([ts, lon])
        print("  - Longitudes: {}".format(path_long))
    
    # 1c. Cambios de longitud detectados (nuevo)
    if cambios_longitud:
        path_cambios_long = os.path.join(output_dir, 'cambios_longitud{}{}.csv'.format(ids_suffix, macro_suffix))
        with open(path_cambios_long, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'longitud_anterior', 'longitud_nueva', 'diferencia'])
            for c in cambios_longitud:
                writer.writerow([c['timestamp'], c['longitud_anterior'],
                               c['longitud_nueva'], c['diferencia']])
        print("  - Cambios de longitud: {}".format(path_cambios_long))
    
    # 2. Change-points con correlación
    path_cp = os.path.join(output_dir, 'change_points_correlacion{}{}.csv'.format(ids_suffix, macro_suffix))
    with open(path_cp, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estado_anterior', 'estado_nuevo',
                         'categoria_path_analysis', 'categorias_en_ventana', 'coincide'])
        for res in resultados_correlacion:
            cats_str = "|".join("{}:{}".format(k, v) for k, v in res['categorias_en_ventana'].items())
            writer.writerow([res['timestamp'], res['estado_anterior'], res['estado_nuevo'],
                           res['categoria_path_analysis'] or 'NINGUNA',
                           cats_str, res['coincide']])
    print("  - Correlación: {}".format(path_cp))
    
    # 3. Residuos (contribución original)
    path_res = os.path.join(output_dir, 'residuos_contribucion_original{}{}.csv'.format(ids_suffix, macro_suffix))
    with open(path_res, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estado_anterior', 'estado_nuevo',
                         'rtt_antes_ms', 'rtt_despues_ms', 'delta_rtt_ms', 'pct_cambio'])
        for r in residuos:
            writer.writerow([r['timestamp'], r['estado_anterior'], r['estado_nuevo'],
                           "{:.3f}".format(r['rtt_antes_ms']) if r['rtt_antes_ms'] else '',
                           "{:.3f}".format(r['rtt_despues_ms']),
                           "{:.3f}".format(r['delta_rtt_ms']),
                           "{:.2f}".format(r['pct_cambio'])])
    print("  - Residuos: {}".format(path_res))
    
    # 4. Eventos de cambio de segmento (formato compatible con Path Analysis)
    path_eventos = os.path.join(output_dir, 'segmentos_hdphmm{}{}.csv'.format(ids_suffix, macro_suffix))
    with open(path_eventos, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'categoria', 'estado_anterior', 'estado_nuevo'])
        for cp in change_points:
            writer.writerow([cp['timestamp'], 'cambio_segmento_hdphmm',
                           cp['estado_anterior'], cp['estado_nuevo']])
    print("  - Eventos de segmento: {}".format(path_eventos))
    
    # 5. Resumen estadístico
    path_resumen = os.path.join(output_dir, 'resumen_ejecucion{}{}.txt'.format(ids_suffix, macro_suffix))
    with open(path_resumen, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RESUMEN DE EJECUCIÓN HDP-HMM\n")
        f.write("=" * 70 + "\n\n")
        if measurement_id and probe_id:
            f.write("Measurement ID: {}\n".format(measurement_id))
            f.write("Probe ID: {}\n".format(probe_id))
            f.write("\n")
        f.write("Fecha: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Dataset: {} puntos temporales\n".format(len(serie_completa)))
        f.write("Rango temporal: {} a {}\n".format(serie_completa[0][0], serie_completa[-1][0]))
        f.write("Macro-estados activados: {}\n".format(usar_macro_estados))
        if usar_macro_estados and mapa_macro:
            f.write("Mapa de fusión: {}\n".format(mapa_macro))
        f.write("\n--- ESTADÍSTICAS DE RTT ---\n")
        rtts = [rtt for _, rtt in serie_completa]
        f.write("Min: {:.2f} ms\n".format(min(rtts)))
        f.write("Max: {:.2f} ms\n".format(max(rtts)))
        f.write("Media: {:.2f} ms\n".format(np.mean(rtts)))
        f.write("Desv. Est.: {:.2f} ms\n".format(np.std(rtts)))
        
        # Estadísticas de longitud del path
        if longitudes:
            f.write("\n--- LONGITUD DEL PATH ---\n")
            long_vals = [l for (_, l) in longitudes]
            f.write("Longitudes detectadas: {}\n".format(sorted(set(long_vals))))
            f.write("Longitud más frecuente: {}\n".format(
                max(set(long_vals), key=long_vals.count)))
            if cambios_longitud:
                f.write("Cambios de longitud: {}\n".format(len(cambios_longitud)))
        
        f.write("\n--- MODELO HDP-HMM ---\n")
        n_estados = len(set(int(e) for arr in estados_por_secuencia for e in arr))
        f.write("Estados activos: {}\n".format(n_estados))
        f.write("Change-points detectados: {}\n".format(len(change_points)))
        f.write("\n--- CORRELACIÓN CON PATH ANALYSIS ---\n")
        n_coinc = sum(1 for r in resultados_correlacion if r['coincide'])
        f.write("Coincidencias: {}/{} ({:.1f}%)\n".format(
            n_coinc, len(resultados_correlacion),
            100.0 * n_coinc / max(len(resultados_correlacion), 1)))
        f.write("\n--- RESIDUOS (CONTRIBUCIÓN ORIGINAL) ---\n")
        f.write("Total residuos: {}\n".format(len(residuos)))
        f.write("\n" + "=" * 70 + "\n")
    print("  - Resumen: {}".format(path_resumen))


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HDP-HMM con bnpy para segmentación de latencia en RIPE Atlas")
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
    parser.add_argument('--macro-estados', action='store_true',
                        help='Fusionar estados en macro-estados (post-hoc)')
    parser.add_argument('--mapa-macro', type=str, default=None,
                        help='Archivo CSV con mapa personalizado de macro-estados '
                             '(formato: estado_original,macro_estado)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("HDP-HMM con bnpy para Segmentación de Latencia en RIPE Atlas")
    print("=" * 70)
    print("Python: {}".format(sys.version))
    print("NumPy: {}".format(np.__version__))
    print("=" * 70)
    
    # Extraer measurement_id y probe_id del archivo de entrada
    measurement_id, probe_id = extraer_ids_de_archivo(args.crudo)
    if measurement_id and probe_id:
        print("Measurement ID: {}".format(measurement_id))
        print("Probe ID: {}".format(probe_id))
    else:
        print("[ADVERTENCIA] No se pudo extraer measurement_id/probe_id del nombre del archivo.")
        print("  Los archivos de salida no tendrán sufijo de IDs.")
    print("=" * 70)
    
    # FASE 1: Carga de datos (ahora retorna también longitudes y cambios)
    serie, longitudes, cambios_longitud = cargar_csv_crudo(args.crudo, dest_hop=args.dest_hop)
    dict_categorias = cargar_categorias(args.categorias)
    
    if not serie:
        print("[ERROR] No se pudo extraer ninguna muestra de RTT.")
        sys.exit(1)
    
    # FASE 2: Segmentación temporal
    secuencias = cortar_en_secuencias(serie, args.gap_minutes)
    secuencias = [s for s in secuencias if len(s) >= 2]
    
    total_ciclos = sum(len(s) for s in secuencias)
    print("\n[FASE 2] Segmentación temporal:")
    print("  - Serie cargada: {} ciclos totales -> {} secuencias (gap > {} min) -> {} ciclos utilizables".format(
        len(serie), len(secuencias), args.gap_minutes, total_ciclos))
    
    if not secuencias:
        print("[ERROR] No quedó ninguna secuencia utilizable.")
        sys.exit(1)
    
    # Estadísticas de RTT
    muestra_rtt = np.array([rtt for seq in secuencias for (_, rtt) in seq])
    print("  - RTT observado: media={:.2f}ms, var={:.2f}, desvio={:.2f}ms".format(
        muestra_rtt.mean(), muestra_rtt.var(), muestra_rtt.std()))
    print("  - (considera --sF cercano a la varianza si el ajuste sale raro)")
    
    # FASE 3: Entrenamiento HDP-HMM
    hmodel, Data, estados_por_secuencia = correr_hdphmm(
        secuencias, K=args.K, nlap=args.nlap,
        transAlpha=args.trans_alpha, startAlpha=args.start_alpha,
        hmmKappa=args.hmm_kappa, sF=args.sF, seed=args.seed,
        bnpy_outdir=args.bnpy_outdir, alg_name=args.alg_name,
    )
    
    if estados_por_secuencia is None:
        print("[ERROR] No se pudo entrenar el modelo.")
        sys.exit(1)
    
    # FASE 3b: Fusión de macro-estados (opcional)
    estados_trabajo = estados_por_secuencia
    mapa_macro = None
    if args.macro_estados:
        mapa_macro = cargar_mapa_macro_estados(args.mapa_macro)
        estados_trabajo = fusionar_estados(estados_por_secuencia, mapa_macro)
    
    # FASE 4: Suavizado post-hoc
    print("\n[FASE 4] Suavizado post-hoc (--min-dwell {})...".format(args.min_dwell))
    n_cambios_crudos = sum(
        1 for arr in estados_trabajo for i in range(1, len(arr)) if arr[i] != arr[i - 1]
    )
    print("  - Cambios sin suavizar: {}".format(n_cambios_crudos))
    
    estados_suaves_por_secuencia = [
        suavizar_estados(arr, args.min_dwell) for arr in estados_trabajo
    ]
    n_cambios_suaves = sum(
        1 for arr in estados_suaves_por_secuencia for i in range(1, len(arr)) if arr[i] != arr[i - 1]
    )
    print("  - Cambios tras suavizado: {}".format(n_cambios_suaves))
    
    # FASE 5: Detección de change-points
    change_points = detectar_change_points(secuencias, estados_suaves_por_secuencia)
    
    # FASE 5b: Incorporar cambios de longitud como eventos
    dict_categorias_extendido = incorporar_cambios_longitud(
        cambios_longitud, dict_categorias)
    
    # FASE 6: Correlación con Path Analysis (usando dict extendido)
    resultados_correlacion = correlacionar_con_categorias(
        change_points, dict_categorias_extendido)
    
    # FASE 7: Análisis de residuos
    residuos = analizar_residuos(resultados_correlacion, serie)
    
    # FASE 8: Exportación
    exportar_resultados(
        serie, secuencias, estados_por_secuencia,
        estados_suaves_por_secuencia, change_points,
        resultados_correlacion, residuos, args.output_dir,
        measurement_id, probe_id,
        longitudes=longitudes,
        cambios_longitud=cambios_longitud,
        mapa_macro=mapa_macro,
        usar_macro_estados=args.macro_estados
    )
    
    print("\n" + "=" * 70)
    print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("Resultados disponibles en: {}".format(args.output_dir))


if __name__ == '__main__':
    main()
