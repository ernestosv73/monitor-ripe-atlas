# -*- coding: utf-8 -*-
"""
=============================================================================
HDP-HMM para Segmentacion de Latencia en Traceroutes de RIPE Atlas
=============================================================================
Proyecto de Investigacion: Modelado Generativo de Latencia en Enrutamiento BGP
Basado en: Mouchet et al., "Large-Scale Characterization and Segmentation of
           Internet Path Delays with Infinite HMMs" (arXiv:1910.12714)

Entorno: Python 2.7 + bnpy + numpy 1.16.6 + scipy 1.2.1 + Intel MKL

OBJETIVO:
    1. Extraer serie temporal end-to-end de RTT desde CSV crudo de traceroute
    2. Entrenar HDP-HMM para segmentar la serie en estados ocultos
    3. Detectar change-points (transiciones entre estados)
    4. Correlacionar con categorias del Path Analysis de RIPE Atlas
    5. Identificar residuos (cambios detectados por HDP-HMM no explicados
       por el Path Analysis) -> contribucion original

USO:
    python hdphmm_ripe_atlas.py --crudo traceroute_crudo.csv \
                                --categorias categorias.csv \
                                --output-dir resultados/
=============================================================================
"""
from __future__ import print_function
import os
import sys
import argparse
import csv
import warnings
import tempfile
from collections import defaultdict, OrderedDict
from datetime import datetime

import numpy as np

# ============================================================================
# CONFIGURACION GLOBAL
# ============================================================================
HDP_GAMMA = 1.0
HDP_ALPHA = 1.0
HDP_K_INIT = 10
HDP_N_ITERS = 100
HDP_LAG = 1
HDP_SEED = 42
TOLERANCIA_MINUTOS = 15


# ============================================================================
# FASE 1: PREPROCESAMIENTO DE DATOS
# ============================================================================
def cargar_csv_crudo(path_csv):
    """
    Carga el CSV de traceroute en formato largo y extrae el RTT end-to-end
    por timestamp (Opcion A del plan metodologico).
    """
    print("[FASE 1] Cargando CSV crudo: {}".format(path_csv))

    data_por_timestamp = defaultdict(list)

    with open(path_csv, 'r') as f:
        reader = csv.DictReader(f)
        n_lineas = 0
        for row in reader:
            n_lineas += 1
            ts = row['timestamp'].strip()
            try:
                hop = int(row['hop'])
                ip = row['ip'].strip()
                asn = row['asn'].strip()
                rtt = float(row['rtt_ms'])
                data_por_timestamp[ts].append((hop, ip, asn, rtt))
            except (ValueError, KeyError) as e:
                warnings.warn("Fila mal formada ignorada: {} - Error: {}".format(
                    row, e))
                continue

    print("  - Lineas procesadas: {}".format(n_lineas))
    print("  - Timestamps unicos: {}".format(len(data_por_timestamp)))

    timestamps = sorted(data_por_timestamp.keys())

    rtts = []
    dict_hops = OrderedDict()
    timestamps_validos = []

    for ts in timestamps:
        hops = data_por_timestamp[ts]
        hops_sorted = sorted(hops, key=lambda x: x[0])
        dict_hops[ts] = hops_sorted
        hop_destino = hops_sorted[-1]
        rtts.append(hop_destino[3])
        timestamps_validos.append(ts)

    rtts = np.array(rtts, dtype=np.float64)

    print("  - Puntos de datos end-to-end: {}".format(len(rtts)))
    print("  - RTT stats: min={:.2f}ms, max={:.2f}ms, "
          "mean={:.2f}ms, std={:.2f}ms".format(
              rtts.min(), rtts.max(), rtts.mean(), rtts.std()))

    return timestamps_validos, rtts, dict_hops


def cargar_categorias(path_csv):
    """
    Carga el CSV de categorias del Path Analysis.
    Cada fila representa un hop especifico que cambio.
    """
    print("[FASE 1] Cargando CSV de categorias: {}".format(path_csv))

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

    print("  - Lineas procesadas: {}".format(n_lineas))
    print("  - Timestamps con eventos: {}".format(len(dict_categorias)))

    todas_cats = defaultdict(int)
    for info in dict_categorias.values():
        for c in info['categorias']:
            todas_cats[c] += 1
    print("  - Distribucion de categorias:")
    for cat, count in sorted(todas_cats.items(), key=lambda x: -x[1]):
        print("      {}: {} timestamps".format(cat, count))

    return dict(dict_categorias)


# ============================================================================
# FASE 2: ENTRENAMIENTO DEL HDP-HMM
# ============================================================================
def preparar_datos_bnpy(rtts):
    """
    Prepara la serie de RTTs para bnpy.
    bnpy espera datos en formato 2D: (n_samples, n_features).
    """
    mean_val = rtts.mean()
    std_val = rtts.std()
    if std_val > 0:
        skewness = np.mean(((rtts - mean_val) / std_val) ** 3)
    else:
        skewness = 0.0

    print("[FASE 2] Preparando datos para bnpy...")
    print("  - Skewness de RTT: {:.3f}".format(skewness))

    if abs(skewness) > 2.0:
        print("  - Advertencia: alta asimetria. "
              "Considere transformacion log.")
        print("  - Continuando con valores originales "
              "para interpretabilidad.")

    X = rtts.reshape(-1, 1).astype(np.float64)
    return X


def configurar_entorno_bnpy(output_subdir):
    """
    Configura las variables de entorno requeridas por bnpy.
    bnpy exige BNPYOUTDIR para guardar resultados temporales.
    """
    if output_subdir is None:
        output_subdir = os.path.join(tempfile.gettempdir(), 'bnpy_results')

    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    os.environ['BNPYOUTDIR'] = output_subdir
    os.environ['BNPYDATADIR'] = output_subdir

    print("  - BNPYOUTDIR configurado: {}".format(output_subdir))
    return output_subdir


def entrenar_hdphmm(X, output_subdir=None):
    """
    Entrena un HDP-HMM usando bnpy.
    Intenta Gibbs Sampling primero, cae a memoVB si falla.
    """
    print("[FASE 2] Entrenando HDP-HMM con bnpy...")
    print("  - Parametros: gamma={}, alpha={}, K_init={}, "
          "n_iters={}".format(
              HDP_GAMMA, HDP_ALPHA, HDP_K_INIT, HDP_N_ITERS))

    # Configurar entorno antes de importar bnpy
    configurar_entorno_bnpy(output_subdir)

    # Importar bnpy
    try:
        import bnpy
        print("  - bnpy importado exitosamente")
    except ImportError as e:
        print("  [ERROR] bnpy no esta disponible: {}".format(e))
        return None, None, None
    except ValueError as e:
        print("  [ERROR] bnpy rechazo la configuracion: {}".format(e))
        return None, None, None

    # Crear dataset bnpy
    try:
        dataset = bnpy.data.XData(X=X)
        print("  - Dataset bnpy creado: {} muestras, "
              "{} dimensiones".format(X.shape[0], X.shape[1]))
    except Exception as e:
        print("  [ERROR] Creando dataset bnpy: {}".format(e))
        return None, None, None

    # Entrenar modelo usando bnpy.run() (API de alto nivel)
    trained_model = None
    info_dict = None

    # Intento 1: Gibbs Sampling (metodo del paper)
    print("  - Intento 1: Gibbs Sampling...")
    try:
        trained_model, info_dict = bnpy.run(
            dataset,
            'HDPHMM', 'Gauss', 'GibbsSampler',
            nLap=HDP_N_ITERS,
            gamma=HDP_GAMMA,
            alpha=HDP_ALPHA,
            K=HDP_K_INIT,
            sF=1.0,
            ECovMat='full',
            nBatch=1,
            seed=HDP_SEED
        )
        print("  - Gibbs Sampling completado exitosamente")
    except Exception as e:
        print("  - Gibbs Sampling fallo: {}".format(e))
        print("  - Intento 2: Variational Bayes (memoVB)...")

        # Intento 2: memoVB (mas estable)
        try:
            trained_model, info_dict = bnpy.run(
                dataset,
                'HDPHMM', 'Gauss', 'memoVB',
                nLap=HDP_N_ITERS,
                gamma=HDP_GAMMA,
                alpha=HDP_ALPHA,
                K=HDP_K_INIT,
                sF=1.0,
                ECovMat='full',
                seed=HDP_SEED
            )
            print("  - memoVB completado exitosamente")
        except Exception as e2:
            print("  [ERROR] memoVB tambien fallo: {}".format(e2))
            print("  - Intento 3: DP mixture (sin HMM, fallback)...")

            # Intento 3: DP mixture model (sin dependencia temporal)
            try:
                trained_model, info_dict = bnpy.run(
                    dataset,
                    'DPMixture', 'Gauss', 'memoVB',
                    nLap=HDP_N_ITERS,
                    gamma=HDP_GAMMA,
                    K=HDP_K_INIT,
                    sF=1.0,
                    ECovMat='full',
                    seed=HDP_SEED
                )
                print("  - DP Mixture completado (nota: sin "
                      "dependencia temporal)")
            except Exception as e3:
                print("  [ERROR] Todos los metodos fallaron: "
                      "{}".format(e3))
                return None, None, None

    # Extraer secuencia de estados inferidos
    secuencia_estados = None

    # Metodo 1: calc_local_params -> resp -> argmax
    try:
        LP = trained_model.calc_local_params(dataset)
        resp = LP['resp']
        secuencia_estados = np.argmax(resp, axis=1)
        print("  - Estados extraidos via calc_local_params")
    except Exception as e:
        print("  - Metodo 1 fallo: {}".format(e))

    # Metodo 2: atributo directo assignments
    if secuencia_estados is None:
        try:
            secuencia_estados = np.array(trained_model.assignments)
            print("  - Estados extraidos via .assignments")
        except Exception as e:
            print("  - Metodo 2 fallo: {}".format(e))

    # Metodo 3: desde info_dict
    if secuencia_estados is None:
        try:
            secuencia_estados = np.array(info_dict['assignments'])
            print("  - Estados extraidos via info_dict")
        except Exception as e:
            print("  - Metodo 3 fallo: {}".format(e))

    if secuencia_estados is None:
        print("  [ERROR] No se pudo extraer la secuencia de estados")
        return trained_model, info_dict, None

    # Estadisticas del modelo
    estados_unicos = np.unique(secuencia_estados)
    print("  - Entrenamiento completado exitosamente")
    print("  - Estados inferidos: {} (de K_init={})".format(
        len(estados_unicos), HDP_K_INIT))
    print("  - Distribucion de estados:")
    for est in estados_unicos:
        count = int(np.sum(secuencia_estados == est))
        pct = 100.0 * count / len(secuencia_estados)
        # Calcular media de RTT para este estado
        mask = (secuencia_estados == est)
        rtt_mean = X[mask].mean()
        rtt_std = X[mask].std()
        print("      Estado {}: {} puntos ({:.1f}%) | "
              "RTT mean={:.2f}ms std={:.2f}ms".format(
                  est, count, pct, rtt_mean, rtt_std))

    return trained_model, info_dict, secuencia_estados


# ============================================================================
# FASE 3: DETECCION DE CHANGE-POINTS
# ============================================================================
def detectar_change_points(secuencia_estados):
    """
    Identifica los indices donde ocurre una transicion entre estados.
    """
    print("[FASE 3] Detectando change-points...")

    change_points = []
    detalles = []

    for i in range(1, len(secuencia_estados)):
        if secuencia_estados[i] != secuencia_estados[i - 1]:
            change_points.append(i)
            detalles.append((
                i,
                int(secuencia_estados[i - 1]),
                int(secuencia_estados[i])
            ))

    print("  - Change-points detectados: {}".format(len(change_points)))
    if len(change_points) > 0:
        print("  - Frecuencia: 1 cambio cada {:.1f} puntos".format(
            float(len(secuencia_estados)) / len(change_points)))
    else:
        print("  - No se detectaron change-points")

    return change_points, detalles


# ============================================================================
# FASE 4: CORRELACION CON PATH ANALYSIS
# ============================================================================
def correlacionar_con_categorias(timestamps, change_points,
                                  dict_categorias,
                                  tolerancia_min=TOLERANCIA_MINUTOS):
    """
    Cruza los change-points detectados por HDP-HMM con los eventos del
    Path Analysis, usando una ventana de tolerancia temporal.
    """
    print("[FASE 4] Correlacionando con Path Analysis "
          "(tolerancia: {} min)...".format(tolerancia_min))

    ts_dt = []
    for ts in timestamps:
        ts_dt.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))

    cat_dt = {}
    for ts_str in dict_categorias.keys():
        try:
            cat_dt[ts_str] = datetime.strptime(
                ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    resultados = []
    n_coincidencias = 0

    for idx in change_points:
        ts_cp = ts_dt[idx]
        categorias_ventana = defaultdict(int)

        for ts_str, dt_cat in cat_dt.items():
            diff_sec = (ts_cp - dt_cat).total_seconds()
            diff_min = abs(diff_sec) / 60.0
            if diff_min <= tolerancia_min:
                for cat in dict_categorias[ts_str]['categorias']:
                    categorias_ventana[cat] += 1

        if categorias_ventana:
            cat_dominante = max(
                categorias_ventana.items(),
                key=lambda x: x[1]
            )[0]
            coincide = True
            n_coincidencias += 1
        else:
            cat_dominante = None
            coincide = False

        resultados.append({
            'indice': idx,
            'timestamp': timestamps[idx],
            'categoria_path_analysis': cat_dominante,
            'categorias_en_ventana': dict(categorias_ventana),
            'coincide': coincide
        })

    total_cp = len(change_points)
    if total_cp > 0:
        pct = 100.0 * n_coincidencias / total_cp
    else:
        pct = 0.0
    print("  - Change-points con evento correspondiente: "
          "{}/{} ({:.1f}%)".format(
              n_coincidencias, total_cp, pct))

    return resultados


# ============================================================================
# FASE 5: ANALISIS DE RESIDUOS (CONTRIBUCION ORIGINAL)
# ============================================================================
def analizar_residuos(resultados_correlacion, dict_hops,
                       timestamps, rtts):
    """
    Identifica change-points del HDP-HMM que NO tienen evento
    correspondiente en el Path Analysis.
    """
    print("[FASE 5] Analizando residuos (contribucion original)...")

    residuos = []

    for res in resultados_correlacion:
        if not res['coincide']:
            idx = res['indice']
            ts = res['timestamp']

            if idx > 0 and idx < len(rtts):
                rtt_antes = rtts[idx - 1]
                rtt_despues = rtts[idx]
                delta_rtt = rtt_despues - rtt_antes
                if rtt_antes > 0.001:
                    pct_cambio = 100.0 * delta_rtt / rtt_antes
                else:
                    pct_cambio = 0.0
            else:
                rtt_antes = None
                rtt_despues = rtts[idx] if idx < len(rtts) else 0
                delta_rtt = 0.0
                pct_cambio = 0.0

            hops_antes = []
            if idx > 0:
                ts_antes = timestamps[idx - 1]
                hops_antes = dict_hops.get(ts_antes, [])
            hops_despues = dict_hops.get(ts, [])

            asn_antes = tuple(h[2] for h in hops_antes)
            asn_despues = tuple(h[2] for h in hops_despues)
            cambio_asn = (asn_antes != asn_despues)

            ip_antes = tuple(h[1] for h in hops_antes)
            ip_despues = tuple(h[1] for h in hops_despues)
            cambio_ip = (ip_antes != ip_despues)

            residuos.append({
                'indice': idx,
                'timestamp': ts,
                'rtt_antes_ms': rtt_antes,
                'rtt_despues_ms': rtt_despues,
                'delta_rtt_ms': delta_rtt,
                'pct_cambio': pct_cambio,
                'cambio_asn_detectado': cambio_asn,
                'cambio_ip_detectado': cambio_ip,
                'longitud_ruta_antes': len(hops_antes),
                'longitud_ruta_despues': len(hops_despues)
            })

    print("  - Residuos identificados: {}".format(len(residuos)))

    n_con_cambio = sum(
        1 for r in residuos if r['cambio_ip_detectado'])
    n_puros_rtt = sum(
        1 for r in residuos if not r['cambio_ip_detectado'])

    print("  - Residuos con cambio topologico sutil "
          "(ECMP/IP): {}".format(n_con_cambio))
    print("  - Residuos puros de latencia "
          "(sin cambio topologico): {}".format(n_puros_rtt))
    print("  - ESTOS ULTIMOS SON LA CONTRIBUCION "
          "ORIGINAL DEL MODELO")

    return residuos


# ============================================================================
# FASE 6: EXPORTACION DE RESULTADOS
# ============================================================================
def exportar_resultados(timestamps, rtts, secuencia_estados,
                         change_points, resultados_correlacion,
                         residuos, output_dir):
    """
    Exporta todos los resultados a archivos CSV y un resumen.
    """
    print("[FASE 6] Exportando resultados a: {}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Segmentacion completa
    path_seg = os.path.join(output_dir, 'segmentacion_hdphmm.csv')
    with open(path_seg, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'rtt_ms', 'estado_inferido',
            'es_change_point'
        ])
        cp_set = set(change_points)
        for i in range(len(timestamps)):
            ts = timestamps[i]
            rtt = rtts[i]
            est = int(secuencia_estados[i])
            es_cp = (i in cp_set)
            writer.writerow([
                ts, "{:.3f}".format(rtt), est, es_cp
            ])
    print("  - Segmentacion: {}".format(path_seg))

    # 2. Change-points con correlacion
    path_cp = os.path.join(
        output_dir, 'change_points_correlacion.csv')
    with open(path_cp, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'indice', 'timestamp', 'categoria_path_analysis',
            'categorias_en_ventana', 'coincide'
        ])
        for res in resultados_correlacion:
            cats_parts = []
            for k, v in res['categorias_en_ventana'].items():
                cats_parts.append("{}:{}".format(k, v))
            cats_str = "|".join(cats_parts)
            cat_pa = res['categoria_path_analysis']
            if cat_pa is None:
                cat_pa = 'NINGUNA'
            writer.writerow([
                res['indice'], res['timestamp'],
                cat_pa, cats_str, res['coincide']
            ])
    print("  - Correlacion: {}".format(path_cp))

    # 3. Residuos (contribucion original)
    path_res = os.path.join(
        output_dir, 'residuos_contribucion_original.csv')
    with open(path_res, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'indice', 'timestamp', 'rtt_antes_ms',
            'rtt_despues_ms', 'delta_rtt_ms', 'pct_cambio',
            'cambio_asn', 'cambio_ip',
            'longitud_antes', 'longitud_despues'
        ])
        for r in residuos:
            rtt_a = ''
            if r['rtt_antes_ms'] is not None:
                rtt_a = "{:.3f}".format(r['rtt_antes_ms'])
            writer.writerow([
                r['indice'], r['timestamp'],
                rtt_a,
                "{:.3f}".format(r['rtt_despues_ms']),
                "{:.3f}".format(r['delta_rtt_ms']),
                "{:.2f}".format(r['pct_cambio']),
                r['cambio_asn_detectado'],
                r['cambio_ip_detectado'],
                r['longitud_ruta_antes'],
                r['longitud_ruta_despues']
            ])
    print("  - Residuos: {}".format(path_res))

    # 4. Resumen estadistico
    path_resumen = os.path.join(
        output_dir, 'resumen_ejecucion.txt')
    with open(path_resumen, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RESUMEN DE EJECUCION HDP-HMM\n")
        f.write("=" * 70 + "\n\n")
        f.write("Fecha: {}\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Dataset: {} puntos temporales\n".format(
            len(timestamps)))
        f.write("Rango temporal: {} a {}\n".format(
            timestamps[0], timestamps[-1]))
        f.write("\n--- ESTADISTICAS DE RTT ---\n")
        f.write("Min: {:.2f} ms\n".format(rtts.min()))
        f.write("Max: {:.2f} ms\n".format(rtts.max()))
        f.write("Media: {:.2f} ms\n".format(rtts.mean()))
        f.write("Desv. Est.: {:.2f} ms\n".format(rtts.std()))
        f.write("\n--- MODELO HDP-HMM ---\n")
        f.write("Parametros: gamma={}, alpha={}, "
                "K_init={}, n_iters={}\n".format(
                    HDP_GAMMA, HDP_ALPHA,
                    HDP_K_INIT, HDP_N_ITERS))
        estados_unicos = np.unique(secuencia_estados)
        f.write("Estados inferidos: {}\n".format(
            len(estados_unicos)))
        f.write("Change-points detectados: {}\n".format(
            len(change_points)))

        f.write("\n--- CORRELACION CON PATH ANALYSIS ---\n")
        n_coinc = sum(
            1 for r in resultados_correlacion if r['coincide'])
        total_cp = len(resultados_correlacion)
        if total_cp > 0:
            pct_coinc = 100.0 * n_coinc / total_cp
        else:
            pct_coinc = 0.0
        f.write("Coincidencias: {}/{} ({:.1f}%)\n".format(
            n_coinc, total_cp, pct_coinc))

        cat_counts = defaultdict(int)
        for r in resultados_correlacion:
            if r['categoria_path_analysis'] is not None:
                cat_counts[r['categoria_path_analysis']] += 1
        f.write("\nDistribucion de categorias en "
                "change-points:\n")
        for cat, count in sorted(
                cat_counts.items(), key=lambda x: -x[1]):
            f.write("  {}: {}\n".format(cat, count))

        f.write("\n--- RESIDUOS (CONTRIBUCION ORIGINAL) ---\n")
        f.write("Total residuos: {}\n".format(len(residuos)))
        n_topo = sum(
            1 for r in residuos if r['cambio_ip_detectado'])
        n_puros = len(residuos) - n_topo
        f.write("  Con cambio topologico sutil: {}\n".format(
            n_topo))
        f.write("  Puros de latencia "
                "(descubrimiento): {}\n".format(n_puros))
        f.write("\n" + "=" * 70 + "\n")
    print("  - Resumen: {}".format(path_resumen))


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HDP-HMM para segmentacion de latencia "
                    "en traceroutes de RIPE Atlas")
    parser.add_argument(
        '--crudo', required=True,
        help='Ruta al CSV crudo de traceroute')
    parser.add_argument(
        '--categorias', required=True,
        help='Ruta al CSV de categorias del Path Analysis')
    parser.add_argument(
        '--output-dir', default='resultados_hdphmm',
        help='Directorio de salida (default: resultados_hdphmm)')
    args = parser.parse_args()

    print("=" * 70)
    print("HDP-HMM para Segmentacion de Latencia en RIPE Atlas")
    print("=" * 70)
    print("Python: {}".format(sys.version))
    print("NumPy: {}".format(np.__version__))
    print("=" * 70)

    # FASE 1: Preprocesamiento
    timestamps, rtts, dict_hops = cargar_csv_crudo(args.crudo)
    dict_categorias = cargar_categorias(args.categorias)

    if len(timestamps) < 50:
        print("[ERROR] Dataset muy pequeno ({} puntos). "
              "Se requieren al menos 50.".format(len(timestamps)))
        sys.exit(1)

    # FASE 2: Entrenamiento HDP-HMM
    X = preparar_datos_bnpy(rtts)
    bnpy_output_dir = os.path.join(args.output_dir, 'bnpy_temp')
    modelo, info, secuencia_estados = entrenar_hdphmm(
        X, output_subdir=bnpy_output_dir)

    if secuencia_estados is None:
        print("[ERROR] No se pudo entrenar el modelo. Abortando.")
        sys.exit(1)

    # FASE 3: Deteccion de change-points
    change_points, detalles_cp = detectar_change_points(
        secuencia_estados)

    # FASE 4: Correlacion con Path Analysis
    resultados_correlacion = correlacionar_con_categorias(
        timestamps, change_points, dict_categorias)

    # FASE 5: Analisis de residuos
    residuos = analizar_residuos(
        resultados_correlacion, dict_hops, timestamps, rtts)

    # FASE 6: Exportacion
    exportar_resultados(
        timestamps, rtts, secuencia_estados, change_points,
        resultados_correlacion, residuos, args.output_dir)

    print("")
    print("=" * 70)
    print("EJECUCION COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("Resultados disponibles en: {}".format(args.output_dir))


if __name__ == '__main__':
    main()
