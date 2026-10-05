# -*- coding: utf-8 -*-
"""
HMM con seleccion de K via BIC para Segmentacion de Latencia en RIPE Atlas
Adaptacion metodologica: HMM finito con seleccion de K via BIC
Equivalente estadistico al HDP-HMM para seleccion del numero de estados
"""
from __future__ import print_function
import os
import sys
import argparse
import csv
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
import numpy as np

# Configuracion
K_MIN = 2
K_MAX = 10
N_INIT = 5
TOLERANCIA_MINUTOS = 15


def cargar_csv_crudo(path_csv):
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
            except (ValueError, KeyError):
                continue
    
    timestamps = sorted(data_por_timestamp.keys())
    rtts = []
    dict_hops = OrderedDict()
    
    for ts in timestamps:
        hops = sorted(data_por_timestamp[ts], key=lambda x: x[0])
        dict_hops[ts] = hops
        rtts.append(hops[-1][3])
    
    rtts = np.array(rtts, dtype=np.float64)
    print("  - Lineas procesadas: {}".format(n_lineas))
    print("  - Puntos de datos end-to-end: {}".format(len(rtts)))
    print("  - RTT stats: min={:.2f}ms, max={:.2f}ms, mean={:.2f}ms, std={:.2f}ms".format(
        rtts.min(), rtts.max(), rtts.mean(), rtts.std()))
    
    return timestamps, rtts, dict_hops


def cargar_categorias(path_csv):
    print("[FASE 1] Cargando categorias: {}".format(path_csv))
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


def seleccionar_k_optimo(X):
    """Selecciona K optimo usando BIC (Bayesian Information Criterion)"""
    from hmmlearn.hmm import GaussianHMM
    
    print("\n[FASE 2] Seleccionando K optimo via BIC...")
    print("  - Rango: K={} a K={}".format(K_MIN, K_MAX))
    print("  - Inicializaciones por K: {}".format(N_INIT))
    
    mejor_bic = np.inf
    mejor_k = K_MIN
    mejor_modelo = None
    
    for k in range(K_MIN, K_MAX + 1):
        bic_scores = []
        modelos_validos = []
        
        for init in range(N_INIT):
            try:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type='full',
                    n_iter=100,
                    random_state=init,
                    tol=1e-4,
                    verbose=False
                )
                model.fit(X)
                log_likelihood = model.score(X)
                
                # Calcular BIC
                # n_params = transiciones (k*k) + medias (k*d) + covarianzas (k*d*(d+1)/2)
                d = X.shape[1]
                n_params = k * k + k * d + k * d * (d + 1) / 2
                n_samples = len(X)
                bic = -2 * log_likelihood * n_samples + 2 * n_params
                
                bic_scores.append(bic)
                modelos_validos.append(model)
            except Exception as e:
                continue
        
        if bic_scores:
            bic_promedio = np.mean(bic_scores)
            bic_std = np.std(bic_scores)
            print("  - K={}: BIC promedio = {:.2f} +/- {:.2f}".format(
                k, bic_promedio, bic_std))
            
            if bic_promedio < mejor_bic:
                mejor_bic = bic_promedio
                mejor_k = k
                mejor_modelo = modelos_validos[bic_scores.index(min(bic_scores))]
    
    print("\n[FASE 2] K optimo seleccionado: K={}".format(mejor_k))
    print("  - BIC: {:.2f}".format(mejor_bic))
    
    return mejor_modelo, mejor_k


def entrenar_hmm_final(X, k_optimo):
    """Entrena HMM final con K optimo usando multiples inicializaciones"""
    from hmmlearn.hmm import GaussianHMM
    
    print("\n[FASE 2] Entrenando HMM final con K={}...".format(k_optimo))
    
    mejor_modelo = None
    mejor_score = -np.inf
    
    for init in range(N_INIT * 2):
        try:
            model = GaussianHMM(
                n_components=k_optimo,
                covariance_type='full',
                n_iter=200,
                random_state=init,
                tol=1e-5,
                verbose=False
            )
            model.fit(X)
            score = model.score(X)
            
            if score > mejor_score:
                mejor_score = score
                mejor_modelo = model
        except Exception:
            continue
    
    if mejor_modelo is None:
        return None, None
    
    # Extraer secuencia de estados
    secuencia_estados = mejor_modelo.predict(X)
    
    print("  - Log-likelihood final: {:.2f}".format(mejor_score))
    print("  - Estados inferidos: {}".format(np.unique(secuencia_estados)))
    
    # Estadisticas por estado
    for est in np.unique(secuencia_estados):
        mask = (secuencia_estados == est)
        rtt_mean = X[mask].mean()
        rtt_std = X[mask].std()
        count = mask.sum()
        pct = 100.0 * count / len(secuencia_estados)
        print("      Estado {}: {} puntos ({:.1f}%) | RTT={:.2f}+/-{:.2f}ms".format(
            est, count, pct, rtt_mean, rtt_std))
    
    return mejor_modelo, secuencia_estados


def detectar_change_points(secuencia_estados):
    print("\n[FASE 3] Detectando change-points...")
    change_points = []
    
    for i in range(1, len(secuencia_estados)):
        if secuencia_estados[i] != secuencia_estados[i - 1]:
            change_points.append(i)
    
    print("  - Change-points detectados: {}".format(len(change_points)))
    if len(change_points) > 0:
        print("  - Frecuencia: 1 cambio cada {:.1f} puntos".format(
            float(len(secuencia_estados)) / len(change_points)))
    
    return change_points


def correlacionar_con_categorias(timestamps, change_points, dict_categorias):
    print("\n[FASE 4] Correlacionando con Path Analysis (tolerancia: {} min)...".format(
        TOLERANCIA_MINUTOS))
    
    ts_dt = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
    cat_dt = {}
    for ts_str in dict_categorias.keys():
        try:
            cat_dt[ts_str] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    
    resultados = []
    n_coincidencias = 0
    
    for idx in change_points:
        ts_cp = ts_dt[idx]
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
            'indice': idx,
            'timestamp': timestamps[idx],
            'categoria_path_analysis': cat_dominante,
            'categorias_en_ventana': dict(categorias_ventana),
            'coincide': coincide
        })
    
    print("  - Change-points con evento correspondiente: {}/{} ({:.1f}%)".format(
        n_coincidencias, len(change_points),
        100.0 * n_coincidencias / max(len(change_points), 1)))
    
    return resultados


def analizar_residuos(resultados_correlacion, dict_hops, timestamps, rtts):
    print("\n[FASE 5] Analizando residuos (contribucion original)...")
    residuos = []
    
    for res in resultados_correlacion:
        if not res['coincide']:
            idx = res['indice']
            ts = res['timestamp']
            
            if idx > 0 and idx < len(rtts):
                rtt_antes = rtts[idx - 1]
                rtt_despues = rtts[idx]
                delta_rtt = rtt_despues - rtt_antes
                pct_cambio = 100.0 * delta_rtt / max(rtt_antes, 0.001)
            else:
                continue
            
            hops_antes = dict_hops.get(timestamps[idx - 1], []) if idx > 0 else []
            hops_despues = dict_hops.get(ts, [])
            
            cambio_ip = (tuple(h[1] for h in hops_antes) != 
                        tuple(h[1] for h in hops_despues))
            cambio_asn = (tuple(h[2] for h in hops_antes) != 
                         tuple(h[2] for h in hops_despues))
            
            residuos.append({
                'indice': idx,
                'timestamp': ts,
                'rtt_antes_ms': rtt_antes,
                'rtt_despues_ms': rtt_despues,
                'delta_rtt_ms': delta_rtt,
                'pct_cambio': pct_cambio,
                'cambio_asn_detectado': cambio_asn,
                'cambio_ip_detectado': cambio_ip
            })
    
    print("  - Residuos identificados: {}".format(len(residuos)))
    n_topo = sum(1 for r in residuos if r['cambio_ip_detectado'])
    print("  - Con cambio topologico sutil: {}".format(n_topo))
    print("  - Puros de latencia (descubrimiento): {}".format(len(residuos) - n_topo))
    
    return residuos


def exportar_resultados(timestamps, rtts, secuencia_estados, change_points,
                        resultados_correlacion, residuos, output_dir):
    print("\n[FASE 6] Exportando resultados a: {}".format(output_dir))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Segmentacion
    path_seg = os.path.join(output_dir, 'segmentacion.csv')
    with open(path_seg, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'rtt_ms', 'estado', 'es_change_point'])
        cp_set = set(change_points)
        for i in range(len(timestamps)):
            writer.writerow([timestamps[i], "{:.3f}".format(rtts[i]),
                           int(secuencia_estados[i]), i in cp_set])
    print("  - Segmentacion: {}".format(path_seg))
    
    # Correlacion
    path_corr = os.path.join(output_dir, 'correlacion.csv')
    with open(path_corr, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['indice', 'timestamp', 'categoria', 'coincide'])
        for res in resultados_correlacion:
            writer.writerow([res['indice'], res['timestamp'],
                           res['categoria_path_analysis'] or 'NINGUNA',
                           res['coincide']])
    print("  - Correlacion: {}".format(path_corr))
    
    # Residuos
    path_res = os.path.join(output_dir, 'residuos.csv')
    with open(path_res, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['indice', 'timestamp', 'delta_rtt', 'pct_cambio',
                        'cambio_ip', 'cambio_asn'])
        for r in residuos:
            writer.writerow([r['indice'], r['timestamp'],
                           "{:.3f}".format(r['delta_rtt_ms']),
                           "{:.2f}".format(r['pct_cambio']),
                           r['cambio_ip_detectado'], r['cambio_asn_detectado']])
    print("  - Residuos: {}".format(path_res))
    
    # Resumen
    path_resumen = os.path.join(output_dir, 'resumen.txt')
    with open(path_resumen, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RESUMEN DE EJECUCION HMM + BIC\n")
        f.write("=" * 70 + "\n\n")
        f.write("Fecha: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Dataset: {} puntos temporales\n".format(len(timestamps)))
        f.write("Rango temporal: {} a {}\n".format(timestamps[0], timestamps[-1]))
        f.write("\n--- ESTADISTICAS DE RTT ---\n")
        f.write("Min: {:.2f} ms\n".format(rtts.min()))
        f.write("Max: {:.2f} ms\n".format(rtts.max()))
        f.write("Media: {:.2f} ms\n".format(rtts.mean()))
        f.write("Desv. Est.: {:.2f} ms\n".format(rtts.std()))
        f.write("\n--- MODELO HMM ---\n")
        f.write("K optimo (seleccionado via BIC): {}\n".format(len(np.unique(secuencia_estados))))
        f.write("Change-points detectados: {}\n".format(len(change_points)))
        f.write("\n--- CORRELACION CON PATH ANALYSIS ---\n")
        n_coinc = sum(1 for r in resultados_correlacion if r['coincide'])
        f.write("Coincidencias: {}/{} ({:.1f}%)\n".format(
            n_coinc, len(resultados_correlacion),
            100.0 * n_coinc / max(len(resultados_correlacion), 1)))
        f.write("\n--- RESIDUOS (CONTRIBUCION ORIGINAL) ---\n")
        f.write("Total residuos: {}\n".format(len(residuos)))
        n_topo = sum(1 for r in residuos if r['cambio_ip_detectado'])
        f.write("  Con cambio topologico sutil: {}\n".format(n_topo))
        f.write("  Puros de latencia (descubrimiento): {}\n".format(len(residuos) - n_topo))
        f.write("\n" + "=" * 70 + "\n")
    print("  - Resumen: {}".format(path_resumen))


def main():
    parser = argparse.ArgumentParser(
        description="HMM + BIC para segmentacion de latencia en traceroutes de RIPE Atlas")
    parser.add_argument('--crudo', required=True, help='Ruta al CSV crudo de traceroute')
    parser.add_argument('--categorias', required=True, help='Ruta al CSV de categorias')
    parser.add_argument('--output-dir', default='resultados_hmm', help='Directorio de salida')
    args = parser.parse_args()
    
    print("=" * 70)
    print("HMM + BIC para Segmentacion de Latencia en RIPE Atlas")
    print("=" * 70)
    print("Python: {}".format(sys.version))
    print("NumPy: {}".format(np.__version__))
    print("=" * 70)
    
    # FASE 1: Preprocesamiento
    timestamps, rtts, dict_hops = cargar_csv_crudo(args.crudo)
    dict_categorias = cargar_categorias(args.categorias)
    
    if len(timestamps) < 50:
        print("[ERROR] Dataset muy pequeno ({} puntos). Se requieren al menos 50.".format(
            len(timestamps)))
        sys.exit(1)
    
    # Preparar datos
    X = rtts.reshape(-1, 1)
    
    # FASE 2: Seleccion de K y entrenamiento
    modelo_inicial, k_optimo = seleccionar_k_optimo(X)
    if modelo_inicial is None:
        print("[ERROR] No se pudo seleccionar K optimo")
        sys.exit(1)
    
    modelo_final, secuencia_estados = entrenar_hmm_final(X, k_optimo)
    if secuencia_estados is None:
        print("[ERROR] No se pudo entrenar el modelo final")
        sys.exit(1)
    
    # FASE 3: Deteccion de change-points
    change_points = detectar_change_points(secuencia_estados)
    
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
    
    print("\n" + "=" * 70)
    print("EJECUCION COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("Resultados disponibles en: {}".format(args.output_dir))


if __name__ == '__main__':
    main()
