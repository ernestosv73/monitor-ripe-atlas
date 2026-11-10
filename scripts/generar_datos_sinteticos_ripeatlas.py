#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Generador de Datos Sintéticos para Validación de Detección de Puntos de Cambio
Metodología: Inyección Paramétrica Controlada sobre Base Real (RIPE Atlas)
=============================================================================
Entorno: Python 3.x + pandas + numpy + re

OBJETIVO:
Generar un dataset sintético que preserve las características estadísticas 
(jitter, outliers, autocorrelación) de mediciones reales de RIPE Atlas, 
pero con puntos de cambio (changepoints) inyectados artificialmente en 
ubicaciones y magnitudes conocidas (Ground Truth).
=============================================================================
"""
import os
import sys
import argparse
import csv
import random
import re
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.stderr.write("Este script requiere pandas y numpy. Instalar con: pip install pandas numpy\n")
    sys.exit(1)


def extraer_ids_de_archivo(filename):
    """Extrae measurement_id y probe_id del nombre del archivo."""
    basename = os.path.basename(filename)
    match = re.search(r'(?:measurement)?(\d+)_(?:probe)?(\d+)', basename)
    if match:
        return match.group(1), match.group(2)
    return "sintetico", "0000000"


def generar_datos_sinteticos(input_csv, output_dir, num_cambios, magnitud_ms, duracion_ciclos, semilla):
    print("=" * 70)
    print("GENERADOR DE DATOS SINTÉTICOS RIPE ATLAS")
    print("=" * 70)
    print(f"Entrada: {input_csv}")
    print(f"Cambios a inyectar: {num_cambios}")
    print(f"Magnitud del cambio: +{magnitud_ms} ms")
    print(f"Duración por cambio: {duracion_ciclos} ciclos")
    print(f"Semilla aleatoria: {semilla}")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1/5] Cargando datos reales...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el CSV: {e}")
        sys.exit(1)

    columnas_requeridas = ['timestamp', 'hop', 'rtt_ms']
    if not all(col in df.columns for col in columnas_requeridas):
        print(f"[ERROR] El CSV debe contener las columnas: {columnas_requeridas}")
        sys.exit(1)

    # Ordenar por tiempo para garantizar continuidad
    df = df.sort_values('timestamp').reset_index(drop=True)
    unique_times = sorted(df['timestamp'].unique())
    total_ciclos = len(unique_times)

    print(f"  - Ciclos únicos encontrados: {total_ciclos}")
    print(f"  - Filas totales (hops): {len(df)}")

    # 2. Validar viabilidad
    ciclos_minimos_requeridos = num_cambios * duracion_ciclos + (num_cambios - 1) * 5
    if total_ciclos < ciclos_minimos_requeridos:
        print(f"[ERROR] El dataset es demasiado pequeño. Se necesitan al menos {ciclos_minimos_requeridos} ciclos.")
        sys.exit(1)

    # 3. Seleccionar puntos de inyección (evitando solapamientos)
    print("\n[2/5] Seleccionando puntos de inyección aleatorios...")
    random.seed(semilla)
    np.random.seed(semilla)

    available_starts = list(range(total_ciclos - duracion_ciclos + 1))
    chosen_starts = []

    for _ in range(num_cambios):
        if not available_starts:
            print("  [ADVERTENCIA] No hay más espacio no solapado para inyectar cambios.")
            break
        
        start_idx = random.choice(available_starts)
        chosen_starts.append(start_idx)
        
        # Eliminar índices que se solaparían con este cambio (incluyendo un buffer de 5 ciclos)
        buffer = 5
        available_starts = [
            x for x in available_starts 
            if x < start_idx - duracion_ciclos - buffer or x > start_idx + duracion_ciclos + buffer
        ]

    chosen_starts.sort()
    print(f"  - Puntos de inicio seleccionados (índices de ciclo): {chosen_starts}")

    # 4. Inyectar cambios y registrar Ground Truth
    print("\n[3/5] Inyectando degradaciones y registrando Ground Truth...")
    ground_truth = []
    df_modificado = df.copy()

    for i, start_idx in enumerate(chosen_starts):
        end_idx = start_idx + duracion_ciclos - 1
        start_ts = unique_times[start_idx]
        end_ts = unique_times[end_idx]
        
        affected_times = unique_times[start_idx:end_idx+1]
        
        # Calcular la media original del último salto (destino) para estos ciclos
        orig_rtts = []
        indices_a_modificar = []
        
        for ts in affected_times:
            ts_df = df[df['timestamp'] == ts]
            max_hop = ts_df['hop'].max()
            # Obtener el índice de la fila del destino para modificarlo específicamente
            idx_destino = ts_df[ts_df['hop'] == max_hop].index[0]
            indices_a_modificar.append(idx_destino)
            
            rtt_val = ts_df.loc[idx_destino, 'rtt_ms']
            orig_rtts.append(rtt_val)
            
        orig_mean = float(np.mean(orig_rtts))
        new_mean = orig_mean + magnitud_ms
        
        # Aplicar la inyección: sumar la magnitud SOLO al RTT del salto de destino
        df_modificado.loc[indices_a_modificar, 'rtt_ms'] = df_modificado.loc[indices_a_modificar, 'rtt_ms'] + magnitud_ms
        
        ground_truth.append({
            'evento_id': i + 1,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'duration_cycles': duracion_ciclos,
            'original_mean_rtt_ms': round(orig_mean, 3),
            'injected_mean_rtt_ms': round(new_mean, 3),
            'magnitude_injected_ms': magnitud_ms
        })
        
        print(f"  - Evento {i+1}: {start_ts} a {end_ts} ({duracion_ciclos} ciclos) | Media destino: {orig_mean:.2f} -> {new_mean:.2f} ms")

    # 5. Guardar resultados
    print("\n[4/5] Guardando archivos de salida...")
    os.makedirs(output_dir, exist_ok=True)
    
    measurement_id, probe_id = extraer_ids_de_archivo(input_csv)
    suffix = f"_measurement{measurement_id}_probe{probe_id}_sintetico"
    
    output_csv = os.path.join(output_dir, f"historial_sintetico{suffix}.csv")
    gt_csv = os.path.join(output_dir, f"ground_truth_changepoints{suffix}.csv")
    
    df_modificado.to_csv(output_csv, index=False)
    print(f"  - Datos sintéticos guardados en: {output_csv}")
    
    df_gt = pd.DataFrame(ground_truth)
    df_gt.to_csv(gt_csv, index=False)
    print(f"  - Ground Truth guardado en: {gt_csv}")

    # 6. Resumen
    print("\n[5/5] Resumen de la generación:")
    print(f"  - Cambios inyectados exitosamente: {len(ground_truth)}")
    print("=" * 70)
    print("GENERACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    
    return output_csv, gt_csv


def main():
    parser = argparse.ArgumentParser(
        description="Generador de datos sintéticos para validación de HDP-HMM en RIPE Atlas"
    )
    parser.add_argument('--input-csv', required=True, help='CSV crudo real de traceroute (fuente de verdad)')
    parser.add_argument('--output-dir', default='datos_sinteticos', help='Directorio de salida')
    parser.add_argument('--num-cambios', type=int, default=3, help='Número de degradaciones a inyectar (default: 3)')
    parser.add_argument('--magnitud', type=float, default=5.0, help='Incremento de RTT en ms para la degradación (default: 5.0)')
    parser.add_argument('--duracion', type=int, default=20, help='Duración de cada degradación en número de ciclos (default: 20)')
    parser.add_argument('--semilla', type=int, default=42, help='Semilla para reproducibilidad aleatoria (default: 42)')
    
    args = parser.parse_args()
    
    generar_datos_sinteticos(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        num_cambios=args.num_cambios,
        magnitud_ms=args.magnitud,
        duracion_ciclos=args.duracion,
        semilla=args.semilla
    )

if __name__ == '__main__':
    main()
