#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mide cuanto "duran" los cambios detectados por extract_traceroute_v2.py:
agrupa los timestamps marcados en el ground truth (eventos_measurement*.csv)
en rachas de ciclos CONSECUTIVOS (segun el timeline completo del CSV crudo
historial_traceroute_measurement*.csv), y reporta la distribucion de
largos de racha.

Por que importa: el metodo de comparacion consecutiva (ciclo N vs N-1)
marca DOS ciclos por cada pico aislado de 1 ciclo -- el de entrada (RTT
sube respecto del anterior) y el de salida (RTT vuelve a bajar respecto
del pico). Eso da una racha de largo 2 por cada pico transitorio. Si la
gran mayoria de las rachas tienen largo 2, es evidencia de que el
"salto" de conteo entre Path Analysis/script de reglas (cuentan eventos
puntuales) y el HDP-HMM (cuenta regimenes sostenidos, y el suavizado por
--min-dwell borra cualquier cosa mas corta que ese umbral) se explica
por la naturaleza de los datos, no por un error de ninguno de los dos
metodos.

Uso:
    python3 medir_persistencia_eventos.py \
        --historial historial_traceroute_measurement126502326_probe64883.csv \
        --eventos eventos_measurement126502326_probe64883.csv
"""
from __future__ import print_function, division

import argparse
import csv
import sys
from collections import Counter


def cargar_timeline(historial_path):
    """Lista ordenada de timestamps UNICOS presentes en el CSV crudo (un timestamp por ciclo)."""
    vistos = set()
    with open(historial_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vistos.add(row["timestamp"])
    return sorted(vistos)


def cargar_timestamps_marcados(eventos_path):
    marcados = set()
    with open(eventos_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            marcados.add(row["timestamp"])
    return marcados


def agrupar_en_rachas(timeline, marcados):
    """
    timeline: lista ordenada de TODOS los timestamps de ciclo (marcados o no).
    marcados: set de los timestamps que el ground truth senalo con algun cambio.
    Devuelve una lista de largos de racha (cantidad de ciclos consecutivos
    marcados, segun la posicion en 'timeline' -- no segun diferencia de
    tiempo, para que un hueco de datos no una dos rachas que en realidad
    estan separadas).
    """
    largos = []
    racha_actual = 0
    for ts in timeline:
        if ts in marcados:
            racha_actual += 1
        else:
            if racha_actual > 0:
                largos.append(racha_actual)
            racha_actual = 0
    if racha_actual > 0:
        largos.append(racha_actual)
    return largos


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mide la distribucion de largos de racha de los cambios detectados por extract_traceroute_v2.py."
    )
    parser.add_argument("--historial", required=True, help="CSV crudo (historial_traceroute_*.csv)")
    parser.add_argument("--eventos", required=True, help="CSV de ground truth (eventos_measurement*.csv)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    timeline = cargar_timeline(args.historial)
    marcados = cargar_timestamps_marcados(args.eventos)

    if not timeline:
        sys.stderr.write("No se pudo leer ningun timestamp del CSV crudo.\n")
        sys.exit(1)

    largos = agrupar_en_rachas(timeline, marcados)
    if not largos:
        print("No se encontraron ciclos marcados en el ground truth que coincidan con el timeline crudo.")
        return

    conteo = Counter(largos)
    total_rachas = len(largos)
    total_ciclos_marcados = sum(largos)

    print("Ciclos totales en el timeline: {0}".format(len(timeline)))
    print("Ciclos marcados (con algun cambio): {0}".format(len(marcados)))
    print("Rachas de ciclos consecutivos marcados: {0}".format(total_rachas))
    print()
    print("Distribucion de largos de racha:")
    for largo in sorted(conteo.keys()):
        n = conteo[largo]
        pct_rachas = 100.0 * n / total_rachas
        pct_ciclos = 100.0 * (largo * n) / total_ciclos_marcados
        print("   largo={0:>3} ciclos: {1:>5} rachas ({2:5.1f}% de las rachas, "
              "{3:5.1f}% de los ciclos marcados)".format(largo, n, pct_rachas, pct_ciclos))

    pct_largo2 = 100.0 * conteo.get(2, 0) / total_rachas
    pct_largo1 = 100.0 * conteo.get(1, 0) / total_rachas
    pct_largo_ge3 = 100.0 * sum(n for l, n in conteo.items() if l >= 3) / total_rachas
    print()
    print("Resumen: {0:.1f}% de las rachas son de largo 2 (pico aislado de 1 ciclo, "
          "contado 2 veces: entrada+salida), {1:.1f}% de largo 1 (cambio que no revierte "
          "en el ciclo siguiente, o ultimo ciclo del rango), {2:.1f}% de largo >=3 "
          "(excursion sostenida por varios ciclos).".format(pct_largo2, pct_largo1, pct_largo_ge3))
    if pct_largo2 + pct_largo1 > 70:
        print("-> La gran mayoria de los 'cambios' son eventos puntuales de 1-2 ciclos, no "
              "regimenes sostenidos. Es coherente con que el HDP-HMM (que busca regimenes "
              "sostenidos, y con --min-dwell filtra todo lo mas corto que ese umbral) reporte "
              "un numero mucho menor -- no es una discrepancia a 'corregir', es una diferencia "
              "de que fenomeno mide cada metodo.")


if __name__ == "__main__":
    main()
