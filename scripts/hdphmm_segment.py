#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentacion de series de RTT (RIPE Atlas) con un HDP-HMM, siguiendo el
enfoque de:

    Hughes, M.C., Sudderth, E.B. "Large-Scale Characterization and
    Segmentation of Internet Path Delays with Infinite HMMs."

usando la libreria bnpy (build py27 instalada por el usuario via
miniconda/Docker: dranew::bnpy-8019474-py27_1, numpy 1.16.6, scipy 1.2.1).

IMPORTANTE -- estado de validacion de este script:
    No tengo bnpy disponible en mi propio sandbox (solo numpy/scipy), asi
    que la parte que arma los datos y los recorta en secuencias (pasos 1-3
    mas abajo) SI fue probada con datos sinteticos en este entorno, pero la
    llamada a bnpy.run(...) y la extraccion de la secuencia de estados
    (paso 4) estan escritas contra la API publica de bnpy tal como la
    documenta el propio paper/tutoriales de Hughes (circa 2015, que es la
    epoca de ese build py27), y necesitan validarse corriendo este script
    en tu entorno real. Si algo no coincide (nombre de parametro, nombre
    de metodo), pegame el traceback completo y lo ajustamos iterando,
    igual que veniamos haciendo con extract_traceroute_v2.py.

Que hace:
  1. Lee el CSV crudo que ya genera extract_traceroute.py
     (columnas: timestamp, hop, ip, asn, rtt_ms).
  2. Para cada ciclo (timestamp), toma el RTT del hop de destino (el de
     numero de hop mas alto presente en ESE ciclo, o el que vos indiques
     con --dest-hop).
  3. Ordena los ciclos por tiempo y los corta en "secuencias" cada vez que
     hay un hueco temporal mayor a --gap-minutes (ciclos faltantes,
     perdida total, etc.) -- bnpy.GroupXData trata cada secuencia como
     independiente, para no "pegar" una transicion de estado a traves de
     un hueco real en la medicion.
  4. Corre un HDP-HMM (bnpy.run con allocModel='HDPHMM', obsModel='Gauss',
     algo='memoVB') sobre el conjunto de secuencias, con truncamiento
     --K (numero maximo de estados latentes; el proceso de Dirichlet
     decide cuantos usa realmente).
  5. Extrae la secuencia de estados mas probable por ciclo y exporta:
       a) un CSV "completo" (timestamp, rtt_ms, seq_id, state)
       b) un CSV de "cambios de segmento" (timestamp, categoria) con el
          mismo formato que el ground truth de extract_traceroute_v2.py
          (categoria='cambio_segmento_hdphmm'), pensado para poder
          superponerlo en anomaly_report.html y comparar contra los
          cambios de ruta/RTT detectados por el script de reglas.

Uso tipico (en tu entorno py27 con bnpy):
    python hdphmm_segment.py \
        --input historial_traceroute_measurement59176905_probe23108.csv \
        --output-states estados_hdphmm_measurement59176905_probe23108.csv \
        --output-events segmentos_hdphmm_measurement59176905_probe23108.csv \
        --gap-minutes 90 --K 15 --nlap 100

Para probar solo el armado de secuencias (sin bnpy, en cualquier Python
3 con numpy), usa --dry-run: hace los pasos 1-3, imprime cuantas
secuencias quedaron y sus largos, y no intenta importar bnpy.
"""
from __future__ import print_function, division

import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    sys.stderr.write("Este script necesita numpy instalado.\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paso 1-2: leer el CSV crudo y quedarnos con el RTT del hop de destino
# ---------------------------------------------------------------------------

def cargar_serie_destino(input_path, dest_hop=None):
    """
    Devuelve una lista de tuplas (timestamp_str, rtt_ms) ordenada por
    tiempo, una por ciclo, tomando el RTT del hop de destino de cada
    ciclo.

    Si dest_hop es None, usa automaticamente el numero de hop mas alto
    presente en CADA ciclo (asume que ese es el que llego mas lejos / al
    destino en ese ciclo). Si pasas --dest-hop, usa siempre ese numero de
    hop fijo y descarta los ciclos donde no este presente (recomendado si
    sabes el numero de hops del path "normal" y queres una serie mas
    limpia, a costa de generar mas huecos).
    """
    por_ciclo = defaultdict(dict)  # timestamp_str -> {hop_num: rtt}

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row["timestamp"]
            try:
                hop = int(row["hop"])
                rtt = float(row["rtt_ms"])
            except (ValueError, TypeError, KeyError):
                continue
            por_ciclo[ts][hop] = rtt

    serie = []
    for ts in sorted(por_ciclo.keys()):
        hops = por_ciclo[ts]
        if not hops:
            continue
        if dest_hop is not None:
            if dest_hop in hops:
                serie.append((ts, hops[dest_hop]))
            # si no esta ese hop en este ciclo, se omite (genera hueco)
        else:
            hop_maximo = max(hops.keys())
            serie.append((ts, hops[hop_maximo]))

    return serie


# ---------------------------------------------------------------------------
# Paso 3: cortar en secuencias por huecos temporales
# ---------------------------------------------------------------------------

def parse_ts(ts_str):
    import datetime
    return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def cortar_en_secuencias(serie, gap_minutes):
    """
    serie: lista de (timestamp_str, rtt_ms) ordenada por tiempo.
    Devuelve una lista de secuencias, cada una lista de (timestamp_str, rtt_ms),
    cortando cada vez que el hueco entre dos ciclos consecutivos supera
    gap_minutes.
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


# ---------------------------------------------------------------------------
# Paso 4: correr el HDP-HMM con bnpy
# ---------------------------------------------------------------------------

def correr_hdphmm(secuencias, K, nlap, transAlpha, startAlpha, hmmKappa, sF, seed, bnpy_outdir, alg_name):
    """
    secuencias: lista de listas de (timestamp_str, rtt_ms).
    Devuelve (hmodel, Data, estados_por_secuencia) donde
    estados_por_secuencia es una lista paralela a 'secuencias' con, para
    cada una, un array de enteros (estado inferido por ciclo).
    """
    # bnpy exige la variable de entorno BNPYOUTDIR ya al importar el
    # paquete (lo tira en __init__.py, antes de llegar a bnpy.run), asi
    # que hay que fijarla ANTES del import, no como argumento de run().
    if not os.environ.get("BNPYOUTDIR"):
        os.environ["BNPYOUTDIR"] = bnpy_outdir
    if not os.path.isdir(os.environ["BNPYOUTDIR"]):
        os.makedirs(os.environ["BNPYOUTDIR"])

    import bnpy  # import tardio: solo hace falta para esta funcion, y recien ahora BNPYOUTDIR ya esta seteada

    # bnpy.data.GroupXData espera X como matriz (N, D) y doc_range como
    # array de limites [0, len(seq0), len(seq0)+len(seq1), ...]
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
    # bnpy arma la carpeta de salida (BNPYOUTDIR/<dataName>/<jobname>/<taskid>/)
    # a partir de Data.name -- GroupXData no lo setea solo, hay que darselo
    # a mano o bnpy.run tira "dataName argument must be a string".
    Data.name = "ripe_atlas_rtt"

    # Hiperparametros: ver el paper (seccion de experimentos) y los
    # demos de bnpy para HDP-HMM + Gauss. sF controla la escala de la
    # covarianza a priori -- con RTT en ms conviene fijarlo cerca de la
    # varianza tipica de tu serie (ver nota en main()).
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

    # Asignacion dura de estado por ciclo: tomamos el componente con
    # mayor responsabilidad posterior (argmax de 'resp'). Esto es lo que
    # se usa habitualmente para graficar la segmentacion en los
    # tutoriales de bnpy; si tu build expone un camino de Viterbi mas
    # directo (por ejemplo via hmodel.allocModel / LP['respPair']),
    # avisame la salida/API exacta y lo cambiamos por el camino optimo.
    LP = hmodel.calc_local_params(Data)
    resp = LP["resp"]           # shape (N, K)
    estados = resp.argmax(axis=1)

    estados_por_secuencia = []
    for i in range(len(secuencias)):
        ini, fin = doc_range[i], doc_range[i + 1]
        estados_por_secuencia.append(estados[ini:fin])

    return hmodel, Data, estados_por_secuencia


# ---------------------------------------------------------------------------
# Paso 4b: suavizado post-hoc (dwell minimo)
# ---------------------------------------------------------------------------

def suavizar_estados(estados, min_dwell):
    """
    La asignacion dura por ciclo (argmax de la responsabilidad posterior,
    sin pasar por la matriz de transicion tipo Viterbi) suele quedar
    "nerviosa": un par de ciclos sueltos cambian de estado por el ruido
    normal del RTT y vuelven enseguida. Esto fusiona toda corrida
    (secuencia de ciclos consecutivos con el mismo estado) mas corta que
    min_dwell ciclos con la corrida vecina mas larga, de forma iterativa,
    hasta que no quedan corridas cortas (o se agota el limite de pasadas).

    Esto es puramente post-proceso sobre la lista de enteros resultante
    -- no toca el modelo ni bnpy -- asi que se puede probar y ajustar
    sin reentrenar.
    """
    estados = list(estados)
    n = len(estados)
    if n == 0 or min_dwell <= 1:
        return estados

    for _ in range(20):  # limite de pasadas de fusion
        runs = []
        i = 0
        while i < n:
            j = i
            while j < n and estados[j] == estados[i]:
                j += 1
            runs.append([i, j, estados[i]])  # [inicio, fin_exclusivo, valor]
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


# ---------------------------------------------------------------------------
# Paso 5: exportar resultados
# ---------------------------------------------------------------------------

def exportar_estados(secuencias, estados_crudos_por_secuencia, estados_suaves_por_secuencia, output_path):
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rtt_ms", "seq_id", "state_raw", "state"])
        for seq_id, (seq, crudos, suaves) in enumerate(
            zip(secuencias, estados_crudos_por_secuencia, estados_suaves_por_secuencia)
        ):
            for (ts, rtt), e_crudo, e_suave in zip(seq, crudos, suaves):
                writer.writerow([ts, rtt, seq_id, int(e_crudo), int(e_suave)])


def exportar_eventos_segmento(secuencias, estados_por_secuencia, output_path):
    """
    Un 'evento' = un cambio de estado respecto del ciclo anterior DENTRO
    de la misma secuencia (no se cuentan cambios a traves de un hueco).
    Formato igual al ground truth de extract_traceroute_v2.py
    (timestamp, categoria) para poder cargarlo en anomaly_report.html
    como una capa mas a comparar. Usa el estado YA SUAVIZADO (ver
    suavizar_estados) -- es el que se compara contra Path Analysis.
    """
    filas = []
    for seq, estados in zip(secuencias, estados_por_secuencia):
        for i in range(1, len(estados)):
            if estados[i] != estados[i - 1]:
                ts = seq[i][0]
                filas.append({
                    "timestamp": ts,
                    "categoria": "cambio_segmento_hdphmm",
                    "estado_anterior": int(estados[i - 1]),
                    "estado_nuevo": int(estados[i]),
                })

    with open(output_path, "w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "categoria", "estado_anterior", "estado_nuevo"]
        )
        writer.writeheader()
        for fila in filas:
            writer.writerow(fila)

    return len(filas)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Segmenta una serie de RTT de RIPE Atlas con un HDP-HMM "
            "(bnpy), siguiendo Hughes & Sudderth."
        )
    )
    parser.add_argument("--input", required=True,
                         help="CSV crudo de extract_traceroute_v2.py (timestamp,hop,ip,asn,rtt_ms)")
    parser.add_argument("--output-states", default=None,
                         help="CSV de salida con el estado inferido por ciclo")
    parser.add_argument("--output-events", default=None,
                         help="CSV de salida con los cambios de segmento (formato ground truth)")
    parser.add_argument("--dest-hop", type=int, default=None,
                         help="Numero de hop fijo a usar como destino (default: hop mas alto de cada ciclo)")
    parser.add_argument("--gap-minutes", type=float, default=90.0,
                         help="Hueco temporal (minutos) que corta una secuencia nueva (default: 90)")
    parser.add_argument("--K", type=int, default=15,
                         help="Truncamiento del numero de estados latentes (default: 15)")
    parser.add_argument("--nlap", type=int, default=100,
                         help="Cantidad de 'laps' (pasadas) de entrenamiento VB (default: 100)")
    parser.add_argument("--trans-alpha", type=float, default=0.5,
                         help="Hiperparametro de concentracion de las transiciones (default: 0.5)")
    parser.add_argument("--start-alpha", type=float, default=10.0,
                         help="Hiperparametro de concentracion del estado inicial (default: 10.0)")
    parser.add_argument("--hmm-kappa", type=float, default=1000.0,
                         help="'Sticky-ness': favorece permanecer en el mismo estado (default: 1000.0; "
                              "con valores bajos -tipo 50- la asignacion por ciclo puede salir "
                              "muy nerviosa/ruidosa, ver --min-dwell como red de seguridad adicional)")
    parser.add_argument("--sF", type=float, default=1.0,
                         help="Escala de la covarianza a priori del modelo Gauss (default: 1.0 -- "
                              "ESTE VALOR CASI SEGURO HAY QUE SUBIRLO: usa un numero cercano a la "
                              "varianza real de tu RTT en ms^2, que el propio script te imprime antes "
                              "de entrenar, p.ej. --sF 10 si la varianza impresa es ~10)")
    parser.add_argument("--min-dwell", type=int, default=3,
                         help="Dwell minimo en ciclos: corridas de estado mas cortas se fusionan con "
                              "la corrida vecina (suavizado post-hoc, ver suavizar_estados). "
                              "default: 3. Poner 1 para desactivar el suavizado.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bnpy-outdir", default="./bnpy-output",
                         help="Carpeta donde bnpy guarda sus logs/checkpoints internos "
                              "(fija BNPYOUTDIR si no esta ya seteada en el entorno; "
                              "default: ./bnpy-output)")
    parser.add_argument("--alg-name", default="moVB",
                         help="Nombre del algoritmo de aprendizaje registrado en tu build de "
                              "bnpy (default: moVB -- memoized online VB; tu build (8019474, "
                              "~2015) no reconocio 'memoVB'. Si moVB tampoco anda, probar 'VB' "
                              "o 'soVB'; revisar las carpetas en "
                              "<ruta-de-tu-env>/site-packages/bnpy/learnalg/ para ver cuales "
                              "estan realmente registradas en tu build)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Solo arma y corta las secuencias (sin bnpy); imprime un resumen")
    return parser.parse_args()


def main():
    args = parse_arguments()

    serie = cargar_serie_destino(args.input, dest_hop=args.dest_hop)
    if not serie:
        sys.stderr.write("No se pudo extraer ninguna muestra de RTT del CSV de entrada.\n")
        sys.exit(1)

    secuencias = cortar_en_secuencias(serie, args.gap_minutes)
    secuencias = [s for s in secuencias if len(s) >= 2]  # una secuencia de 1 punto no aporta transiciones

    total_ciclos = sum(len(s) for s in secuencias)
    print("Serie cargada: {0} ciclos totales -> {1} secuencias (gap > {2} min) "
          "-> {3} ciclos utilizables (largo >= 2).".format(
              len(serie), len(secuencias), args.gap_minutes, total_ciclos))
    largos = sorted((len(s) for s in secuencias), reverse=True)
    print("Secuencias mas largas: {0}".format(largos[:10]))

    if args.dry_run:
        print("--dry-run: no se invoca bnpy. Listo para revisar el recorte de secuencias.")
        return

    if not secuencias:
        sys.stderr.write("No quedo ninguna secuencia utilizable (todas de largo < 2).\n")
        sys.exit(1)

    # Nota sobre sF: bnpy escala la covarianza a priori con sF * ECovMat.
    # Con ECovMat='eye' y RTT en milisegundos, una serie con
    # desvio estandar tipico de, digamos, 5-20ms, va a necesitar sF del
    # orden de 10-400 (sF ~ varianza esperada) para que el prior no dome
    # la verosimilitud. Te conviene calcular np.var(rtt) sobre tu serie
    # real y pasar --sF con ese valor aproximado si el resultado sale con
    # muy pocos estados o muy inestable.
    muestra_rtt = np.array([rtt for seq in secuencias for (_, rtt) in seq])
    print("RTT observado: media={0:.2f}ms, var={1:.2f}, desvio={2:.2f}ms "
          "(considera --sF cercano a la varianza si el ajuste sale raro)".format(
              muestra_rtt.mean(), muestra_rtt.var(), muestra_rtt.std()))

    try:
        hmodel, Data, estados_por_secuencia = correr_hdphmm(
            secuencias, K=args.K, nlap=args.nlap,
            transAlpha=args.trans_alpha, startAlpha=args.start_alpha,
            hmmKappa=args.hmm_kappa, sF=args.sF, seed=args.seed,
            bnpy_outdir=args.bnpy_outdir, alg_name=args.alg_name,
        )
    except ImportError:
        sys.stderr.write(
            "No se encontro bnpy en este interprete. Corre este script con "
            "el python del entorno conda donde instalaste bnpy "
            "(ej.: /opt/miniconda3/envs/<tu-env>/bin/python hdphmm_segment.py ...).\n"
        )
        sys.exit(1)

    n_estados_usados = len(set(int(e) for arr in estados_por_secuencia for e in arr))
    print("HDP-HMM entrenado. Estados activos usados: {0} (de K={1} truncados).".format(
        n_estados_usados, args.K))

    n_cambios_crudos = sum(
        1 for arr in estados_por_secuencia for i in range(1, len(arr)) if arr[i] != arr[i - 1]
    )
    print("Cambios de estado SIN suavizar (asignacion dura por ciclo): {0} sobre {1} transiciones.".format(
        n_cambios_crudos, total_ciclos - len(secuencias)))

    estados_suaves_por_secuencia = [
        suavizar_estados(arr, args.min_dwell) for arr in estados_por_secuencia
    ]
    n_cambios_suaves = sum(
        1 for arr in estados_suaves_por_secuencia for i in range(1, len(arr)) if arr[i] != arr[i - 1]
    )
    print("Cambios de estado tras suavizado (--min-dwell {0}): {1}.".format(
        args.min_dwell, n_cambios_suaves))
    if n_cambios_crudos > 0 and n_cambios_suaves > 0.5 * n_cambios_crudos:
        print("AVISO: el suavizado redujo poco el ruido -- probablemente conviene subir --sF "
              "(cercano a la varianza impresa arriba) y/o --hmm-kappa antes de confiar en el resultado.")

    output_states = args.output_states or "estados_hdphmm.csv"
    exportar_estados(secuencias, estados_por_secuencia, estados_suaves_por_secuencia, output_states)
    print("Estados por ciclo (crudo y suavizado) exportados a: {0}".format(output_states))

    if args.output_events:
        n_eventos = exportar_eventos_segmento(secuencias, estados_suaves_por_secuencia, args.output_events)
        print("{0} cambios de segmento (sobre estado suavizado) exportados a: {1}".format(
            n_eventos, args.output_events))


if __name__ == "__main__":
    main()
