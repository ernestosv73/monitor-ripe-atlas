#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analizar_sonda.py - Flujo completo de analisis de una sonda RIPE Atlas en un solo comando.

Pide (o recibe por linea de comandos) measurement ID, probe ID, start-time, stop-time, max-hops, interval (solo para la ayuda) y persistencia, y ejecuta en orden:

  1. extractor:   python3 extract_traceroute_history.py --measurement-id M --probe-id P --start-time S --stop-time E \
                      --resolve-asn --max-hops N --ground-truth-output
  2. verificador: python3 verificar_caminos.py --historial historial_traceroute_measurementM_probeP.csv \
                      --eventos eventos_measurementM_probeP.csv --output-dir verificacion_P --max-hops N \
                      --persistencia K
  3. Path Analysis: python3 pathanalysis_url.py --measurement-id M --probe-id P --start-time S --stop-time E

Uso interactivo (pregunta lo que falte):
    python3 analizar_sonda.py

Uso directo (sin preguntas):
    python3 analizar_sonda.py -m 126502326 -p 64883 --start-time "2026-03-01 12:00" --stop-time "2026-04-01 12:00" --max-hops 30 --interval 3600 --persistencia 4

Los scripts se buscan en la misma carpeta que este archivo (o en la carpeta actual). Si el extractor no se llama
extract_traceroute_history.py se usa el primero que exista de: extract_traceroute_v2_sin_redondeo.py,
extract_traceroute_v2.py; tambien se puede indicar con --extractor.

La salida completa de los tres pasos se guarda en analisis_measurement<M>_probe<P>.log.
Fechas en UTC, formato 'YYYY-MM-DD HH:MM[:SS]'.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

EXTRACTORES = ("extract_traceroute_history.py", "extract_traceroute_v2_sin_redondeo.py", "extract_traceroute_v2.py")
MAX_HOPS_SUGERIDO = 30   # el default del extractor (15) trunca rutas largas
PERSISTENCIA_SUGERIDA = 4   # mismo default que verificar_caminos.py

def fmt_duracion(segundos, en_horas):
    """Duracion en horas (interval >= 3600 s) o en minutos (interval < 3600 s)."""
    if en_horas:
        return "{0:g} h".format(round(segundos / 3600.0, 2))
    return "{0:g} min".format(round(segundos / 60.0, 2))


def ayuda_persistencia(interval):
    """Ayuda de la persistencia con ejemplos expresados segun el interval de la medicion."""
    horas = interval >= 3600
    ej = ", ".join("{0} ciclos = {1}".format(n, fmt_duracion(n * interval, horas)) for n in (2, 4, 6))
    return (
        "\nPersistencia = cantidad minima de CICLOS consecutivos (traceroutes) con la misma longitud de ruta\n"
        "para que un tramo se considere \"sostenido\" (aparece como bloque y cuenta como cambio de camino).\n"
        "  - Duracion minima = persistencia x interval.\n"
        "  - Con el interval de esta medicion ({0}): {1}.\n"
        "  - Valor bajo (2): detecta episodios cortos, pero tambien mas ruido.\n"
        "  - Valor alto (4-6): solo cambios estables; descarta parpadeos y desvios breves.\n"
        "  Elegir: duracion minima que se considera un cambio real / interval "
        "(ej. {2} de minimo -> 2 ciclos).\n"
    ).format(fmt_duracion(interval, horas), ej, fmt_duracion(2 * interval, horas))


def parse_dt(txt):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(txt.strip(), fmt)
        except ValueError:
            pass
    raise ValueError("fecha no valida: {0!r} (usar 'YYYY-MM-DD HH:MM[:SS]', UTC)".format(txt))


def preguntar(texto, validar, valor=None, defecto=None):
    """Usa 'valor' si ya vino por linea de comandos; si no, pregunta hasta obtener un valor valido."""
    if valor is not None:
        try:
            return validar(valor)
        except ValueError as e:
            print("  [!] {0}".format(e))
    while True:
        sufijo = " [{0}]".format(defecto) if defecto is not None else ""
        try:
            r = input("{0}{1}: ".format(texto, sufijo)).strip()
        except EOFError:
            sys.exit("No hay entrada disponible: indicar el dato por linea de comandos.")
        if not r and defecto is not None:
            r = str(defecto)
        try:
            return validar(r)
        except ValueError as e:
            print("  [!] {0}".format(e))


def v_entero(txt):
    try:
        v = int(str(txt).strip())
    except ValueError:
        raise ValueError("debe ser un numero entero")
    if v <= 0:
        raise ValueError("debe ser mayor que 0")
    return v


def v_fecha(txt):
    parse_dt(txt)
    return str(txt).strip()


def buscar_script(nombre, extra_dirs):
    for d in extra_dirs:
        p = os.path.join(d, nombre)
        if os.path.isfile(p):
            return p
    return None


def correr(titulo, cmd, log, capturar=False):
    """Ejecuta un comando mostrando su salida en vivo y copiandola al log. Devuelve (codigo, salida_si_capturar)."""
    cab = "\n{0}\n{1}\n$ {2}\n{0}".format("=" * 78, titulo, " ".join(
        '"{0}"'.format(c) if " " in c else c for c in cmd))
    print(cab)
    log.write(cab + "\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    partes = []
    for linea in iter(p.stdout.readline, b""):
        txt = linea.decode("utf-8", errors="replace")
        if not capturar:
            sys.stdout.write(txt)
            sys.stdout.flush()
        log.write(txt)
        partes.append(txt)
    p.wait()
    pie = "[{0}] codigo de salida {1} ({2:.0f} s)".format(titulo.split(":")[0], p.returncode, time.time() - t0)
    print(pie)
    log.write(pie + "\n")
    return p.returncode, "".join(partes)


def main():
    ap = argparse.ArgumentParser(description="Extrae, verifica y genera la URL de Path Analysis de una sonda.")
    ap.add_argument("--measurement-id", "-m", type=int, default=None)
    ap.add_argument("--probe-id", "-p", type=int, default=None)
    ap.add_argument("--start-time", default=None, help="UTC, 'YYYY-MM-DD HH:MM[:SS]'")
    ap.add_argument("--stop-time", default=None, help="UTC, 'YYYY-MM-DD HH:MM[:SS]'")
    ap.add_argument("--max-hops", type=int, default=None, help="default sugerido: {0}".format(MAX_HOPS_SUGERIDO))
    ap.add_argument("--interval", type=int, default=None,
                    help="intervalo de la medicion en segundos (RIPE Atlas, campo Interval); solo se usa para la ayuda "
                         "de la persistencia: <3600 se muestra en minutos, >=3600 en horas")
    ap.add_argument("--persistencia", type=int, default=None,
                    help="ciclos minimos para que un tramo de igual longitud sea sostenido (default sugerido: {0}); "
                         "duracion = ciclos x cadencia".format(PERSISTENCIA_SUGERIDA))
    ap.add_argument("--extractor", default=None, help="nombre o ruta del script extractor")
    ap.add_argument("--output-dir", default=None, help="carpeta del verificador (default: verificacion_<probe-id>)")
    a = ap.parse_args()

    aqui = os.path.dirname(os.path.abspath(__file__))
    dirs = [aqui, os.getcwd()]

    # ---- scripts necesarios
    extractor = None
    if a.extractor:
        extractor = a.extractor if os.path.isfile(a.extractor) else buscar_script(a.extractor, dirs)
    else:
        for n in EXTRACTORES:
            extractor = buscar_script(n, dirs)
            if extractor:
                break
    verificador = buscar_script("verificar_caminos.py", dirs)
    pa = buscar_script("pathanalysis_url.py", dirs)
    faltan = [n for n, p in (("extractor ({0})".format(" / ".join(EXTRACTORES)), extractor),
                             ("verificar_caminos.py", verificador), ("pathanalysis_url.py", pa)) if not p]
    if faltan:
        sys.exit("No se encontraron los scripts: {0}\nBuscados en: {1}".format(", ".join(faltan), " y ".join(dirs)))

    # ---- datos
    print("Analisis de una sonda RIPE Atlas (fechas en UTC)\n")
    mid = preguntar("Measurement ID", v_entero, a.measurement_id)
    pid = preguntar("Probe ID", v_entero, a.probe_id)
    while True:
        start = preguntar("Start time (YYYY-MM-DD HH:MM)", v_fecha, a.start_time)
        stop = preguntar("Stop time  (YYYY-MM-DD HH:MM)", v_fecha, a.stop_time)
        if parse_dt(stop) > parse_dt(start):
            break
        print("  [!] stop-time debe ser posterior a start-time")
        a.start_time = a.stop_time = None
    hops = preguntar("Max hops (30 evita truncar rutas largas)", v_entero, a.max_hops, MAX_HOPS_SUGERIDO)

    if a.persistencia is None:
        interval = preguntar("Interval de la medicion en segundos (RIPE Atlas, campo Interval; ej. 600 = 10 min, 3600 = 1 h)",
                             v_entero, a.interval)
        print(ayuda_persistencia(interval))
    persist = preguntar("Persistencia minima en ciclos", v_entero, a.persistencia, PERSISTENCIA_SUGERIDA)

    historial = "historial_traceroute_measurement{0}_probe{1}.csv".format(mid, pid)
    eventos = "eventos_measurement{0}_probe{1}.csv".format(mid, pid)
    outdir = a.output_dir or "verificacion_{0}".format(pid)
    py = sys.executable or "python3"
    logp = "analisis_measurement{0}_probe{1}.log".format(mid, pid)

    resumen = []
    with open(logp, "w", encoding="utf-8") as log:
        log.write("analizar_sonda.py - {0}\nmeasurement={1} probe={2} start={3} stop={4} max_hops={5} persistencia={6}\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mid, pid, start, stop, hops, persist))

        # ---- 1. extractor
        rc, _ = correr("PASO 1/3 - Extraccion de traceroutes", [
            py, extractor, "--measurement-id", str(mid), "--probe-id", str(pid), "--start-time", start,
            "--stop-time", stop, "--resolve-asn", "--max-hops", str(hops), "--ground-truth-output"], log)
        if rc != 0 or not os.path.isfile(historial):
            msg = "El extractor fallo o no genero {0}: se detiene el flujo (revisar {1}).".format(historial, logp)
            print("\n[ERROR] " + msg)
            sys.exit(1)
        resumen.append(("Historial de saltos", historial))
        if os.path.isfile(eventos):
            resumen.append(("Eventos del extractor", eventos))

        # ---- 2. verificador
        cmd = [py, verificador, "--historial", historial, "--output-dir", outdir, "--max-hops", str(hops),
               "--persistencia", str(persist)]
        if os.path.isfile(eventos):
            cmd += ["--eventos", eventos]
        else:
            print("\n[AVISO] no existe {0}: la verificacion se hace sin eventos.".format(eventos))
        rc2, _ = correr("PASO 2/3 - Verificacion de caminos", cmd, log)
        if rc2 == 0:
            for n, d in (("Informe de verificacion", "verificacion_caminos_measurement{0}_probe{1}.txt"),
                         ("Visor HTML de caminos", "caminos_measurement{0}_probe{1}.html"),
                         ("Tramos de longitud", "tramos_longitud_measurement{0}_probe{1}.csv")):
                f = os.path.join(outdir, d.format(mid, pid))
                if os.path.isfile(f):
                    resumen.append((n, f))
        else:
            print("\n[AVISO] la verificacion termino con error (codigo {0}); se continua con Path Analysis.".format(rc2))

        # ---- 3. Path Analysis
        rc3, salida = correr("PASO 3/3 - URL de Path Analysis", [
            py, pa, "--measurement-id", str(mid), "--probe-id", str(pid), "--start-time", start,
            "--stop-time", stop], log, capturar=True)
        url = salida.strip().splitlines()[-1] if rc3 == 0 and salida.strip() else None
        if url:
            print("\n" + url)
            resumen.append(("URL de Path Analysis", url))
        else:
            print("\n[AVISO] no se pudo generar la URL de Path Analysis:\n" + salida)
        resumen.append(("Registro completo", logp))

    print("\n" + "=" * 78)
    print("RESUMEN - measurement {0} | probe {1} | {2} -> {3}".format(mid, pid, start, stop))
    print("=" * 78)
    for n, v in resumen:
        print("  {0:<26}{1}".format(n, v))
    sys.exit(0 if rc2 == 0 and url else 2)


if __name__ == "__main__":
    main()
