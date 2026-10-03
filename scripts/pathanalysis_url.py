#!/usr/bin/env python3
"""
Genera la URL de RIPE Atlas Path Analysis para un rango de fechas dado,
sin descargar ni procesar ninguna medición -- pensado para abrir rápido
en el navegador el mismo tramo que reporta extract_traceroute_v2.py y
poder contrastar a mano ("changes detected") durante la validación
cruzada.

Uso:
    python3 pathanalysis_url.py --measurement-id 59176905 --probe-id 23108 \
        --start-time "2026-03-02 12:00" --stop-time "2026-04-04 12:00"

Imprime por stdout una única línea con la URL lista para abrir (sin
texto adicional, para poder usarla directo en un pipe, ej.:
    xdg-open "$(python3 pathanalysis_url.py -m ... -p ... --start-time ... --stop-time ...)"
).

Nota sobre 'window': Path Analysis lo interpreta como el RADIO alrededor
de 'center', no como el rango total -- confirmado empíricamente
comparando el 'window' pasado contra el 'Found N traceroute(s) from .. to
..' que reporta la propia UI (ver measurement 59176905/probe 23108: para
cubrir 2026-03-02 12:00 → 2026-04-04 12:00 completo, RIPE generó
center=2026-03-19T00:00:00Z y window=1425600000, que es exactamente la
mitad del rango). Por eso acá 'window' se calcula como la mitad del
rango pedido y 'center' como su punto medio.
"""
import argparse
import sys
from datetime import datetime, timezone


def parse_dt(dt_str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Formato de fecha no válido: {dt_str!r} (usar 'YYYY-MM-DD HH:MM[:SS]', UTC)")


def build_url(measurement_id, probe_id, start, stop):
    if stop <= start:
        raise ValueError("--stop-time debe ser posterior a --start-time")
    start_ms = int(start.timestamp() * 1000)
    stop_ms = int(stop.timestamp() * 1000)
    window_ms = (stop_ms - start_ms) // 2   # radio -- ver nota en el docstring del módulo
    center_ms = start_ms + window_ms        # punto medio del rango pedido
    center_iso = datetime.fromtimestamp(center_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        "https://atlas.ripe.net/pathanalysis/embed?"
        f"sourceProbeId={probe_id}&measurementId={measurement_id}&"
        f"sequentialMode=true&center={center_iso}&window={window_ms}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Genera la URL de RIPE Atlas Path Analysis para un rango de fechas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--measurement-id", "-m", type=int, required=True)
    parser.add_argument("--probe-id", "-p", type=int, required=True)
    parser.add_argument("--start-time", type=str, required=True,
                         help="Formato 'YYYY-MM-DD HH:MM[:SS]', en UTC (igual que extract_traceroute_v2.py)")
    parser.add_argument("--stop-time", type=str, required=True,
                         help="Formato 'YYYY-MM-DD HH:MM[:SS]', en UTC (igual que extract_traceroute_v2.py)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        start = parse_dt(args.start_time)
        stop = parse_dt(args.stop_time)
        url = build_url(args.measurement_id, args.probe_id, start, stop)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    print(url)


if __name__ == "__main__":
    main()
