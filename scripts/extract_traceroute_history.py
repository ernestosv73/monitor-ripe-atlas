#!/usr/bin/env python3
"""
Extrae y analiza el historial de TRACEROUTE de RIPE Atlas (una sonda) y detecta
cambios entre ciclos CONSECUTIVOS, replicando la logica de Path Diagnostics de
RIPE Path Analysis (PA).

Que cuenta como cambio (ciclo N contra ciclo N-1):
  🟣 RTT       : en un mismo salto, |RTT - RTT_previo| > 10 ms Y > 20 % del previo
                 (severidad Light 20-40 %, Medium 40-200 %, Dark > 200 %).
  🔴 IP/ASN    : un salto cambia de IP y de ASN. Un cambio de IP DENTRO del mismo ASN
                 es balanceo interno y no cuenta (tampoco se compara su RTT).
  🟡 Longitud  : cambia la CANTIDAD de saltos de la ruta, aunque no cambie ningun ASN.
  🟠 Destino   : cambia dst_addr hacia otro ASN (redireccion DNS/anycast).

Representacion de la ruta (igual que el JSON de PA):
  - Cada ciclo es una lista de saltos 1..N, donde N es el ultimo salto que respondio.
  - Los saltos que NO respondieron se conservan como NULL (ip, asn y rtt vacios en el CSV);
    no se eliminan. La longitud de la ruta es N (incluye los NULL intermedios).
  - Un salto NULL en cualquiera de los dos ciclos no se compara (ni RTT ni IP/ASN),
    pero SI altera la longitud si desplaza el ultimo salto que respondio.

La perdida intermedia (saltos sin respuesta) YA NO es un evento: solo se refleja en
la longitud cuando cambia el numero de saltos.

Decisiones validadas contra PA (mediciones 59176905, 182939148 y 9213364):
  - Se usa el PRIMER paquete que respondio en cada salto (IP y RTT del mismo paquete).
  - Comparacion consecutiva, no contra una linea base global: un pico genera dos
    cambios (entrada y salida).
  - El RTT se compara tal cual lo devuelve la API (sin redondear). La variante que lo
    redondea a 0.1 ms antes de comparar (como exporta PA) es extract_traceroute_v2.py
    con --redondear-rtt.
  - Un cambio de IP dentro del mismo ASN (o 'Unknown' en ambos lados) no cuenta.

Salidas:
  - historial_traceroute_measurement<id>_probe<id>.csv  (timestamp,hop,ip,asn,rtt_ms)
  - con --ground-truth-output: eventos_measurement<id>_probe<id>.csv (timestamp,categoria),
    UN SOLO archivo, una fila por anomalia individual.
"""
import argparse
import sys
from datetime import datetime, timezone
from collections import Counter
import pandas as pd
from ripe.atlas.cousteau import AtlasResultsRequest

try:
    from ipwhois import IPWhois
    IPWHOIS_DISPONIBLE = True
except ImportError:
    IPWHOIS_DISPONIBLE = False


# ==========================================
# PARSING DE ARGUMENTOS
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extraer y analizar historial de TRACEROUTE de UNA SOLA SONDA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--measurement-id", "-m", type=int, required=True)
    parser.add_argument("--probe-id", "-p", type=int, required=True)
    parser.add_argument("--start-time", type=str, required=True)
    parser.add_argument("--stop-time", type=str, required=True)
    parser.add_argument("--max-hops", type=int, default=15)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--resolve-asn", action="store_true",
                        help="Resolver ASN vía whois (más lento, requiere red)")
    parser.add_argument("--ground-truth-output", type=str, nargs="?", const="__auto__", default=None,
                        help="Exportar los eventos a UN CSV (timestamp,categoria). Si se usa sin "
                             "valor, el nombre se genera automáticamente incluyendo measurement "
                             "ID y probe ID (ej. eventos_measurement<id>_probe<id>.csv).")
    return parser.parse_args()


def parse_dt(dt_str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Formato de fecha no válido: {dt_str}")


# ==========================================
# RESOLUCIÓN DE ASN (con caché, para no repetir lookups)
# ==========================================
_asn_cache = {}

def resolver_asn(ip):
    if ip in ('Unknown', None, '*'):
        return 'Unknown'
    if ip in _asn_cache:
        return _asn_cache[ip]
    if not IPWHOIS_DISPONIBLE:
        return 'Unknown'
    try:
        resultado = IPWhois(ip).lookup_rdap(depth=0)
        asn = resultado.get('asn', 'Unknown')
    except Exception:
        asn = 'Unknown'
    _asn_cache[ip] = asn
    return asn


# ==========================================
# CLASIFICACIÓN
# ==========================================
def clasificar_severidad(rtt_pct):
    if rtt_pct > 200:
        return "🟣⬛ Dark (>200%)"
    elif rtt_pct > 40:
        return "🟣🟪 Medium (40-200%)"
    else:
        return "🟣 Light (20-40%)"


def clasificar_anomalia(anom):
    """
    Categoriza una línea de anomalía a partir de su emoji inicial -- la usan tanto
    el resumen en consola como la exportación a CSV, para que nunca se desincronicen.
    Devuelve None para las líneas informativas (ℹ️), que NO cuentan como anomalía.
    """
    if anom.startswith('🟣⬛'):
        return 'rtt_dark'
    elif anom.startswith('🟣🟪'):
        return 'rtt_medium'
    elif anom.startswith('🟣'):
        return 'rtt_light'
    elif anom.startswith('🔴'):
        return 'cambio_ip_asn'
    elif anom.startswith('🟡'):
        return 'cambio_longitud'
    elif anom.startswith('🟠'):
        return 'cambio_destino'
    return None


def exportar_ground_truth(anomalies_per_cycle, output_path):
    """
    Exporta los eventos a un CSV (timestamp,categoria): una fila por cada anomalía
    INDIVIDUAL (no por ciclo), para poder filtrar por categoría al validar contra
    otros detectores o contra el visor HTML.
    """
    rows = []
    for item in anomalies_per_cycle:
        for anom in item['anomalies']:
            categoria = clasificar_anomalia(anom)
            if categoria is not None:
                rows.append({'timestamp': item['timestamp'], 'categoria': categoria})
    df = pd.DataFrame(rows, columns=['timestamp', 'categoria'])
    if not df.empty:
        df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return len(df)


# ==========================================
# LÓGICA PRINCIPAL
# ==========================================
def main():
    args = parse_arguments()
    start_time = parse_dt(args.start_time)
    stop_time = parse_dt(args.stop_time)

    print("=" * 70)
    print("EXTRACCIÓN Y ANÁLISIS DE TRACEROUTE (RIPE Atlas)")
    print("=" * 70)
    print(f"\n⚙️  Configuración:")
    print(f"   Measurement ID: {args.measurement_id}")
    print(f"   Probe ID: {args.probe_id}")
    print(f"   Rango: {start_time.strftime('%Y-%m-%d %H:%M')} a {stop_time.strftime('%Y-%m-%d %H:%M')}")
    if args.resolve_asn and not IPWHOIS_DISPONIBLE:
        print("   ⚠️ 'ipwhois' no está instalado -- ASN quedará como 'Unknown'.")
        print("      Instalar con: pip install ipwhois --break-system-packages")

    # 1. Descargar datos
    print(f"\n📥 Solicitando historial de traceroute...")
    kwargs = {
        "msm_id": args.measurement_id,
        "start": int(start_time.timestamp()),
        "stop": int(stop_time.timestamp()),
        "probe_ids": [args.probe_id]
    }
    is_success, results = AtlasResultsRequest(**kwargs).create()
    if not is_success or not results:
        print(f"❌ Error o sin resultados: {results}")
        sys.exit(1)

    results = sorted(results, key=lambda r: r.get('timestamp', 0))
    print(f"   ✅ {len(results)} ciclos de traceroute descargados y ordenados"
          f" ({datetime.fromtimestamp(results[0]['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
          f" → {datetime.fromtimestamp(results[-1]['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}).")

    # 2. Estructurar cada ciclo como lista de saltos 1..N (N = último salto que respondió).
    #    Los saltos sin respuesta quedan como None (NULL) en vez de eliminarse.
    print("\n🔍 Procesando resultados...")
    cycles_data = []
    ips_unicas = set()

    for res in results:
        ts = res.get('timestamp')
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        dst_addr = res.get('dst_addr', 'Unknown')
        if dst_addr not in ('Unknown', None, '*'):
            ips_unicas.add(dst_addr)

        respondidos = {}
        for h in res.get('result', []):
            hop_num = h.get('hop')
            if hop_num is None or hop_num > args.max_hops:
                continue
            # El PRIMER paquete válido: IP y RTT salen del mismo paquete de referencia.
            first_pkt = next((r for r in h.get('result', [])
                              if isinstance(r, dict) and r.get('rtt') is not None), None)
            if first_pkt is None:
                continue  # salto sin respuesta -> NULL (se rellena abajo)
            rtt_hop = float(first_pkt['rtt'])
            ip = first_pkt.get('from', 'Unknown')
            ips_unicas.add(ip)
            respondidos[hop_num] = {'hop': hop_num, 'ip': ip, 'rtt': rtt_hop, 'asn': None}

        if not respondidos:
            continue  # ciclo sin ninguna respuesta: no hay ruta que comparar
        length = max(respondidos)
        cycles_data.append({
            'timestamp': dt,
            'ts_ms': int(ts * 1000),
            'hops': {n: respondidos.get(n) for n in range(1, length + 1)},  # None = NULL
            'length': length,
            'dst_addr': dst_addr,
            'dst_asn': None,
        })

    # Resolver ASN por IP única (con caché) y adjuntarlo a cada salto de cada ciclo.
    if args.resolve_asn:
        print(f"\n🌐 Resolviendo ASN para {len(ips_unicas)} IPs únicas...")
        for ip in ips_unicas:
            resolver_asn(ip)  # llena _asn_cache
        for cycle in cycles_data:
            for h in cycle['hops'].values():
                if h is not None:
                    h['asn'] = resolver_asn(h['ip'])
            if cycle['dst_addr'] not in ('Unknown', None, '*'):
                cycle['dst_asn'] = resolver_asn(cycle['dst_addr'])
    else:
        print("\n🌐 Resolución de ASN desactivada (usar --resolve-asn para activarla).")

    normal_path_length = Counter(c['length'] for c in cycles_data).most_common(1)[0][0]
    print(f"   ✅ {len(cycles_data)} ciclos procesados (longitud típica: {normal_path_length} saltos).")

    # 3. Comparación CONSECUTIVA (ciclo N vs ciclo N-1): un pico genera DOS cambios
    #    (al entrar y al salir), igual que Path Analysis.
    print("\n🔎 Analizando ciclos en busca de cambios (comparación consecutiva)...")
    anomalies_per_cycle = []   # solo ciclos con al menos un cambio real (🟣🔴🟡🟠)
    csv_rows = []
    prev_hops = {}             # {hop: dict | None} del ciclo ANTERIOR
    prev_length = None
    prev_dst_addr = None
    prev_dst_asn = None

    for cycle in cycles_data:
        dt_str = cycle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        cycle_anomalies = []   # cuentan como cambio
        cycle_info = []        # solo informativo (ℹ️ balanceo interno), no cuenta
        cur_hops = cycle['hops']

        # 🟠 Cambio de DESTINO (dst_addr). Si alterna dentro del MISMO ASN es balanceo
        # de carga / ECMP en el borde del proveedor y no cuenta. Requiere --resolve-asn;
        # sin la flag, cualquier cambio de dst_addr cuenta.
        if prev_dst_addr is not None and cycle['dst_addr'] != prev_dst_addr:
            dst_asn = cycle['dst_asn']
            mismo_asn_dst = args.resolve_asn and (
                (dst_asn not in (None, 'Unknown') and dst_asn == prev_dst_asn)
                or (dst_asn in (None, 'Unknown') and prev_dst_asn in (None, 'Unknown'))
            )
            if mismo_asn_dst:
                motivo = f"mismo ASN {dst_asn}" if dst_asn not in (None, 'Unknown') else "ASN no resoluble en ningún lado"
                cycle_info.append(
                    f"ℹ️ Destino cambió: {cycle['dst_addr']} (Anterior: {prev_dst_addr}) — {motivo}, "
                    f"NO cuenta como cambio (balanceo de carga interno)"
                )
            else:
                etiqueta_asn = f", ASN {prev_dst_asn}->{dst_asn}" if args.resolve_asn else ""
                cycle_anomalies.append(
                    f"🟠 Destino cambió: {cycle['dst_addr']} (Anterior: {prev_dst_addr}{etiqueta_asn}) "
                    f"— probable redirección DNS/anycast, no cambio de ruta hacia el mismo destino"
                )

        # 🟡 Cambio de LONGITUD: distinta cantidad de saltos respecto al ciclo anterior,
        # aunque no cambie ningún ASN (saltos NULL incluidos en la cuenta).
        if prev_length is not None and cycle['length'] != prev_length:
            cycle_anomalies.append(
                f"🟡 Longitud de ruta: {cycle['length']} saltos (Anterior: {prev_length})"
            )

        # Comparación salto a salto. Un salto NULL en cualquiera de los dos ciclos no se compara.
        for hop_num in range(1, cycle['length'] + 1):
            h = cur_hops.get(hop_num)
            prev_h = prev_hops.get(hop_num)
            if h is not None and prev_h is not None:
                rtt, ip, asn = h['rtt'], h['ip'], h['asn']
                prev_rtt, prev_ip, prev_asn = prev_h['rtt'], prev_h['ip'], prev_h['asn']

                # ¿Cambió de IP dentro del MISMO ASN (balanceo interno / ECMP)? 'Unknown'
                # en AMBOS lados (típico de IPs privadas) se trata igual que 'mismo ASN'.
                ip_changed = ip != 'Unknown' and prev_ip != 'Unknown' and ip != prev_ip
                mismo_asn = ip_changed and args.resolve_asn and (
                    (asn not in (None, 'Unknown') and asn == prev_asn)
                    or (asn in (None, 'Unknown') and prev_asn in (None, 'Unknown'))
                )

                # 🟣 RTT contra el ciclo anterior. Se omite si el salto cambió de IP por
                # balanceo interno: dos rutas paralelas del mismo ASN pueden tener un RTT
                # base distinto sin que exista degradación real.
                if not mismo_asn:
                    rtt_diff = abs(rtt - prev_rtt)
                    rtt_pct = (rtt_diff / prev_rtt * 100) if prev_rtt > 0 else 0
                    if rtt_diff > 10.0 and rtt_pct > 20.0:
                        severidad = clasificar_severidad(rtt_pct)
                        cycle_anomalies.append(
                            f"{severidad} Hop {hop_num} RTT: {rtt:.1f}ms (Anterior: {prev_rtt:.1f}ms, +{rtt_pct:.0f}%)"
                        )

                # 🔴 IP y ASN distintos a los del ciclo anterior en la misma posición.
                if ip_changed:
                    if mismo_asn:
                        motivo = f"mismo ASN {asn}" if asn not in (None, 'Unknown') else "IP privada, sin ASN público resoluble en ningún lado"
                        cycle_info.append(
                            f"ℹ️ Hop {hop_num} IP: {ip} (Anterior: {prev_ip}) — {motivo}, "
                            f"NO cuenta como cambio (balanceo de carga interno)"
                        )
                    else:
                        etiqueta_asn = f", ASN {prev_asn}->{asn}" if args.resolve_asn else ""
                        cycle_anomalies.append(f"🔴 Hop {hop_num} IP: {ip} (Anterior: {prev_ip}{etiqueta_asn})")

            # Una fila por salto, también los NULL (ip/asn/rtt vacíos)
            csv_rows.append({
                'timestamp': dt_str, 'hop': hop_num,
                'ip': h['ip'] if h is not None else None,
                'asn': h['asn'] if h is not None else None,
                'rtt_ms': h['rtt'] if h is not None else None,
            })

        if cycle_anomalies:
            anomalies_per_cycle.append({
                'timestamp': dt_str, 'ts_ms': cycle['ts_ms'],
                'anomalies': cycle_anomalies + cycle_info,  # info solo se muestra si YA hay una anomalía real
            })

        prev_hops = cur_hops
        prev_length = cycle['length']
        prev_dst_addr = cycle['dst_addr']
        prev_dst_asn = cycle['dst_asn']

    # 4. Guardar CSV (con los saltos NULL)
    output_file = args.output or f"historial_traceroute_measurement{args.measurement_id}_probe{args.probe_id}.csv"
    pd.DataFrame(csv_rows, columns=['timestamp', 'hop', 'ip', 'asn', 'rtt_ms']).to_csv(output_file, index=False)
    print(f"\n💾 Datos detallados guardados en: {output_file}")

    if args.ground_truth_output:
        gt_output = args.ground_truth_output
        if gt_output == "__auto__":
            gt_output = f"eventos_measurement{args.measurement_id}_probe{args.probe_id}.csv"
        n_exportadas = exportar_ground_truth(anomalies_per_cycle, gt_output)
        print(f"💾 {n_exportadas} eventos exportados a: {gt_output}")

    # 5. Reporte
    print(f"\n📊 Resumen de Cambios Detectados:")
    print(f"   Ciclos totales analizados: {len(cycles_data)}")
    print(f"   Ciclos con cambios: {len(anomalies_per_cycle)} "
          f"({100*len(anomalies_per_cycle)/max(1, len(cycles_data)):.1f}%)")

    # Desglose por categoría: cuenta cada LÍNEA de anomalía (un ciclo puede tener varias).
    NOMBRES_CATEGORIA = {
        'rtt_dark': 'RTT - Dark (>200%)', 'rtt_medium': 'RTT - Medium (40-200%)',
        'rtt_light': 'RTT - Light (20-40%)', 'cambio_ip_asn': 'Cambio de IP/ASN en un salto',
        'cambio_longitud': 'Cambio de longitud (cantidad de saltos)',
        'cambio_destino': 'Cambio de destino (dst_addr)',
    }
    conteo = Counter()
    for item in anomalies_per_cycle:
        for anom in item['anomalies']:
            categoria = clasificar_anomalia(anom)
            if categoria is not None:
                conteo[NOMBRES_CATEGORIA[categoria]] += 1

    if conteo:
        print(f"\n📋 Desglose por categoría ({sum(conteo.values())} líneas totales):")
        for categoria, cantidad in conteo.most_common():
            print(f"   {categoria:45s}: {cantidad}")

    if anomalies_per_cycle:
        print(f"\n🔗 Los {len(anomalies_per_cycle)} ciclos con cambios (orden cronológico):")
        for item in anomalies_per_cycle:
            print(f"\n   🕒 {item['timestamp']} UTC:")
            for anom in item['anomalies']:
                print(f"      {anom}")
            url = (f"https://atlas.ripe.net/pathanalysis/embed?"
                   f"measurementId={args.measurement_id}&sourceProbeId={args.probe_id}&"
                   f"center={item['ts_ms']}&window=7200000")
            print(f"      🔗 URL: {url}")

    print("\n✅ ¡Proceso completado!")


if __name__ == "__main__":
    main()
