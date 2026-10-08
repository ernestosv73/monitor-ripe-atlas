#!/usr/bin/env python3
"""
Script corregido para extraer y analizar historial de TRACEROUTE de RIPE Atlas.
Replica la lógica de detección de anomalías de RIPE Path Analysis:
🟣 RTT Degradation (>10ms Y >20% vs. CICLO ANTERIOR, con severidad graduada)
🔴 ASN/IP Change (vs. CICLO ANTERIOR en la misma posición de salto)
🟡 Path Length Change (vs. CICLO ANTERIOR)

Correcciones acumuladas respecto a la versión original:
1. Se ordena 'results' por timestamp ANTES de procesar.
2. Se usa el MÁXIMO de los paquetes por salto (no mínimo ni promedio) --
   confirmado empíricamente contra Path Analysis: un pico en 1 de 3
   paquetes (ej. [130, 7.7, 7.8]) es lo que la herramienta reporta.
3. Se resuelve ASN vía librería externa (ipwhois), adjuntado por
   ciclo/salto específico, no como lista aparte desalineada.
4. Se agregan los 3 niveles de severidad documentados por RIPE.
5. La comparación es CONSECUTIVA (ciclo N vs. ciclo N-1), no contra una
   línea base histórica global -- confirmado con evidencia real: un
   mismo pico genera DOS "changes" en Path Analysis (entrada y salida),
   algo que una comparación contra línea base global no puede replicar.
6. ✅ FIX #13 (2026-10-02) — Pérdida intermedia DEJA de contar como
   'change': se valida empíricamente (measurement 182939148, probe
   1009160, racha 28-29 jun 2026) que el conteo de 'Ciclos con cambios'
   del script reproduce EXACTO el de Path Analysis (8/8) una vez que
   solo se consideran 🟣🔴🟡🟠 -- Path Analysis no contempla pérdida
   intermedia como 'change'. Se sigue detectando y exportando al CSV de
   ground truth (para observarla en el visor HTML), pero en una lista
   aparte (perdida_log) que ya NO alimenta 'cycle_anomalies' ni
   'Ciclos con cambios'.
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from collections import Counter
import pandas as pd
import numpy as np
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
                         help="Exportar eventos clasificados a CSV (timestamp,categoria) para "
                              "usar con detect_changepoints_ripe.py --ground-truth. Si se usa sin "
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
# CLASIFICACIÓN DE SEVERIDAD (graduada, igual que RIPE Path Analysis)
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
    Categoriza una línea de anomalía a partir de su emoji inicial --
    reutilizada tanto para el resumen en consola como para la exportación
    a CSV de ground truth (exportar_ground_truth), para que ambas vistas
    nunca puedan desincronizarse entre sí. Devuelve None para las líneas
    informativas (ℹ️/⚪) que NO cuentan como anomalía real.
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
    elif anom.startswith('📉'):
        return 'perdida_intermedia'
    return None  # ℹ️, ⚪ -- informativo, no cuenta


def ruta_sin_perdida(output_path):
    """
    Nombre del tercer CSV (eventos sin pérdida intermedia): inserta
    '_sin_perdida' antes de la extensión. Ej.: eventos_x.csv -> eventos_x_sin_perdida.csv
    """
    base, ext = os.path.splitext(output_path)
    return f"{base}_sin_perdida{ext or '.csv'}"


def exportar_ground_truth(anomalies_per_cycle, perdida_log, output_path, incluir_perdida=True):
    """
    Exporta los eventos clasificados a un CSV (columnas timestamp,categoria)
    -- una fila por cada anomalía INDIVIDUAL (no por ciclo), para poder
    filtrar por categoría específica al validar contra PELT en
    detect_changepoints_ripe.py. Esto importa porque PELT solo ve la serie
    de destino: no puede detectar pérdida intermedia, ni todo cambio de
    ruta deja huella en el RTT de destino (confirmado con el caso
    2026-03-04 05:34, cambio de ASN sin degradación visible en destino) --
    poder filtrar a 'rtt_*' específicamente da una comparación más justa
    contra lo que PELT puede, en principio, encontrar.

    ✅ FIX #13 -- 'perdida_intermedia' ya NO se deriva de 'anomalies_per_cycle'
    (ahí solo quedan cambios reales, ver main()): se recibe aparte desde
    'perdida_log' para que siga observable en el CSV/visor HTML sin volver
    a contar como 'change' en ningún resumen.
    """
    rows = []
    for item in anomalies_per_cycle:
        for anom in item['anomalies']:
            categoria = clasificar_anomalia(anom)
            # 'perdida_intermedia' puede aparecer acá como línea informativa
            # (cycle_info) cuando coincide con un cambio real del mismo ciclo;
            # se omite para no duplicarla -- 'perdida_log' es su única fuente.
            if categoria is not None and categoria != 'perdida_intermedia':
                rows.append({'timestamp': item['timestamp'], 'categoria': categoria})
    if incluir_perdida:
        for item in perdida_log:
            rows.append({'timestamp': item['timestamp'], 'categoria': 'perdida_intermedia'})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('timestamp').reset_index(drop=True)
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

    # ✅ FIX #1 — ordenar explícitamente por timestamp antes de procesar
    results = sorted(results, key=lambda r: r.get('timestamp', 0))
    print(f"   ✅ {len(results)} ciclos de traceroute descargados y ordenados"
          f" ({datetime.fromtimestamp(results[0]['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
          f" → {datetime.fromtimestamp(results[-1]['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}).")

    # 2. Estructurar datos (IP + RTT por salto y ciclo)
    print("\n🔍 Procesando resultados...")
    cycles_data = []
    ips_unicas = set()

    for res in results:
        ts = res.get('timestamp')
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hops = res.get('result', [])
        dst_addr = res.get('dst_addr', 'Unknown')
        dst_responded = res.get('destination_ip_responded', None)  # ✅ FIX #9
        if dst_addr not in ('Unknown', None, '*'):
            ips_unicas.add(dst_addr)  # ✅ FIX #12 -- se resuelve también su ASN

        cycle_hops = []
        current_len = 0
        timeout_slots = 0  # ✅ FIX #10 -- saltos con pérdida total (todos los paquetes '*')

        for h in hops:
            hop_num = h.get('hop')
            if hop_num is None or hop_num > args.max_hops:
                continue

            hop_res = h.get('result', [])

            # ✅ FIX #2 (definitivo) — el PRIMER paquete válido, ni máximo ni
            # mínimo ni promedio. Confirmado con evidencia estadística: sobre
            # 11 saltos reales comparados contra Path Analysis, "primero"
            # acierta 11/11; mediana 8/11; máximo y mínimo solo 5/11 cada uno.
            # Es, además, la MISMA fuente que ya se usaba para la IP (el
            # primer paquete que respondió) -- ahora IP y RTT vienen del
            # mismo paquete de referencia, sin inconsistencia interna.
            first_pkt = next((r for r in hop_res if isinstance(r, dict) and r.get('rtt') is not None), None)

            if first_pkt is None:
                timeout_slots += 1
                continue

            rtt_hop = float(first_pkt['rtt'])
            ip = first_pkt.get('from', 'Unknown')
            ips_unicas.add(ip)

            cycle_hops.append({'hop': hop_num, 'ip': ip, 'rtt': rtt_hop, 'asn': None})
            current_len = max(current_len, hop_num)

        if cycle_hops:
            cycles_data.append({
                'timestamp': dt,
                'ts_ms': int(ts * 1000),
                'hops': cycle_hops,
                'length': current_len,
                'dst_addr': dst_addr,
                'dst_asn': None,
                'dst_responded': dst_responded,
                'timeout_slots': timeout_slots,
            })

    # ✅ FIX #3 — resolver ASN por IP única (con caché), y adjuntarlo a CADA
    # salto de CADA ciclo -- antes quedaba en una lista aparte, desalineada
    # respecto a qué ciclo específico correspondía cada valor.
    if args.resolve_asn:
        print(f"\n🌐 Resolviendo ASN para {len(ips_unicas)} IPs únicas...")
        for ip in ips_unicas:
            resolver_asn(ip)  # llena _asn_cache
        for cycle in cycles_data:
            for h in cycle['hops']:
                h['asn'] = resolver_asn(h['ip'])
            if cycle['dst_addr'] not in ('Unknown', None, '*'):
                cycle['dst_asn'] = resolver_asn(cycle['dst_addr'])
    else:
        print("\n🌐 Resolución de ASN desactivada (usar --resolve-asn para activarla).")

    normal_path_length = Counter(c['length'] for c in cycles_data).most_common(1)[0][0]
    print(f"   ✅ {len(cycles_data)} ciclos procesados (longitud típica: {normal_path_length} saltos).")

    # ✅ FIX #5 — comparación CONSECUTIVA (ciclo N vs ciclo N-1), no contra
    # línea base global. Confirmado con evidencia real: Path Analysis reporta
    # "1 change" al ENTRAR a un pico Y otro "1 change" al SALIR de él -- un
    # solo evento produce DOS cambios, porque compara pares consecutivos, no
    # contra una mediana histórica.
    print("\n🔎 Analizando ciclos en busca de cambios (comparación consecutiva)...")
    anomalies_per_cycle = []  # SOLO cambios reales (🟣🔴🟡🟠) -- determina 'Ciclos
                              # con cambios' y es lo comparable 1:1 contra Path Analysis
    perdida_log = []          # 📉 pérdida intermedia -- observacional, YA NO cuenta
                              # como 'change' (FIX #13), pero sigue yendo al CSV/HTML
    csv_rows = []
    prev_hops_by_num = {}   # {hop_num: {'ip', 'rtt', 'asn'}} del ciclo ANTERIOR
    prev_length = None
    prev_dst_addr = None
    prev_dst_asn = None
    prev_timeout_slots = None

    for cycle in cycles_data:
        dt_str = cycle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        cycle_anomalies = []   # cuentan para el total (🟣 🟡 🔴 🟠 real)
        cycle_info = []        # solo informativo, NO cuenta (ℹ️ balanceo interno)
        current_hops_by_num = {h['hop']: h for h in cycle['hops']}

        # 🟠 FIX #8 — cambio de DESTINO (dst_addr), distinto de cambio de ruta
        # hacia el mismo destino. Confirmado con evidencia real: 'facebook.com'
        # resolvió a un PoP distinto (Amsterdam->Frankfurt) durante el episodio
        # del 2026-04-02 -- lo que antes se interpretaba como 'cambio de AS-path'
        # era, en realidad, redirección DNS/anycast hacia un destino distinto.
        # ✅ FIX #12 — igual que FIX #6 para los saltos intermedios: un
        # dst_addr que alterna DENTRO DEL MISMO ASN es balanceo de carga /
        # ECMP en el borde del proveedor (ej. Cloudflare AS13335 alternando
        # entre dos direcciones del mismo nodo anycast), no una redirección
        # real hacia otro destino/PoP. Confirmado con evidencia real
        # (measurement 182939148, probe 1009160, muestra 23-25 jun 2026):
        # dst_addr alterna entre solo 2 valores dentro de AS13335, con RTT
        # prácticamente idéntico (~12-14ms) en ambos casos -- no hay indicio
        # de redirección geográfica. Requiere --resolve-asn; sin la flag, se
        # mantiene el comportamiento anterior (cualquier cambio de dst_addr
        # cuenta, porque no hay forma de saber si es el mismo ASN).
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

        # 🟡 Cambio de longitud vs. el ciclo INMEDIATAMENTE anterior
        # ✅ FIX #9 (simplificado) — en vez de perseguir causas técnicas
        # específicas (destino sin responder, MPLS, pérdida intermedia --
        # cada una explica ALGUNOS casos pero no todos, confirmado con
        # evidencia real de al menos 3 mecanismos distintos produciendo el
        # mismo síntoma), se aplica una regla empírica más simple y robusta:
        # un cambio de longitud SOLO cuenta como cambio de ruta genuino si
        # viene acompañado de al menos un 🔴 real en el MISMO ciclo. Aislado,
        # se reclasifica como ⚪ (sospechoso, no cuenta) y queda a criterio
        # del investigador revisar la URL si quiere confirmar la causa
        # puntual. Confirmado con 7/7 episodios investigados manualmente.
        length_changed = prev_length is not None and cycle['length'] != prev_length
        length_desc = f"🟡 Path Length: {cycle['length']} hops (Anterior: {prev_length})" if length_changed else None

        # 📉 FIX #10 — pérdida intermedia como categoría PROPIA, separada de
        # 'cambio de ruta'. Confirmado con evidencia real (2026-03-02 20:44):
        # un aumento de timeouts intermedios puede desplazar el conteo de
        # saltos SIN que haya degradación de RTT hacia el destino ni cambio
        # de ruta genuino -- es una señal real de pérdida, pero de una
        # naturaleza distinta a las otras 3 categorías, y merece su propio
        # registro en vez de perderse o confundirse con 'Path Length'.
        #
        # ✅ FIX #13 (2026-10-02) — DEJA de agregarse a 'cycle_anomalies':
        # validado empíricamente que Path Analysis NO cuenta la pérdida
        # intermedia como 'change' (measurement 182939148, racha 28-29 jun
        # 2026: el script reproduce 8/8 'changes' de Path Analysis sin
        # necesidad de incluirla). Se guarda en 'perdida_msg' para: (a)
        # mostrarse igual junto a un cambio real del MISMO ciclo si lo hay
        # (como antes), vía cycle_info, y (b) registrarse siempre en
        # 'perdida_log', exista o no otro cambio en el ciclo, para que
        # 'exportar_ground_truth' la siga volcando al CSV que lee el visor
        # HTML -- sin que infle 'Ciclos con cambios' ni el desglose de
        # categorías reales.
        perdida_msg = None
        if prev_timeout_slots is not None and cycle['timeout_slots'] > prev_timeout_slots:
            perdida_msg = (
                f"📉 Pérdida intermedia aumentó: {cycle['timeout_slots']} saltos sin respuesta "
                f"(Anterior: {prev_timeout_slots}) — informativo, no cuenta como cambio de ruta"
            )
            cycle_info.append(perdida_msg)
            perdida_log.append({'timestamp': dt_str, 'ts_ms': cycle['ts_ms'], 'mensaje': perdida_msg})

        for h in cycle['hops']:
            hop_num, rtt, ip, asn = h['hop'], h['rtt'], h['ip'], h['asn']
            prev_h = prev_hops_by_num.get(hop_num)

            if prev_h is not None:
                prev_rtt, prev_ip, prev_asn = prev_h['rtt'], prev_h['ip'], prev_h['asn']

                # Se determina PRIMERO si el salto cambió de IP dentro del MISMO
                # ASN (balanceo de carga interno / ECMP, FIX #6 más abajo), para
                # poder usarlo también en el chequeo de RTT que sigue.
                # ✅ FIX #7 — ASN 'Unknown' en AMBOS lados (típico de IPs
                # privadas, ej. 10.226.x.x) se trata igual que 'mismo ASN'.
                # Confirmado con evidencia exacta: excluir 'Unknown' de esta
                # regla dejaba 4 ciclos de más marcados (33 vs 29 reales) --
                # los 4 correspondían EXACTAMENTE a saltos con IP privada
                # alternando sin ningún ASN público resoluble en ningún lado.
                # Sin evidencia de un ASN real distinto, no hay base para
                # contarlo como cambio genuino.
                ip_changed = ip != 'Unknown' and prev_ip != 'Unknown' and ip != prev_ip
                mismo_asn = ip_changed and args.resolve_asn and (
                    (asn not in (None, 'Unknown') and asn == prev_asn)
                    or (asn in (None, 'Unknown') and prev_asn in (None, 'Unknown'))
                )

                # 🟣 RTT: diferencia contra el ciclo ANTERIOR, no la mediana global.
                # ✅ FIX #11 — se omite la comparación de RTT cuando el salto
                # cambió de IP por balanceo de carga interno (mismo ASN, FIX #6):
                # dos rutas ECMP paralelas dentro del mismo ASN pueden tener un
                # RTT base distinto sin que exista degradación real -- comparar
                # el RTT de un camino contra el de OTRO camino paralelo genera
                # falsos positivos. Confirmado con evidencia real (measurement
                # 182939148, probe 1009160: hops 3-6 y 9 alternan de IP dentro
                # de AS6697/AS13335 entre ciclos consecutivos, sin cambio de ruta
                # real, pero con RTT base ligeramente distinto en cada camino).
                if not mismo_asn:
                    rtt_diff = abs(rtt - prev_rtt)
                    rtt_pct = (rtt_diff / prev_rtt * 100) if prev_rtt > 0 else 0
                    if rtt_diff > 10.0 and rtt_pct > 20.0:
                        severidad = clasificar_severidad(rtt_pct)
                        cycle_anomalies.append(
                            f"{severidad} Hop {hop_num} RTT: {rtt:.1f}ms (Anterior: {prev_rtt:.1f}ms, +{rtt_pct:.0f}%)"
                        )

                # 🔴 IP y/o ASN: distintos a los del ciclo ANTERIOR en la misma posición
                # ✅ FIX #6 — un cambio de IP DENTRO DEL MISMO ASN (balanceo de
                # carga interno, ej. AS1299 alternando routers) NO CUENTA como
                # 'change' -- confirmado por evidencia real: Path Analysis
                # compara 20:38 directo contra 22:08, saltándose 5 ciclos
                # intermedios donde el ASN se mantenía igual (1299) pese a que
                # la IP específica alternaba. Va a 'cycle_info' (no cuenta),
                # no a 'cycle_anomalies'. Requiere --resolve-asn; sin la flag,
                # se mantiene el comportamiento anterior (cualquier cambio de
                # IP cuenta, porque no hay forma de saber si es el mismo ASN).
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
            # Si prev_h es None (primer ciclo, o el salto no existía antes),
            # no hay nada contra qué comparar -- no se marca nada.

            csv_rows.append({
                'timestamp': dt_str, 'hop': hop_num, 'ip': ip, 'asn': asn, 'rtt_ms': rtt,
            })

        # Decisión final del cambio de longitud, ya con el ciclo completo procesado
        if length_desc is not None:
            hay_cambio_real = any(a.startswith('🔴') for a in cycle_anomalies)
            if hay_cambio_real:
                cycle_anomalies.insert(0, length_desc)
            else:
                cycle_info.insert(0, f"⚪ {length_desc[2:]} — sin cambio de ASN/IP real acompañante, "
                                      f"probable artefacto de pérdida intermedia, NO cuenta como cambio de ruta")

        if cycle_anomalies:
            anomalies_per_cycle.append({
                'timestamp': dt_str, 'ts_ms': cycle['ts_ms'],
                'anomalies': cycle_anomalies + cycle_info,  # info solo se muestra si YA hay una anomalía real
            })

        prev_hops_by_num = current_hops_by_num
        prev_length = cycle['length']
        prev_dst_addr = cycle['dst_addr']
        prev_dst_asn = cycle['dst_asn']
        prev_timeout_slots = cycle['timeout_slots']

    # 4. Guardar CSV
    output_file = args.output or f"historial_traceroute_measurement{args.measurement_id}_probe{args.probe_id}.csv"
    pd.DataFrame(csv_rows).to_csv(output_file, index=False)
    print(f"\n💾 Datos detallados guardados en: {output_file}")

    if args.ground_truth_output:
        gt_output = args.ground_truth_output
        if gt_output == "__auto__":
            gt_output = f"eventos_measurement{args.measurement_id}_probe{args.probe_id}.csv"
        n_exportadas = exportar_ground_truth(anomalies_per_cycle, perdida_log, gt_output)
        print(f"💾 {n_exportadas} eventos de ground truth exportados a: {gt_output}")

        # Tercer archivo: mismos eventos pero SIN 'perdida_intermedia' (para
        # validar el HDP-HMM contra un ground truth solo de cambios).
        gt_sin_perdida = ruta_sin_perdida(gt_output)
        n_sin_perdida = exportar_ground_truth(anomalies_per_cycle, perdida_log, gt_sin_perdida,
                                              incluir_perdida=False)
        print(f"💾 {n_sin_perdida} eventos (sin pérdida intermedia) exportados a: {gt_sin_perdida}")

    # 5. Reporte
    print(f"\n📊 Resumen de Cambios Detectados:")
    print(f"   Ciclos totales analizados: {len(cycles_data)}")
    print(f"   Ciclos con cambios: {len(anomalies_per_cycle)} "
          f"({100*len(anomalies_per_cycle)/max(1, len(cycles_data)):.1f}%)")

    # ✅ Desglose por categoría -- cuenta cada LÍNEA de anomalía (no ciclo,
    # ya que un mismo ciclo puede tener varias categorías a la vez, como
    # vimos con el caso Facebook: 🟠+🟡+🔴 juntos en el mismo ciclo).
    # Reutiliza clasificar_anomalia() -- la MISMA función que usa
    # exportar_ground_truth, para que consola y CSV nunca diverjan.
    NOMBRES_CATEGORIA = {
        'rtt_dark': 'RTT - Dark (>200%)', 'rtt_medium': 'RTT - Medium (40-200%)',
        'rtt_light': 'RTT - Light (20-40%)', 'cambio_ip_asn': 'Cambio de IP/ASN en un salto',
        'cambio_longitud': 'Cambio de longitud (con 🔴 real acompañante)',
        'cambio_destino': 'Cambio de destino (dst_addr)', 'perdida_intermedia': 'Pérdida intermedia',
    }
    conteo = Counter()
    conteo_info = Counter()
    for item in anomalies_per_cycle:
        for anom in item['anomalies']:
            categoria = clasificar_anomalia(anom)
            # ✅ FIX #13 -- 'perdida_intermedia' puede aparecer acá solo como
            # línea informativa (cycle_info) junto a un cambio real del mismo
            # ciclo; se excluye de 'conteo' (cambios reales) y se contabiliza
            # aparte más abajo desde 'perdida_log', su única fuente de verdad.
            if categoria is not None and categoria != 'perdida_intermedia':
                conteo[NOMBRES_CATEGORIA[categoria]] += 1
            elif anom.startswith('⚪'):
                conteo_info['Longitud sospechosa (sin 🔴 real, no cuenta)'] += 1
    if perdida_log:
        conteo_info['Pérdida intermedia (informativo, no cuenta como cambio)'] = len(perdida_log)

    if conteo:
        print(f"\n📋 Desglose por categoría ({sum(conteo.values())} líneas totales):")
        for categoria, cantidad in conteo.most_common():
            print(f"   {categoria:45s}: {cantidad}")
    if conteo_info:
        print(f"\nℹ️ Informativo, no cuenta como cambio:")
        for categoria, cantidad in conteo_info.most_common():
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

    # ✅ FIX #13 -- listado aparte, solo informativo, de los ciclos donde la
    # ÚNICA señal fue pérdida intermedia (no están en 'anomalies_per_cycle'
    # porque ya no cuentan como cambio). Los que coincidieron con un cambio
    # real ya se imprimieron arriba, dentro de ese ciclo -- se excluyen acá
    # para no duplicarlos.
    timestamps_con_cambio_real = {item['timestamp'] for item in anomalies_per_cycle}
    perdida_aislada = [p for p in perdida_log if p['timestamp'] not in timestamps_con_cambio_real]
    if perdida_aislada:
        print(f"\n📉 {len(perdida_aislada)} ciclos con pérdida intermedia AISLADA "
              f"(observacional, NO cuentan como cambio):")
        for item in perdida_aislada:
            print(f"   🕒 {item['timestamp']} UTC — {item['mensaje']}")

    print("\n✅ ¡Proceso completado!")


if __name__ == "__main__":
    main()
