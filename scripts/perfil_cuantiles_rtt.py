# -*- coding: utf-8 -*-
"""
=============================================================================
perfil_cuantiles_rtt.py
Perfil de percentiles moviles y perfil horario del RTT del ultimo salto de una serie real.
=============================================================================
Responde, sin modelos ni generador, a una pregunta previa a calibrar el HDP-HMM:
  La variacion del RTT es un CAMBIO DE NIVEL sostenido (se mueve el piso) o solo cambia la DISPERSION (rafagas, picos)?

Lee los mismos archivos que el HDP-HMM (historial crudo, bloques_longitud_*.csv y ciclos_excluidos_*.csv de
verificar_caminos) y arma, por bloque coherente, la serie del RTT del ultimo salto (ciclos que llegaron al destino).

Salidas (en --output-dir):
  percentiles_moviles_<w>h.csv : P10, P25, P50, P75, P90 e IQR en ventanas moviles centradas (una fila por --paso-h).
  perfil_horario.csv           : percentiles y tasa de picos por hora del dia (con --utc-offset-h).
  perfil_diario.csv            : percentiles por dia (deriva dia a dia).
  elevaciones_piso.csv         : episodios en que el piso (percentil --percentil-piso, ventana --ventana-piso-h) queda
                                 al menos --umbrales-ms por encima de la referencia (P25 de la serie, como el HDP-HMM)
                                 durante al menos --min-persistencia-h.
  resumen_perfil.txt           : lectura automatica (piso vs cola, episodios, perfil diurno).
  perfil_rtt.png               : grafico (si matplotlib esta instalado; --sin-grafico lo omite).

Criterio de lectura (heuristico, no una prueba estadistica):
  * Si el PISO (P10/P25 movil) casi no se mueve y la COLA (P90) si, la variacion es de dispersion: no hay meseta que
    un modelo de niveles pueda segmentar.
  * Un episodio de 'elevacion del piso' es la evidencia directa de una degradacion sostenida de nivel.

Uso:
  python3 perfil_cuantiles_rtt.py \\
      --crudo historial_traceroute_measurement223088562_probe15483.csv \\
      --bloques-csv verificacion_15483/bloques_longitud_measurement223088562_probe15483.csv \\
      --ciclos-excluidos verificacion_15483/ciclos_excluidos_measurement223088562_probe15483.csv \\
      --interval 900 --utc-offset-h -3 --output-dir perfil_15483

Solo usa numpy (matplotlib opcional). No modifica ningun otro script.
=============================================================================
"""
from __future__ import print_function, division
import os
import io
import sys
import csv
import argparse
from datetime import datetime, timedelta

import numpy as np

FMT = "%Y-%m-%d %H:%M:%S"
EPOCH = datetime(1970, 1, 1)
PCT = (10, 25, 50, 75, 90)


def a_seg(ts):
    return (datetime.strptime(ts.strip()[:19], FMT) - EPOCH).total_seconds()


def de_seg(t):
    return (EPOCH + timedelta(seconds=float(t))).strftime(FMT)


def a_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ============================================================================
# CARGA
# ============================================================================
def cargar_bloques(path):
    bl = []
    with io.open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            bl.append({"id": r.get("bloque_id", ""), "longitud": int(float(r["longitud"])),
                       "inicio": r["inicio"], "fin": r["fin"], "ruta": r.get("ruta", "")})
    return bl


def cargar_excluidos(path):
    ex = set()
    if path and os.path.exists(path):
        with io.open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("ciclo") or "").strip():
                    ex.add(r["ciclo"].strip())
    return ex


def cargar_serie(historial, bloque, excl):
    """RTT del ultimo salto (hop == longitud del bloque) por ciclo, dentro del bloque y sin ciclos excluidos."""
    t0, t1 = a_seg(bloque["inicio"]), a_seg(bloque["fin"])
    ult = {}
    with io.open(historial, "r", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            ts = (r.get("timestamp") or "").strip()
            if not ts or ts in excl:
                continue
            try:
                hop = int(r["hop"])
            except (ValueError, TypeError, KeyError):
                continue
            if hop != bloque["longitud"] or not (r.get("ip") or "").strip():
                continue
            v = a_float(r.get("rtt_ms"))
            if v is None:
                continue
            t = a_seg(ts)
            if t0 <= t <= t1:
                ult[ts] = v
    orden = sorted(ult)
    return np.array([a_seg(ts) for ts in orden]), np.array([ult[ts] for ts in orden])


# ============================================================================
# CALCULOS
# ============================================================================
def percentiles_moviles(t, y, w_h, paso_h, min_frac, interval):
    """Ventanas centradas de w_h horas cada paso_h horas. Devuelve lista de (centro, n, p10..p90)."""
    out = []
    mitad = w_h * 1800.0
    esperado = w_h * 3600.0 / interval
    c = t[0] + mitad
    while c <= t[-1] - mitad + 1e-6:
        i0, i1 = np.searchsorted(t, c - mitad, "left"), np.searchsorted(t, c + mitad, "right")
        n = i1 - i0
        if n >= min_frac * esperado:
            q = np.percentile(y[i0:i1], PCT)
            out.append((c, n) + tuple(float(v) for v in q))
        c += paso_h * 3600.0
    return out


def episodios_sobre_umbral(centros, valores, ref, umbral, paso_s, min_persist_s):
    """Rachas consecutivas (paso a paso) con valor - ref >= umbral y duracion >= min_persist_s."""
    eps, i, n = [], 0, len(valores)
    while i < n:
        if valores[i] - ref >= umbral:
            j = i
            while j + 1 < n and valores[j + 1] - ref >= umbral and centros[j + 1] - centros[j] <= 1.5 * paso_s:
                j += 1
            dur = centros[j] - centros[i] + paso_s
            if dur >= min_persist_s:
                eps.append((centros[i] - paso_s / 2.0, centros[j] + paso_s / 2.0, dur / 3600.0,
                            max(valores[k] - ref for k in range(i, j + 1))))
            i = j + 1
        else:
            i += 1
    return eps


def perfil_horario(t, y, off_h, umbral_pico, mediana_global):
    h = (((t + off_h * 3600.0) % 86400.0) // 3600.0).astype(int)
    filas = []
    for hh in range(24):
        v = y[h == hh]
        if len(v) < 5:
            continue
        q = np.percentile(v, PCT)
        filas.append((hh, len(v)) + tuple(float(x) for x in q) + (float(np.mean(v)),
                                                                   float(np.mean(v > mediana_global + umbral_pico))))
    return filas


def perfil_diario(t, y, off_h):
    d = np.floor((t + off_h * 3600.0) / 86400.0).astype(int)
    filas = []
    for dd in np.unique(d):
        v = y[d == dd]
        if len(v) < 20:
            continue
        q = np.percentile(v, PCT)
        filas.append((de_seg(dd * 86400.0 - off_h * 3600.0)[:10], len(v)) + tuple(float(x) for x in q))
    return filas


def fnum(x, nd=2):
    return "-" if x is None else ("%." + str(nd) + "f") % x


def main():
    ap = argparse.ArgumentParser(description="Percentiles moviles y perfil horario del RTT del ultimo salto")
    ap.add_argument("--crudo", required=True, help="historial_traceroute_*.csv")
    ap.add_argument("--bloques-csv", required=True, help="bloques_longitud_*.csv de verificar_caminos")
    ap.add_argument("--ciclos-excluidos", default=None)
    ap.add_argument("--interval", type=float, required=True, help="Cadencia en segundos")
    ap.add_argument("--output-dir", default="perfil_rtt")
    ap.add_argument("--ventanas-h", default="6,24", help="Ventanas moviles en horas (default 6,24)")
    ap.add_argument("--paso-h", type=float, default=1.0, help="Paso entre ventanas en horas (default 1)")
    ap.add_argument("--min-cobertura", type=float, default=0.5, help="Fraccion minima de ciclos esperados por ventana (default 0.5)")
    ap.add_argument("--utc-offset-h", type=float, default=0.0, help="Desfase horario para el perfil horario (default 0 = UTC; Argentina = -3)")
    ap.add_argument("--percentil-piso", type=float, default=25.0, help="Percentil que define el piso (default 25, como la referencia del HDP-HMM)")
    ap.add_argument("--ventana-piso-h", type=float, default=6.0, help="Ventana del piso en horas (default 6)")
    ap.add_argument("--umbrales-ms", default="2,5", help="Elevacion del piso a buscar, en ms (default 2,5 = umbrales del script HDP-HMM)")
    ap.add_argument("--min-persistencia-h", type=float, default=6.0, help="Duracion minima de una elevacion en horas (default 6)")
    ap.add_argument("--umbral-pico-ms", type=float, default=10.0, help="Pico = RTT por encima de la mediana + este valor (default 10)")
    ap.add_argument("--sin-grafico", action="store_true")
    a = ap.parse_args()

    ventanas = [float(x) for x in a.ventanas_h.split(",") if x.strip()]
    umbrales = [float(x) for x in a.umbrales_ms.split(",") if x.strip()]
    paso_s = a.paso_h * 3600.0
    excl = cargar_excluidos(a.ciclos_excluidos)
    bloques = cargar_bloques(a.bloques_csv)
    os.makedirs(a.output_dir, exist_ok=True)
    L = []
    P = L.append
    P("=" * 78)
    P("PERFIL DE PERCENTILES MOVILES Y HORARIO DEL RTT (ultimo salto)")
    P("=" * 78)
    graf = []

    for b in bloques:
        t, y = cargar_serie(a.crudo, b, excl)
        sfx = "" if len(bloques) == 1 else "_b{}".format(b["id"])
        if len(y) < 100:
            P("Bloque {}: solo {} ciclos; omitido.".format(b["id"], len(y)))
            continue
        ref = float(np.percentile(y, 25))
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med))) * 1.4826
        gq = np.percentile(y, [10, 25, 50, 75, 90, 99])
        dias = (t[-1] - t[0]) / 86400.0
        P("Bloque {} ({} saltos, {}): {} ciclos, {:.1f} dias, cadencia {:g} s".format(
            b["id"], b["longitud"], b["ruta"], len(y), dias, a.interval))
        P("Global: P10 {:.2f} | P25 {:.2f} | P50 {:.2f} | P75 {:.2f} | P90 {:.2f} | P99 {:.2f} ms | MAD robusta {:.2f} ms".format(*(list(gq) + [mad])))
        P("Referencia (P25 de la serie, igual que el HDP-HMM): {:.2f} ms".format(ref))
        P("")

        # ---- percentiles moviles
        P("--- 1. PERCENTILES MOVILES: piso (P10) vs cola (P90) ---")
        P("{:<8s}{:>9s}{:>14s}{:>14s}{:>14s}{:>16s}".format("ventana", "ventanas", "P10: rango*", "P50: rango*", "P90: rango*", "lectura"))
        piso_cols = {}
        for w in ventanas:
            filas = percentiles_moviles(t, y, w, a.paso_h, a.min_cobertura, a.interval)
            if not filas:
                continue
            arr = np.array(filas)
            with io.open(os.path.join(a.output_dir, "percentiles_moviles_{:g}h{}.csv".format(w, sfx)), "w",
                         encoding="utf-8", newline="") as fo:
                wr = csv.writer(fo, lineterminator="\n")
                wr.writerow(["timestamp", "n", "p10", "p25", "p50", "p75", "p90", "iqr"])
                for r in filas:
                    wr.writerow([de_seg(r[0]), r[1]] + ["%.3f" % v for v in r[2:]] + ["%.3f" % (r[5] - r[3])])
            rng = lambda col: float(np.percentile(arr[:, col], 95) - np.percentile(arr[:, col], 5))   # noqa: E731
            r10, r50, r90 = rng(2), rng(4), rng(6)
            if r10 < 2.0 and r90 > 2.0 * r10:
                lect = "dispersion"
            elif r10 >= 2.0:
                lect = "mueve el piso"
            else:
                lect = "estable"
            P("{:<8s}{:>9d}{:>14.2f}{:>14.2f}{:>14.2f}{:>16s}".format("{:g} h".format(w), len(filas), r10, r50, r90, lect))
            piso_cols[w] = arr
        P("* rango = P95 - P5 del percentil movil a lo largo de la serie (ms). 'dispersion': el piso se mueve menos de "
          "2 ms y la cola mas del doble.")
        P("")

        # ---- elevaciones del piso
        pf = percentiles_moviles(t, y, a.ventana_piso_h, a.paso_h, a.min_cobertura, a.interval)
        P("--- 2. ELEVACIONES SOSTENIDAS DEL PISO (P{:g} movil de {:g} h sobre la referencia {:.2f} ms; persistencia >= {:g} h) ---".format(
            a.percentil_piso, a.ventana_piso_h, ref, a.min_persistencia_h))
        col = 3 if abs(a.percentil_piso - 25) < 1e-9 else 2 if abs(a.percentil_piso - 10) < 1e-9 else None
        if col is None:
            pcol = []
            mitad = a.ventana_piso_h * 1800.0
            cc = t[0] + mitad
            while cc <= t[-1] - mitad + 1e-6:
                i0, i1 = np.searchsorted(t, cc - mitad, "left"), np.searchsorted(t, cc + mitad, "right")
                if i1 - i0 >= a.min_cobertura * a.ventana_piso_h * 3600.0 / a.interval:
                    pcol.append((cc, float(np.percentile(y[i0:i1], a.percentil_piso))))
                cc += paso_s
            centros = [x[0] for x in pcol]
            valores = [x[1] for x in pcol]
        else:
            centros = [r[0] for r in pf]
            valores = [r[col] for r in pf]
        filas_ep = []
        if valores:
            P("Piso movil: min {:.2f} | mediana {:.2f} | max {:.2f} ms (referencia {:.2f})".format(
                min(valores), float(np.median(valores)), max(valores), ref))
        for u in umbrales:
            eps = episodios_sobre_umbral(centros, valores, ref, u, paso_s, a.min_persistencia_h * 3600.0)
            horas = sum(e[2] for e in eps)
            P("  >= {:g} ms: {} episodio(s), {:.1f} h ({:.1%} de la serie)".format(u, len(eps), horas, horas / (dias * 24.0)))
            for e in eps[:8]:
                P("      {} -> {}  ({:.1f} h, elevacion max {:.1f} ms)".format(de_seg(e[0]), de_seg(e[1]), e[2], e[3]))
            if len(eps) > 8:
                P("      ... ({} mas; ver elevaciones_piso.csv)".format(len(eps) - 8))
            for e in eps:
                filas_ep.append([u, de_seg(e[0]), de_seg(e[1]), "%.1f" % e[2], "%.2f" % e[3]])
        with io.open(os.path.join(a.output_dir, "elevaciones_piso{}.csv".format(sfx)), "w", encoding="utf-8", newline="") as fo:
            wr = csv.writer(fo, lineterminator="\n")
            wr.writerow(["umbral_ms", "inicio", "fin", "duracion_h", "elevacion_max_ms"])
            wr.writerows(filas_ep)
        P("")

        # ---- perfil horario y diario
        ph = perfil_horario(t, y, a.utc_offset_h, a.umbral_pico_ms, med)
        with io.open(os.path.join(a.output_dir, "perfil_horario{}.csv".format(sfx)), "w", encoding="utf-8", newline="") as fo:
            wr = csv.writer(fo, lineterminator="\n")
            wr.writerow(["hora", "n", "p10", "p25", "p50", "p75", "p90", "media", "tasa_picos"])
            for r in ph:
                wr.writerow([r[0], r[1]] + ["%.3f" % v for v in r[2:7]] + ["%.3f" % r[7], "%.4f" % r[8]])
        pdia = perfil_diario(t, y, a.utc_offset_h)
        with io.open(os.path.join(a.output_dir, "perfil_diario{}.csv".format(sfx)), "w", encoding="utf-8", newline="") as fo:
            wr = csv.writer(fo, lineterminator="\n")
            wr.writerow(["dia", "n", "p10", "p25", "p50", "p75", "p90"])
            for r in pdia:
                wr.writerow([r[0], r[1]] + ["%.3f" % v for v in r[2:]])
        P("--- 3. PERFIL HORARIO (hora local = UTC {:+g} h) ---".format(a.utc_offset_h))
        if ph:
            h50 = np.array([r[4] for r in ph])
            h10 = np.array([r[2] for r in ph])
            h90 = np.array([r[6] for r in ph])
            hp = np.array([r[8] for r in ph])
            hh = [r[0] for r in ph]
            P("Amplitud entre horas (max - min): P10 {:.2f} ms | P50 {:.2f} ms | P90 {:.2f} ms".format(
                h10.max() - h10.min(), h50.max() - h50.min(), h90.max() - h90.min()))
            P("Hora con mediana mas alta: {:02d}:00 ({:.2f} ms) | mas baja: {:02d}:00 ({:.2f} ms)".format(
                hh[int(np.argmax(h50))], h50.max(), hh[int(np.argmin(h50))], h50.min()))
            P("Tasa de picos (> mediana + {:g} ms): global {:.1%} | por hora, de {:.1%} a {:.1%} (hora mas alta {:02d}:00)".format(
                a.umbral_pico_ms, float(np.mean(y > med + a.umbral_pico_ms)), hp.min(), hp.max(), hh[int(np.argmax(hp))]))
            diurno = (h50.max() - h50.min()) >= 2.0
            P("Lectura: {}".format("hay perfil diurno en la mediana (amplitud >= 2 ms)" if diurno else
                                   "la mediana casi no depende de la hora (amplitud < 2 ms)"
                                   + ("; la tasa de picos si varia con la hora" if (hp.max() > 2 * max(hp.min(), 1e-3) and hp.max() > 0.02) else "")))
        if pdia:
            d10 = np.array([r[2] for r in pdia])
            d50 = np.array([r[4] for r in pdia])
            d90 = np.array([r[6] for r in pdia])
            P("Deriva diaria (rango entre dias): P10 {:.2f} ms | P50 {:.2f} ms | P90 {:.2f} ms ({} dias)".format(
                d10.max() - d10.min(), d50.max() - d50.min(), d90.max() - d90.min(), len(pdia)))
        P("")
        graf.append((b, t, y, piso_cols, ph, ref, pf))

    P("=" * 78)
    P("Nota: lectura heuristica; no sustituye la inspeccion visual ni una validacion con verdad conocida.")
    texto = "\n".join(L)
    print(texto)
    with io.open(os.path.join(a.output_dir, "resumen_perfil.txt"), "w", encoding="utf-8") as fo:
        fo.write(texto + u"\n")

    if not a.sin_grafico and graf:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except Exception:
            print("[AVISO] matplotlib no esta disponible: sin grafico (use --sin-grafico para silenciarlo).")
            return
        for b, t, y, piso_cols, ph, ref, pf in graf:
            fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1.3]})
            fechas = [EPOCH + timedelta(seconds=float(x)) for x in t]
            ax[0].plot(fechas, y, ".", color="#9AA5B1", markersize=1.5, label="ciclos")
            colores = {0: "#0F766E", 1: "#1F2933"}
            for k, (w, arr) in enumerate(sorted(piso_cols.items())):
                f_ = [EPOCH + timedelta(seconds=float(x)) for x in arr[:, 0]]
                ax[0].plot(f_, arr[:, 2], color="#0F766E", alpha=0.9 if k == 0 else 0.5, lw=1.2 if k == 0 else 2.0,
                           label="P10 movil {:g} h".format(w))
                ax[0].plot(f_, arr[:, 4], color="#1F2933", alpha=0.9 if k == 0 else 0.5, lw=1.2 if k == 0 else 2.0,
                           label="P50 movil {:g} h".format(w))
                ax[0].plot(f_, arr[:, 6], color="#B45309", alpha=0.9 if k == 0 else 0.5, lw=1.2 if k == 0 else 2.0,
                           label="P90 movil {:g} h".format(w))
            ax[0].axhline(ref, color="#52606D", ls="--", lw=0.8, label="referencia P25 ({:.1f} ms)".format(ref))
            ax[0].set_ylabel("RTT (ms)")
            ax[0].set_ylim(max(0, np.percentile(y, 0.5) - 2), np.percentile(y, 99.5) + 3)
            ax[0].legend(loc="upper right", fontsize=7, ncol=4)
            ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax[0].set_title("RTT del ultimo salto: percentiles moviles (bloque {})".format(b["id"]))
            if ph:
                hh = [r[0] for r in ph]
                ax[1].fill_between(hh, [r[2] for r in ph], [r[6] for r in ph], color="#CBD2D9", label="P10-P90")
                ax[1].plot(hh, [r[4] for r in ph], color="#0F766E", lw=1.6, label="mediana")
                ax[1].set_xlabel("hora del dia (UTC {:+g} h)".format(a.utc_offset_h))
                ax[1].set_ylabel("RTT (ms)")
                ax[1].set_xticks(range(0, 24, 2))
                ax[1].legend(loc="upper right", fontsize=7)
            fig.tight_layout()
            nombre = "perfil_rtt{}.png".format("" if len(graf) == 1 else "_b{}".format(b["id"]))
            fig.savefig(os.path.join(a.output_dir, nombre), dpi=130)
            plt.close(fig)
        print("Grafico: {}".format(os.path.join(a.output_dir, "perfil_rtt.png")))
    print("Archivos escritos en: {}".format(a.output_dir))


if __name__ == "__main__":
    main()
