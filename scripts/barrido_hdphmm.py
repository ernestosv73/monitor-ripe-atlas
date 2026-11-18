# -*- coding: utf-8 -*-
"""
=============================================================================
barrido_hdphmm.py
Lanza el script HDP-HMM sobre UNA serie real con varias combinaciones de --sF, --hmm-kappa, --percentil-outliers y
--seed, y resume los resultados en una tabla. NO modifica el script HDP-HMM ni usa el generador: solo ejecuta el
script existente como proceso externo y lee lo que imprime y los CSV que exporta.

Pide por consola los valores a barrer (Enter acepta el valor entre corchetes):
  --sF, --hmm-kappa, --percentil-outliers y las semillas. El PRIMER valor de cada lista es la linea base.
Modos:
  uno    (default): linea base + cada valor distinto de la base, un parametro por vez. Cada configuracion se repite
         con todas las semillas.
  rejilla: todas las combinaciones.

Por corrida extrae de la salida: estados activos usados, macro-estados (si hay fusion), puntos de cambio, grupos,
estados (media y desviacion) y, de segmentos_caracterizados_*.csv, segmentos, % de ciclos en 'degradacion sostenida'.
Por configuracion (sobre las semillas) calcula la ESTABILIDAD: rango de estados y de puntos de cambio, y el acuerdo
medio entre semillas de las fronteras de segmento (F1 con tolerancia en ciclos).

Uso (con el mismo interprete que usa para el script HDP-HMM, p. ej. el de Python 2.7 con bnpy):
  python barrido_hdphmm.py \\
      --script hdphmm_ripe_atlas_bnpy_f.py \\
      --crudo historial_traceroute_measurement223088562_probe15483.csv \\
      --bloques-csv verificacion_15483/bloques_longitud_measurement223088562_probe15483.csv \\
      --ciclos-excluidos verificacion_15483/ciclos_excluidos_measurement223088562_probe15483.csv \\
      --interval 900 --gap-minutes 90 --K 15 --nlap 300 --min-dwell 4 \\
      --output-dir barrido_15483

Compatible con Python 2.7 y 3. Solo biblioteca estandar. Cada corrida queda en <output-dir>/run_<etiqueta>/ con su
salida.txt; con --reanudar se saltan las corridas ya terminadas.
=============================================================================
"""
from __future__ import print_function, division
import os
import re
import io
import sys
import csv
import time
import shlex
import argparse
import itertools
import subprocess
from datetime import datetime

FMT = "%Y-%m-%d %H:%M:%S"
EPOCH = datetime(1970, 1, 1)

try:
    _input = raw_input          # Python 2
except NameError:
    _input = input


def a_seg(ts):
    return (datetime.strptime(ts.strip()[:19], FMT) - EPOCH).total_seconds()


def pedir(texto, defecto, cast, sin_prompt):
    """Pide una lista separada por comas; Enter acepta el defecto."""
    if sin_prompt:
        return [cast(x) for x in defecto.split(",") if x.strip()]
    while True:
        try:
            r = _input("{} [{}]: ".format(texto, defecto)).strip()
        except EOFError:
            r = ""
        r = r or defecto
        try:
            vals = [cast(x) for x in r.split(",") if x.strip()]
            if vals:
                return vals
        except ValueError:
            pass
        print("  Valor no valido; use numeros separados por coma.")


def etiqueta_val(x):
    return ("%g" % x).replace(".", "p").replace("-", "m")


# ============================================================================
# EXTRACCION DE LA SALIDA
# ============================================================================
def parsear_salida(texto):
    d = {"estados": None, "k": None, "macro": None, "cps": None, "grupos": None, "estados_det": [],
         "completo": ("EJECUCI" in texto and "COMPLETADA EXITOSAMENTE" in texto)}
    m = re.search(r"Estados activos usados: (\d+) \(de K=(\d+)", texto)
    if m:
        d["estados"], d["k"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"Mapa de fusi.n generado \((\d+) macro", texto)
    if m:
        d["macro"] = int(m.group(1))
    m = re.search(r"Puntos de cambio detectados: (\d+)", texto)
    if m:
        d["cps"] = int(m.group(1))
    m = re.search(r"\((\d+) grupos\)", texto)
    if m:
        d["grupos"] = int(m.group(1))
    for m in re.finditer(r"Estado\s+(\d+): (\d+) puntos \(([\d.]+)%\) \| RTT=([\d.]+).{1,3}?([\d.]+)ms", texto):
        d["estados_det"].append((int(m.group(1)), int(m.group(2)), float(m.group(4)), float(m.group(5))))
    return d


def leer_segmentos(run_dir):
    """Devuelve (filas, fronteras_en_segundos) del segmentos_caracterizados_*.csv de la corrida."""
    cand = [f for f in sorted(os.listdir(run_dir))
            if f.startswith("segmentos_caracterizados") and "agrupados" not in f and f.endswith(".csv")]
    if not cand:
        return [], []
    filas = []
    with io.open(os.path.join(run_dir, cand[0]), "r", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            filas.append(r)
    fronteras, vistos = [], set()
    for r in filas:
        sid = r.get("seq_id")
        if sid in vistos:
            fronteras.append(a_seg(r["start_ts"]))     # inicio de segmento que no es el primero de su secuencia
        vistos.add(sid)
    return filas, sorted(fronteras)


def resumen_segmentos(filas):
    tot = deg = nseg_deg = 0
    for r in filas:
        try:
            n = int(float(r.get("n_ciclos") or 0))
        except ValueError:
            n = 0
        tot += n
        tipo = (r.get("tipo_degradacion") or "")
        if tipo.startswith("degradacion") and "sostenida" in tipo:
            deg += n
            nseg_deg += 1
    return {"n_segmentos": len(filas), "pct_deg": (100.0 * deg / tot) if tot else None, "n_seg_deg": nseg_deg}


def f1_fronteras(a, b, tol):
    """F1 entre dos conjuntos de instantes (segundos), emparejamiento uno a uno dentro de +-tol."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    usados, vp = set(), 0
    for x in a:
        mejor, mj = None, None
        for j, y in enumerate(b):
            if j in usados:
                continue
            dd = abs(x - y)
            if dd <= tol and (mejor is None or dd < mejor):
                mejor, mj = dd, j
        if mj is not None:
            usados.add(mj)
            vp += 1
    p, r = vp / float(len(b)), vp / float(len(a))
    return (2 * p * r / (p + r)) if (p + r) else 0.0


# ============================================================================
# EJECUCION
# ============================================================================
def comando(args, cfg, run_dir):
    c = [args.python, args.script,
         "--crudo", args.crudo, "--bloques-csv", args.bloques_csv, "--ciclos-excluidos", args.ciclos_excluidos,
         "--interval", "%g" % args.interval, "--output-dir", run_dir,
         "--gap-minutes", "%g" % args.gap_minutes, "--K", str(args.K), "--nlap", str(args.nlap),
         "--min-dwell", str(args.min_dwell),
         "--sF", "%g" % cfg["sF"], "--hmm-kappa", "%g" % cfg["kappa"],
         "--percentil-outliers", "%g" % cfg["pct"], "--seed", str(cfg["seed"]),
         "--bnpy-outdir", os.path.join(run_dir, "bnpy")]
    if args.fusion_auto:
        c += ["--fusion-auto", "--umbral-db", "%g" % args.umbral_db]
    if args.extra:
        c += shlex.split(args.extra)
    return c


def correr(args, cfg):
    tag = "sF{}_k{}_p{}_s{}".format(etiqueta_val(cfg["sF"]), etiqueta_val(cfg["kappa"]),
                                      etiqueta_val(cfg["pct"]), cfg["seed"])
    run_dir = os.path.join(args.output_dir, "run_" + tag)
    salida = os.path.join(run_dir, "salida.txt")
    if args.reanudar and os.path.exists(salida):
        with io.open(salida, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        if "COMPLETADA EXITOSAMENTE" in txt:
            return tag, run_dir, txt, 0.0, True
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    env = dict(os.environ)
    env["BNPYOUTDIR"] = os.path.join(run_dir, "bnpy")
    t0 = time.time()
    try:
        p = subprocess.Popen(comando(args, cfg, run_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        out = p.communicate()[0]
    except OSError as e:
        out = ("[ERROR] no se pudo lanzar el script: %s\n" % e).encode("utf-8")
    dt = time.time() - t0
    with open(salida, "wb") as f:
        f.write(out)
    return tag, run_dir, out.decode("utf-8", "replace"), dt, False


def fnum(x, nd=1):
    return "-" if x is None else ("%." + str(nd) + "f") % x


def main():
    ap = argparse.ArgumentParser(description="Barrido de flags del script HDP-HMM sobre una serie real")
    ap.add_argument("--script", required=True, help="Script HDP-HMM (p. ej. hdphmm_ripe_atlas_bnpy_f.py)")
    ap.add_argument("--python", default="python", help="Interprete que ejecuta el script HDP-HMM (default: python)")
    ap.add_argument("--crudo", required=True)
    ap.add_argument("--bloques-csv", required=True)
    ap.add_argument("--ciclos-excluidos", required=True)
    ap.add_argument("--interval", type=float, required=True)
    ap.add_argument("--output-dir", default="barrido_hdphmm")
    ap.add_argument("--gap-minutes", type=float, default=90.0)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--nlap", type=int, default=300)
    ap.add_argument("--min-dwell", type=int, default=4)
    ap.add_argument("--fusion-auto", action="store_true", help="Activa la fusion automatica en todas las corridas")
    ap.add_argument("--umbral-db", type=float, default=0.5, help="Solo con --fusion-auto")
    ap.add_argument("--extra", default="", help="Otros flags del script HDP-HMM, entre comillas (p. ej. \"--trans-alpha 0.5\")")
    ap.add_argument("--modo", choices=["uno", "rejilla"], default="uno")
    ap.add_argument("--sF", default=None, help="Lista separada por comas (si se omite, se pregunta)")
    ap.add_argument("--hmm-kappa", default=None)
    ap.add_argument("--percentil-outliers", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--tol-ciclos", type=float, default=2.0, help="Tolerancia de fronteras para la estabilidad (default 2 ciclos)")
    ap.add_argument("--reanudar", action="store_true", help="Saltar corridas ya completadas")
    ap.add_argument("--si", action="store_true", help="No pedir confirmacion ni valores: usa los pasados por flags o los defectos")
    a = ap.parse_args()

    print("=" * 74)
    print("BARRIDO DEL HDP-HMM (el primer valor de cada lista es la linea base)")
    print("=" * 74)
    sf = [float(x) for x in a.sF.split(",")] if a.sF else pedir("Valores de --sF", "20,5,1", float, a.si)
    ka = [float(x) for x in a.hmm_kappa.split(",")] if a.hmm_kappa else pedir("Valores de --hmm-kappa", "1000,3000,6000", float, a.si)
    pc = [float(x) for x in a.percentil_outliers.split(",")] if a.percentil_outliers else pedir(
        "Valores de --percentil-outliers (0 = sin filtro)", "0,98", float, a.si)
    se = [int(x) for x in a.seeds.split(",")] if a.seeds else pedir("Semillas (--seed)", "0,1,2", int, a.si)

    base = (sf[0], ka[0], pc[0])
    if a.modo == "rejilla":
        cfgs = list(itertools.product(sf, ka, pc))
    else:
        cfgs = [base] + [(v, base[1], base[2]) for v in sf[1:]] + [(base[0], v, base[2]) for v in ka[1:]] + \
               [(base[0], base[1], v) for v in pc[1:]]
    vistos, cfgs_u = set(), []
    for c in cfgs:
        if c not in vistos:
            vistos.add(c)
            cfgs_u.append(c)
    corridas = [{"sF": c[0], "kappa": c[1], "pct": c[2], "seed": s} for c in cfgs_u for s in se]
    print("\nModo {} | configuraciones: {} | semillas: {} | corridas: {}".format(a.modo, len(cfgs_u), len(se), len(corridas)))
    print("Linea base: sF={:g}, kappa={:g}, percentil={:g}".format(*base))
    print("Fusion: {}".format("--fusion-auto (umbral-db {:g})".format(a.umbral_db) if a.fusion_auto else "sin fusion"))
    print("Salida: {}".format(a.output_dir))
    if not a.si:
        try:
            r = _input("Lanzar? [S/n]: ").strip().lower()
        except EOFError:
            r = "s"
        if r in ("n", "no"):
            print("Cancelado.")
            return
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)

    # ---- corridas
    res = []
    t_ini = time.time()
    for i, cfg in enumerate(corridas, 1):
        tag, run_dir, txt, dt, salto = correr(a, cfg)
        d = parsear_salida(txt)
        filas, fronteras = leer_segmentos(run_dir)
        rs = resumen_segmentos(filas)
        d.update(rs)
        sd_max = max([e[3] for e in d["estados_det"]]) if d["estados_det"] else None
        fila = {"tag": tag, "cfg": cfg, "cfgt": (cfg["sF"], cfg["kappa"], cfg["pct"]), "dir": run_dir, "d": d, "fronteras": fronteras, "tiempo": dt, "sd_max": sd_max}
        res.append(fila)
        estado = "reanudada" if salto else ("OK" if d["completo"] else "FALLO")
        print("[{}/{}] {:<34s} {:<9s} estados {:>2s} | macro {:>2s} | cps {:>4s} | grupos {:>4s} | {:>5.0f} s".format(
            i, len(corridas), tag, estado, str(d["estados"] if d["estados"] is not None else "-"),
            str(d["macro"] if d["macro"] is not None else "-"), str(d["cps"] if d["cps"] is not None else "-"),
            str(d["grupos"] if d["grupos"] is not None else "-"), dt))
        if not d["completo"] and not salto:
            print("     (ver {})".format(os.path.join(run_dir, "salida.txt")))
        if i == 1 and not salto and len(corridas) > 1:
            print("     Tiempo estimado restante: {:.0f} min".format(dt * (len(corridas) - 1) / 60.0))

    # ---- tabla por corrida
    tol_s = a.tol_ciclos * a.interval + 5.0
    L = []
    P = L.append
    P("=" * 110)
    P("RESUMEN POR CORRIDA")
    P("=" * 110)
    P("{:<28s}{:>8s}{:>7s}{:>7s}{:>8s}{:>9s}{:>8s}{:>9s}{:>10s}".format(
        "corrida", "estados", "macro", "cps", "grupos", "segm.", "%deg", "sd_max", "tiempo_s"))
    for f in res:
        d = f["d"]
        P("{:<28s}{:>8s}{:>7s}{:>7s}{:>8s}{:>9s}{:>8s}{:>9s}{:>10.0f}".format(
            f["tag"], str(d["estados"] if d["estados"] is not None else "-"),
            str(d["macro"] if d["macro"] is not None else "-"), str(d["cps"] if d["cps"] is not None else "-"),
            str(d["grupos"] if d["grupos"] is not None else "-"), str(d.get("n_segmentos", "-")),
            fnum(d.get("pct_deg")), fnum(f["sd_max"]), f["tiempo"]))
    # cps/dia: se calcula con la duracion cubierta por los segmentos de la primera corrida valida
    dias = None
    for f in res:
        fil, _ = leer_segmentos(f["dir"])
        if fil:
            try:
                tot_ciclos = sum(int(float(r["n_ciclos"])) for r in fil)
                dias = tot_ciclos * a.interval / 86400.0
            except (ValueError, KeyError):
                dias = None
            break

    # ---- estabilidad por configuracion
    P("")
    P("=" * 110)
    P("ESTABILIDAD POR CONFIGURACION (sobre las semillas {})".format(",".join(str(s) for s in se)))
    P("=" * 110)
    P("{:<28s}{:>10s}{:>18s}{:>14s}{:>8s}{:>14s}".format(
        "configuracion", "estados", "puntos de cambio", "cps/dia", "%deg", "acuerdo F1"))
    filas_csv = []
    for c in cfgs_u:
        gr = [f for f in res if f["cfgt"] == c and f["d"]["completo"]]
        etiq = "sF={:g} k={:g} p={:g}".format(*c) + ("  <base>" if c == base else "")
        if not gr:
            P("{:<28s}{:>10s}".format(etiq, "sin corridas OK"))
            continue
        st = [f["d"]["estados"] for f in gr if f["d"]["estados"] is not None]
        cp = [f["d"]["cps"] for f in gr if f["d"]["cps"] is not None]
        pdg = [f["d"]["pct_deg"] for f in gr if f["d"].get("pct_deg") is not None]
        f1s = []
        for x, y in itertools.combinations(gr, 2):
            f1s.append(f1_fronteras(x["fronteras"], y["fronteras"], tol_s))
        rng = lambda v: ("{}-{}".format(min(v), max(v)) if min(v) != max(v) else str(min(v))) if v else "-"   # noqa: E731
        cpd = ("{:.1f}-{:.1f}".format(min(cp) / dias, max(cp) / dias) if (cp and dias) else "-")
        P("{:<28s}{:>10s}{:>18s}{:>14s}{:>8s}{:>14s}".format(
            etiq, rng(st), rng(cp), cpd, fnum(sum(pdg) / len(pdg) if pdg else None),
            fnum(sum(f1s) / len(f1s), 2) if f1s else "(1 semilla)"))
        filas_csv.append([c[0], c[1], c[2], len(gr), rng(st), rng(cp), cpd, fnum(sum(pdg) / len(pdg) if pdg else None),
                          fnum(sum(f1s) / len(f1s), 2) if f1s else ""])
    P("")
    P("Lectura: 'acuerdo F1' = coincidencia media de las fronteras de segmento entre pares de semillas (tolerancia {:g} ciclos).".format(a.tol_ciclos))
    P("Cercano a 1 y rangos estrechos de estados y puntos de cambio = configuracion estable. %deg = ciclos en segmentos "
      "'degradacion sostenida' (umbrales fijos del script: 2 y 5 ms sobre el P25).")
    if dias:
        P("Dias cubiertos por los segmentos: {:.1f}".format(dias))
    P("Sin verdad de campo: el barrido mide estabilidad e interpretabilidad, no exactitud. Revisar el visor HTML de las "
      "configuraciones candidatas.")
    texto = "\n".join(L)
    print("\n" + texto)

    with io.open(os.path.join(a.output_dir, "barrido_resumen.txt"), "w", encoding="utf-8") as fo:
        fo.write(texto + u"\n")
    with open(os.path.join(a.output_dir, "barrido_corridas.csv"), "w") as fo:
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(["corrida", "sF", "hmm_kappa", "percentil_outliers", "seed", "completa", "estados", "k", "macro_estados",
                    "puntos_de_cambio", "grupos", "segmentos", "pct_ciclos_degradacion", "sd_max_estado", "tiempo_s"])
        for f in res:
            d = f["d"]
            w.writerow([f["tag"], f["cfg"]["sF"], f["cfg"]["kappa"], f["cfg"]["pct"], f["cfg"]["seed"],
                        int(d["completo"]), d["estados"], d["k"], d["macro"], d["cps"], d["grupos"], d.get("n_segmentos"),
                        fnum(d.get("pct_deg"), 2), fnum(f["sd_max"], 2), "%.0f" % f["tiempo"]])
    with open(os.path.join(a.output_dir, "barrido_estabilidad.csv"), "w") as fo:
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(["sF", "hmm_kappa", "percentil_outliers", "corridas_ok", "estados", "puntos_de_cambio", "cps_por_dia",
                    "pct_degradacion", "acuerdo_f1"])
        w.writerows(filas_csv)
    print("\nArchivos: {0}/barrido_resumen.txt, barrido_corridas.csv, barrido_estabilidad.csv | tiempo total {1:.1f} min".format(
        a.output_dir, (time.time() - t_ini) / 60.0))


if __name__ == "__main__":
    main()
