# -*- coding: utf-8 -*-
"""
=============================================================================
utilidad_tstr_holdout.py
TSTR con HOLDOUT TEMPORAL REAL: el sintetico se genera solo con la parte de entrenamiento de la serie real y se evalua
en la parte de prueba, que el generador nunca vio.
=============================================================================
Metodologia (los 4 pasos):
  1. SEPARACION DE DATOS REALES. La serie real del bloque principal se divide en el tiempo: los primeros --frac-train
     ciclos son el subconjunto de ENTRENAMIENTO y el resto, tras una purga de --purga-h horas, es la PRUEBA (holdout).
     Division temporal (no al azar): la serie esta autocorrelacionada y un reparto al azar filtraria informacion.
  2. GENERACION SINTETICA. Se recorta el historial y el bloque principal al entrenamiento (historial_train.csv y
     bloque_principal_train.json, escritos en --output-dir) y se lanza generar_bootstrap.py solo con esos archivos,
     una vez por semilla. Ningun ciclo de la prueba llega al generador.
  3. ENTRENAMIENTO. El clasificador se entrena UNICAMENTE con las series sinteticas (etiqueta = verdad exacta de la
     inyeccion: delta_ms >= --umbral-ms).
  4. EVALUACION EN REAL. Se mide en el holdout real: accuracy, precision, recall, F1, AUROC y AP (precision media, que
     es el mAP de una tarea binaria), y por episodio: recall y falsas alarmas por dia.
     R2 no aplica: la tarea es de clasificacion.

Tarea: clasificar cada ciclo como "degradacion sostenida" (1) o no (0) con caracteristicas causales (solo ciclos
pasados) sobre z = RTT - referencia (mediana, P10, P90, MAD y fraccion de z > umbral en cada ventana de --ventanas-h).
Etiqueta real: PLATEADA, la del HDP-HMM (segmentos 'degradacion_*_sostenida'); no es verdad de campo.

Referencias sobre el MISMO holdout (comparables entre si):
  * TSTR        : entrenado solo con sintetico (generado del entrenamiento real).   <- el experimento
  * TRTR        : entrenado con el entrenamiento real (etiquetas plateadas).         <- techo realista
  * Regla simple: mediana de ~1 h sobre la referencia >= --umbral-ms, sin aprendizaje.
  Brecha de utilidad = TRTR - TSTR.

Referencia de nivel (z = RTT - ref): cada serie sintetica usa su propio percentil (--ref-percentil). El real de
entrenamiento usa el de su tramo. El holdout usa por defecto el del ENTRENAMIENTO (--ref-test train): asi la
caracteristica no incorpora ninguna estadistica de la prueba. Con --ref-test test se usa el de la propia prueba.

Uso:
  python3 utilidad_tstr_holdout.py \\
      --real-historial historial_traceroute_measurement126172435_probe1005718.csv \\
      --real-bloque-principal verificacion_1005718/bloque_principal_measurement126172435_probe1005718.json \\
      --real-excluidos verificacion_1005718/ciclos_excluidos_measurement126172435_probe1005718.csv \\
      --real-segmentos resultados_hdphmm_real/segmentos_caracterizados_measurement126172435_probe1005718_auto.csv \\
      --generador generar_bootstrap.py --gen-args "--modo crudo --largo-bloque-h 12" --semillas 1,2,3 \\
      --frac-train 0.5 --output-dir tstr_holdout_1005718 --etiqueta p1005718

Otras opciones: --corte "AAAA-MM-DD HH:MM:SS" (corte explicito), --sinteticos (usar series ya generadas del
entrenamiento, sin lanzar el generador), --solo-preparar (escribe los archivos de entrenamiento y sale), --reanudar
(reutiliza sinteticos ya generados con el mismo corte).

Solo usa numpy (la regresion logistica es propia). --modelo gbm usa scikit-learn si esta instalado.
No modifica ningun otro script.
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import csv
import json
import glob
import shlex
import argparse
import subprocess
from datetime import datetime, timedelta

import numpy as np

FMT = "%Y-%m-%d %H:%M:%S"
EPOCH = datetime(1970, 1, 1)


def a_seg(ts):
    return (datetime.strptime(ts.strip()[:19], FMT) - EPOCH).total_seconds()


def de_seg(s):
    return (EPOCH + timedelta(seconds=float(s))).strftime(FMT)


def a_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ============================================================================
# CARGA
# ============================================================================
def cargar_sintetica(path, umbral_ms):
    """verdad_ciclos_*.csv -> (t, y, etiqueta). Solo ciclos con destino y sin hueco."""
    t, y, lab = [], [], []
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            if (r.get("hueco") or "0").strip() not in ("", "0"):
                continue
            if (r.get("sin_destino") or "0").strip() not in ("", "0"):
                continue
            v = a_float(r.get("rtt_ultimo_ms"))
            if v is None:
                continue
            d = a_float(r.get("delta_ms")) or 0.0
            t.append(a_seg(r["timestamp"]))
            y.append(v)
            lab.append(1 if d >= umbral_ms else 0)
    return np.array(t), np.array(y), np.array(lab, dtype=int)


def cargar_real(historial, bloque_json, excluidos_csv):
    """Serie real del bloque principal: RTT del ultimo salto por ciclo. Devuelve (t, ts_txt, y, interval, bp)."""
    with open(bloque_json) as f:
        bp = json.load(f)
    el = bp["elegido"]
    n_long = int(el["longitud"])
    interval = float(bp["cadencia_s"])
    t0, t1 = a_seg(el["inicio"]), a_seg(el["fin"])
    excl = set()
    if excluidos_csv and os.path.exists(excluidos_csv):
        with open(excluidos_csv, "r") as f:
            for r in csv.DictReader(f):
                if (r.get("ciclo") or "").strip():
                    excl.add(r["ciclo"].strip())
    ult = {}
    with open(historial, "r") as f:
        for r in csv.DictReader(f):
            ts = (r.get("timestamp") or "").strip()
            if not ts or ts in excl:
                continue
            try:
                hop = int(r["hop"])
            except (ValueError, TypeError, KeyError):
                continue
            if hop != n_long or not (r.get("ip") or "").strip():
                continue
            v = a_float(r.get("rtt_ms"))
            if v is None:
                continue
            tt = a_seg(ts)
            if t0 <= tt <= t1:
                ult[ts] = v
    orden = sorted(ult)
    t = np.array([a_seg(ts) for ts in orden])
    y = np.array([ult[ts] for ts in orden])
    return t, orden, y, interval, bp


def etiquetar_real(t, segmentos_csv):
    """Etiqueta plateada por ciclo desde segmentos_caracterizados_*.csv del HDP-HMM."""
    lab = np.zeros(len(t), dtype=int)
    n_seg = n_deg = 0
    with open(segmentos_csv, "r") as f:
        for r in csv.DictReader(f):
            n_seg += 1
            tipo = (r.get("tipo_degradacion") or "").strip()
            if tipo.startswith("degradacion") and "sostenida" in tipo:
                n_deg += 1
                i0 = np.searchsorted(t, a_seg(r["start_ts"]), side="left")
                i1 = np.searchsorted(t, a_seg(r["end_ts"]), side="right")
                lab[i0:i1] = 1
    return lab, n_seg, n_deg


# ============================================================================
# PASO 1 y 2: DIVISION TEMPORAL Y GENERACION SOLO CON EL ENTRENAMIENTO
# ============================================================================
def dividir(t, frac, corte, purga_s):
    """Devuelve (n_train, t_corte, i_test0): train = t[:n_train]; prueba = t[i_test0:] (tras la purga)."""
    if corte:
        t_corte = a_seg(corte)
        n_train = int(np.searchsorted(t, t_corte, side="right"))
    else:
        n_train = int(round(len(t) * frac))
    if n_train < 1 or n_train >= len(t):
        sys.exit("[ERROR] el corte deja un lado vacio.")
    t_corte = float(t[n_train - 1])
    i_test0 = int(np.searchsorted(t, t_corte + purga_s, side="left"))
    return n_train, t_corte, i_test0


def preparar_entrenamiento(historial, bp, t_corte, outdir):
    """Escribe historial_train.csv (filas con timestamp <= corte) y bloque_principal_train.json (fin = corte)."""
    os.makedirs(outdir, exist_ok=True)
    p_h = os.path.join(outdir, "historial_train.csv")
    n_filas, ultimo = 0, None
    with open(historial, "r", newline="") as fi, open(p_h, "w", newline="") as fo:
        rd = csv.reader(fi)
        wr = csv.writer(fo)
        cab = next(rd)
        wr.writerow(cab)
        i_ts = cab.index("timestamp")
        for row in rd:
            if len(row) <= i_ts or not row[i_ts].strip():
                continue
            s = a_seg(row[i_ts])
            if s <= t_corte:
                wr.writerow(row)
                n_filas += 1
                ultimo = s if ultimo is None else max(ultimo, s)
    bp2 = json.loads(json.dumps(bp))
    el = bp2["elegido"]
    el["fin"] = de_seg(t_corte)
    el["duracion_h"] = round((t_corte - a_seg(el["inicio"])) / 3600.0, 1)
    bp2["recortado_para_holdout"] = True
    p_j = os.path.join(outdir, "bloque_principal_train.json")
    with open(p_j, "w") as f:
        json.dump(bp2, f, indent=2)
    return p_h, p_j, n_filas, ultimo


def lanzar_generador(args, p_hist, p_json, semilla, carpeta):
    gen = args.generador
    if not os.path.isabs(gen) and not os.path.exists(gen):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), gen)
        if os.path.exists(alt):
            gen = alt
    if not os.path.exists(gen):
        sys.exit("[ERROR] no encuentro el generador: {}".format(args.generador))
    cmd = [sys.executable, gen, "--historial", p_hist, "--bloque-principal", p_json,
           "--output-dir", carpeta, "--seed", str(semilla)]
    if args.real_excluidos:
        cmd += ["--ciclos-excluidos", args.real_excluidos]
    cmd += shlex.split(args.gen_args)
    os.makedirs(carpeta, exist_ok=True)
    log = os.path.join(carpeta, "salida_generador.log")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        with open(log) as lf:
            cola = lf.read().splitlines()[-8:]
        sys.exit("[ERROR] el generador fallo (semilla {}):\n  {}\n  Log completo: {}".format(semilla, "\n  ".join(cola), log))
    return " ".join(cmd)


# ============================================================================
# CARACTERISTICAS (causales)
# ============================================================================
def ventanas_causales(z, w):
    pad = np.concatenate([np.full(w - 1, z[0]), z])
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(pad, w)
    except ImportError:
        idx = np.arange(len(z))[:, None] + np.arange(w)[None, :]
        return pad[idx]


def caracteristicas(y, interval, ventanas_h, ref, umbral_ms):
    z = y - ref
    cols, nombres = [], []
    for h in ventanas_h:
        w = max(3, int(round(h * 3600.0 / interval)))
        W = ventanas_causales(z, w)
        q = np.percentile(W, [10, 50, 90], axis=1)
        mad = np.median(np.abs(W - q[1][:, None]), axis=1) * 1.4826
        frac = (W > umbral_ms).mean(axis=1)
        for nom, col in (("p10", q[0]), ("med", q[1]), ("p90", q[2]), ("mad", mad), ("frac", frac)):
            cols.append(col)
            nombres.append("{}_{:g}h".format(nom, h))
    return np.column_stack(cols), nombres


# ============================================================================
# MODELOS
# ============================================================================
class LogReg(object):
    """Regresion logistica L2 por IRLS (solo numpy)."""

    def __init__(self, lam=1.0, iters=50):
        self.lam, self.iters = lam, iters

    def fit(self, X, y):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Xb = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        w = np.zeros(Xb.shape[1])
        R = self.lam * np.eye(Xb.shape[1])
        R[0, 0] = 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xb.dot(w), -30, 30)))
            g = Xb.T.dot(p - y) + R.dot(w)
            H = Xb.T.dot(Xb * (p * (1 - p) + 1e-6)[:, None]) + R
            paso = np.linalg.solve(H, g)
            w -= paso
            if np.max(np.abs(paso)) < 1e-6:
                break
        self.w = w
        return self

    def proba(self, X):
        Xb = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return 1.0 / (1.0 + np.exp(-np.clip(Xb.dot(self.w), -30, 30)))


class GBM(object):
    def __init__(self, seed):
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError:
            sys.exit("[ERROR] --modelo gbm requiere scikit-learn; use --modelo logreg.")
        self.m = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=seed)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def proba(self, X):
        return self.m.predict_proba(X)[:, 1]


def nuevo_modelo(args):
    return LogReg() if args.modelo == "logreg" else GBM(args.seed)


# ============================================================================
# METRICAS
# ============================================================================
def runs(b):
    out, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j + 1 < n and b[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def precision_media(y, p):
    """Area bajo la curva precision-recall (AP; en tarea binaria es el mAP)."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * ys) / ys.sum())


def auroc(y, p):
    """AUROC por rangos (Mann-Whitney), con empates promediados."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(p, kind="mergesort")
    ps = p[o]
    r = np.arange(1, len(p) + 1, dtype=float)
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    rangos = np.empty(len(p))
    rangos[o] = r
    return float((rangos[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def metricas(y, p, dias, args):
    pred = p >= args.umbral_prob
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / float(tp + fp) if (tp + fp) else float("nan")
    rec = tp / float(tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec == prec and rec == rec and (prec + rec) > 0) else (0.0 if tp == 0 else float("nan"))
    pred_f = np.zeros(len(pred), dtype=bool)
    for a, b in runs(pred):
        if b - a + 1 >= args.min_racha:
            pred_f[a:b + 1] = True
    ver = [(a, b) for a, b in runs(y == 1) if b - a + 1 >= args.min_racha]
    det = sum(1 for a, b in ver if pred_f[a:b + 1].mean() >= args.cobertura_ep)
    fa = sum(1 for a, b in runs(pred_f) if not (y[a:b + 1] == 1).any())
    return {"prevalencia": float(y.mean()), "accuracy": (tp + tn) / float(len(y)), "precision": prec, "recall": rec,
            "f1": f1, "auroc": auroc(y, p), "ap": precision_media(y, p),
            "ep_verdad": len(ver), "ep_detectados": det,
            "ep_recall": (det / float(len(ver))) if ver else float("nan"),
            "ep_falsas_alarmas": fa, "ep_fa_por_dia": fa / dias if dias > 0 else float("nan")}


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="TSTR con holdout temporal real: sintetico generado solo del entrenamiento")
    ap.add_argument("--real-historial", required=True)
    ap.add_argument("--real-bloque-principal", required=True)
    ap.add_argument("--real-excluidos", default=None)
    ap.add_argument("--real-segmentos", required=True, help="segmentos_caracterizados_*.csv del HDP-HMM sobre la serie REAL")
    ap.add_argument("--frac-train", type=float, default=0.5, help="Fraccion inicial de ciclos para entrenamiento (default 0.5)")
    ap.add_argument("--corte", default=None, help="Corte explicito 'AAAA-MM-DD HH:MM:SS' (reemplaza --frac-train)")
    ap.add_argument("--purga-h", type=float, default=12.0,
                    help="Horas descartadas al inicio de la prueba (default 12; se sube a la ventana mas larga si es menor)")
    ap.add_argument("--generador", default="generar_bootstrap.py")
    ap.add_argument("--gen-args", default="--modo crudo --largo-bloque-h 12",
                    help="Argumentos extra para el generador, entre comillas (default: '--modo crudo --largo-bloque-h 12')")
    ap.add_argument("--semillas", default="1,2,3", help="Semillas del generador, separadas por coma (default 1,2,3)")
    ap.add_argument("--sinteticos", nargs="+", default=None,
                    help="verdad_ciclos_*.csv ya generados SOLO con el entrenamiento: no lanza el generador")
    ap.add_argument("--solo-preparar", action="store_true", help="Escribe historial_train.csv y el JSON recortado y sale")
    ap.add_argument("--reanudar", action="store_true", help="Reutiliza sinteticos ya generados con el mismo corte")
    ap.add_argument("--ventanas-h", default="0.5,1,3,12")
    ap.add_argument("--ref-percentil", type=float, default=10.0)
    ap.add_argument("--ref-test", choices=["train", "test"], default="train",
                    help="Referencia de nivel del holdout: percentil del entrenamiento (default) o de la propia prueba")
    ap.add_argument("--umbral-ms", type=float, default=2.0)
    ap.add_argument("--modelo", choices=["logreg", "gbm"], default="logreg")
    ap.add_argument("--umbral-prob", type=float, default=0.5)
    ap.add_argument("--min-racha", type=int, default=5)
    ap.add_argument("--cobertura-ep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-dir", default="tstr_holdout")
    ap.add_argument("--etiqueta", default="")
    a = ap.parse_args()
    np.random.seed(a.seed)
    ventanas = [float(x) for x in a.ventanas_h.split(",") if x.strip()]
    semillas = [s.strip() for s in a.semillas.split(",") if s.strip()]

    # ---- datos reales y paso 1: division temporal
    t, ts_txt, y, interval, bp = cargar_real(a.real_historial, a.real_bloque_principal, a.real_excluidos)
    if len(y) < 1000:
        sys.exit("[ERROR] la serie real tiene solo {} ciclos.".format(len(y)))
    lab, n_seg, n_deg = etiquetar_real(t, a.real_segmentos)
    purga_h = max(a.purga_h, max(ventanas))
    if purga_h > a.purga_h:
        print("[AVISO] purga subida a {:g} h (la ventana causal mas larga) para que las caracteristicas de la prueba "
              "no incluyan ciclos del entrenamiento.".format(purga_h))
    n_tr, t_corte, i_te0 = dividir(t, a.frac_train, a.corte, purga_h * 3600.0)
    assert t[n_tr - 1] < t[i_te0] if i_te0 < len(t) else True
    n_te = len(t) - i_te0
    if n_te < 500:
        sys.exit("[ERROR] la prueba tiene solo {} ciclos: baje --frac-train o --purga-h.".format(n_te))
    y_tr, lab_tr = y[:n_tr], lab[:n_tr]
    y_te, lab_te = y[i_te0:], lab[i_te0:]
    dias_tr, dias_te = n_tr * interval / 86400.0, n_te * interval / 86400.0

    os.makedirs(a.output_dir, exist_ok=True)
    p_h, p_j, n_filas, ultimo = preparar_entrenamiento(a.real_historial, bp, t_corte, a.output_dir)
    assert ultimo is None or ultimo <= t_corte
    corte_txt = de_seg(t_corte)
    print("[1] Division temporal: entrenamiento {} ciclos ({:.1f} d) hasta {} | purga {:g} h | prueba {} ciclos ({:.1f} d) desde {}".format(
        n_tr, dias_tr, corte_txt, purga_h, n_te, dias_te, ts_txt[i_te0]))
    print("    Archivos para el generador: {} ({} filas, ultimo ciclo {}), {}".format(
        p_h, n_filas, de_seg(ultimo) if ultimo else "-", p_j))
    split_path = os.path.join(a.output_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump({"corte": corte_txt, "n_train": n_tr, "prueba_desde": ts_txt[i_te0], "n_test": n_te,
                   "purga_h": purga_h}, f, indent=2)
    if a.solo_preparar:
        print("Modo --solo-preparar: ejecute el generador con --historial {} --bloque-principal {} y luego use --sinteticos."
              .format(p_h, p_j))
        return

    # ---- paso 2: generacion sintetica solo con el entrenamiento
    rutas = []
    cmds = []
    if a.sinteticos:
        rutas = list(a.sinteticos)
        print("[2] Sinteticos entregados con --sinteticos ({}): se asume que fueron generados SOLO con el entrenamiento "
              "(hasta {}).".format(len(rutas), corte_txt))
    else:
        for s in semillas:
            carpeta = os.path.join(a.output_dir, "sint_s{}".format(s))
            previos = sorted(glob.glob(os.path.join(carpeta, "verdad_ciclos_*.csv")))
            reutilizar = False
            if a.reanudar and previos:
                try:
                    with open(os.path.join(carpeta, "corte.txt")) as f:
                        reutilizar = (f.read().strip() == corte_txt)
                except IOError:
                    reutilizar = False
            if not reutilizar:
                print("[2] Generando sintetico (semilla {}) solo con el entrenamiento ...".format(s))
                cmds.append(lanzar_generador(a, p_h, p_j, s, carpeta))
                with open(os.path.join(carpeta, "corte.txt"), "w") as f:
                    f.write(corte_txt + "\n")
            else:
                print("[2] Reutilizando sintetico de la semilla {} (mismo corte).".format(s))
            nuevos = sorted(glob.glob(os.path.join(carpeta, "verdad_ciclos_*.csv")))
            if not nuevos:
                sys.exit("[ERROR] no hay verdad_ciclos_*.csv en {}".format(carpeta))
            rutas.append(nuevos[0])

    # ---- paso 3: entrenamiento solo con sintetico
    sint = []
    for pth in rutas:
        _, y_s, lab_s = cargar_sintetica(pth, a.umbral_ms)
        ref_s = float(np.percentile(y_s, a.ref_percentil))
        Xs, nombres = caracteristicas(y_s, interval, ventanas, ref_s, a.umbral_ms)
        sint.append({"path": pth, "X": Xs, "y": lab_s, "n": len(y_s), "ref": ref_s})
    ref_tr = float(np.percentile(y_tr, a.ref_percentil))
    ref_te = ref_tr if a.ref_test == "train" else float(np.percentile(y_te, a.ref_percentil))
    X_tr = caracteristicas(y_tr, interval, ventanas, ref_tr, a.umbral_ms)[0]
    X_all_te = caracteristicas(y, interval, ventanas, ref_te, a.umbral_ms)[0]   # causal sobre toda la serie
    X_te = X_all_te[i_te0:]
    filas = []
    preds = {}

    def agregar(esc, det, p):
        filas.append((esc, det, metricas(lab_te, p, dias_te, a)))

    Xs_all = np.vstack([s["X"] for s in sint])
    ys_all = np.concatenate([s["y"] for s in sint])
    if ys_all.min() == ys_all.max():
        sys.exit("[ERROR] los sinteticos no tienen ambas clases.")
    p_tstr = nuevo_modelo(a).fit(Xs_all, ys_all).proba(X_te)
    preds["p_tstr"] = p_tstr
    agregar("TSTR", "entrena: {} sintetico(s) del entrenamiento real".format(len(sint)), p_tstr)
    if len(sint) > 1:
        for i, s in enumerate(sint):
            agregar("TSTR (1 semilla)", "semilla {}".format(semillas[i] if (not a.sinteticos and i < len(semillas)) else i + 1),
                    nuevo_modelo(a).fit(s["X"], s["y"]).proba(X_te))
    # ---- TRTR: entrenamiento real -> prueba real
    if lab_tr.min() == lab_tr.max():
        print("[AVISO] el entrenamiento real no tiene ambas clases segun el HDP-HMM: TRTR no se puede calcular.")
    else:
        p_trtr = nuevo_modelo(a).fit(X_tr, lab_tr).proba(X_te)
        preds["p_trtr"] = p_trtr
        agregar("TRTR", "entrena: tramo real de entrenamiento (etiq. plateada)", p_trtr)
    # ---- regla simple
    j = int(np.argmin([abs(h - 1.0) for h in ventanas]))
    k_med = nombres.index("med_{:g}h".format(ventanas[j]))
    regla = (X_te[:, k_med] >= a.umbral_ms).astype(float)
    preds["regla"] = regla
    agregar("Regla simple", "mediana {:g} h sobre la referencia >= {:g} ms".format(ventanas[j], a.umbral_ms), regla)

    # ---- informe
    def f(x, nd=3):
        return "-" if (x is None or x != x) else ("{:." + str(nd) + "f}").format(x)

    mayoria = max(lab_te.mean(), 1 - lab_te.mean())
    m_tstr = filas[0][2]
    L = []
    P = L.append
    P("=" * 100)
    P("TSTR CON HOLDOUT TEMPORAL REAL" + ((" [" + a.etiqueta + "]") if a.etiqueta else ""))
    P("=" * 100)
    P("1. Division: entrenamiento {} ciclos ({:.1f} d) hasta {} | prueba {} ciclos ({:.1f} d) desde {} | purga {:g} h".format(
        n_tr, dias_tr, corte_txt, n_te, dias_te, ts_txt[i_te0], purga_h))
    P("   Referencia de nivel (P{:g}): entrenamiento {:.2f} ms | prueba {:.2f} ms | usada en la prueba: {} ({:.2f} ms){}".format(
        a.ref_percentil, ref_tr, float(np.percentile(y_te, a.ref_percentil)), a.ref_test, ref_te,
        "  <-- deriva de nivel > 1 ms entre tramos" if abs(float(np.percentile(y_te, a.ref_percentil)) - ref_tr) > 1.0 else ""))
    P("   Etiqueta plateada (HDP-HMM, {} segmentos, {} de degradacion sostenida): prevalencia entrenamiento {:.1%} | prueba {:.1%}".format(
        n_seg, n_deg, lab_tr.mean(), lab_te.mean()))
    P("2. Sintetico generado SOLO con el entrenamiento: {} serie(s), {} ciclos c/u aprox., prevalencia {}".format(
        len(sint), "/".join(str(s["n"]) for s in sint), ", ".join("{:.1%}".format(s["y"].mean()) for s in sint)))
    P("3. Modelo: {} entrenado solo con sintetico | {} caracteristicas causales | ventanas (h): {}".format(
        a.modelo, len(nombres), a.ventanas_h))
    P("4. Evaluacion en el holdout real. Umbral de probabilidad {:g}; episodio = racha >= {} ciclos, detectado si se cubre >= {:.0%}".format(
        a.umbral_prob, a.min_racha, a.cobertura_ep))
    P("   Accuracy de referencia (predecir siempre la clase mayoritaria): {:.3f}".format(mayoria))
    P("")
    P("{:<17s}{:<43s}{:>6s}{:>6s}{:>6s}{:>6s}{:>7s}{:>6s}{:>10s}{:>7s}".format(
        "escenario", "detalle", "acc", "prec", "rec", "F1", "AUROC", "AP", "episodios", "FA/dia"))
    for esc, det, m in filas:
        P("{:<17s}{:<43s}{:>6s}{:>6s}{:>6s}{:>6s}{:>7s}{:>6s}{:>10s}{:>7s}".format(
            esc, det[:42], f(m["accuracy"], 2), f(m["precision"], 2), f(m["recall"], 2), f(m["f1"], 2), f(m["auroc"], 2),
            f(m["ap"], 2), "{}/{}".format(m["ep_detectados"], m["ep_verdad"]), f(m["ep_fa_por_dia"], 2)))
    P("")
    d_trtr = [m for e, _, m in filas if e == "TRTR"]
    if d_trtr:
        P("Brecha de utilidad (TRTR - TSTR): F1 {} | AUROC {} | AP {} | recall de episodios {}".format(
            f(d_trtr[0]["f1"] - m_tstr["f1"]), f(d_trtr[0]["auroc"] - m_tstr["auroc"]), f(d_trtr[0]["ap"] - m_tstr["ap"]),
            f(d_trtr[0]["ep_recall"] - m_tstr["ep_recall"])))
        P("  (positivo = entrenar con real rinde mas; cercano a 0 = el sintetico es util; negativo = el sintetico rinde mas)")
    reg = [m for e, _, m in filas if e == "Regla simple"][0]
    P("Aporte del aprendizaje sobre la regla simple (TSTR - regla): F1 {} | AP {}".format(
        f(m_tstr["f1"] - reg["f1"]), f(m_tstr["ap"] - reg["ap"])))
    P("")
    if m_tstr["ep_verdad"] < 10:
        P("[AVISO] el holdout tiene solo {} episodios de degradacion: las metricas por ciclo y por episodio son inestables; "
          "no se informa intervalo de confianza.".format(m_tstr["ep_verdad"]))
        P("")
    P("--- Limitaciones ---")
    P("* Sin fuga del generador: el historial y el bloque recortados terminan en {} y la prueba empieza {:g} h despues.".format(
        corte_txt, purga_h))
    P("* La etiqueta de la prueba es PLATEADA (HDP-HMM corrido sobre la serie completa, incluida la prueba): mide el acuerdo "
      "con ese detector, no la exactitud de campo.")
    P("* Una sola sonda y un solo corte temporal: no demuestra generalizacion a otras sondas ni a otros periodos.")
    P("* Accuracy depende de la prevalencia; se informa junto con la clase mayoritaria. AUROC y AP no dependen del umbral.")
    P("* La prevalencia sintetica difiere de la real; el umbral de probabilidad no se recalibra.")
    P("=" * 100)
    texto = "\n".join(L)
    print(texto)

    suf = ("_" + a.etiqueta) if a.etiqueta else ""
    with open(os.path.join(a.output_dir, "tstr_holdout_resumen" + suf + ".txt"), "w") as fo:
        fo.write(texto + "\n")
        if cmds:
            fo.write("\nComando del generador (ultima semilla):\n  " + cmds[-1] + "\n")
    with open(os.path.join(a.output_dir, "tstr_holdout_resumen" + suf + ".csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["escenario", "detalle", "prevalencia", "accuracy", "precision", "recall", "f1", "auroc", "ap",
                    "ep_verdad", "ep_detectados", "ep_recall", "ep_falsas_alarmas", "ep_fa_por_dia"])
        for esc, det, m in filas:
            w.writerow([esc, det, f(m["prevalencia"], 4), f(m["accuracy"], 4), f(m["precision"], 4), f(m["recall"], 4),
                        f(m["f1"], 4), f(m["auroc"], 4), f(m["ap"], 4), m["ep_verdad"], m["ep_detectados"],
                        f(m["ep_recall"], 4), m["ep_falsas_alarmas"], f(m["ep_fa_por_dia"], 4)])
    with open(os.path.join(a.output_dir, "predicciones_prueba" + suf + ".csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        cols = ["p_tstr", "p_trtr", "regla"]
        w.writerow(["timestamp", "rtt_ms", "etiqueta_plateada"] + cols)
        for i in range(n_te):
            w.writerow([ts_txt[i_te0 + i], "{:.3f}".format(y_te[i]), int(lab_te[i])] +
                       [("{:.4f}".format(preds[c][i]) if c in preds else "") for c in cols])
    print("Archivos escritos en: {}".format(a.output_dir))


if __name__ == "__main__":
    main()
