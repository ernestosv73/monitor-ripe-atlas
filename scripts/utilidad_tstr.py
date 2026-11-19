# -*- coding: utf-8 -*-
"""
=============================================================================
utilidad_tstr.py
Experimento minimo de UTILIDAD del dato sintetico para aprendizaje supervisado (TSTR: Train on Synthetic, Test on Real).
=============================================================================
Pregunta: un detector supervisado de degradacion entrenado SOLO con series sinteticas (generar_bootstrap.py, modo crudo),
funciona sobre la serie REAL de la sonda de origen, comparado con uno entrenado con datos reales?

Tarea (por ciclo): clasificar "ciclo en degradacion sostenida" (1) vs "no" (0) a partir del RTT del ultimo salto.
  * Etiqueta sintetica: verdad exacta (delta_ms >= --umbral-ms, de verdad_ciclos_*.csv).
  * Etiqueta real: VERDAD PLATEADA, la que asigna el HDP-HMM (segmentos_caracterizados_*.csv: tipo 'degradacion_*_sostenida').
    No es una verdad de campo: la medicion en real es relativa al HDP-HMM (limitacion principal del experimento).
Caracteristicas: ventanas causales (solo ciclos pasados) sobre z = RTT - referencia, con referencia = percentil bajo (default P10)
de cada serie: mediana, P10, P90, MAD y fraccion de ciclos con z > umbral, para cada ventana de --ventanas-h.

Escenarios (todos se informan sobre la MISMA serie real completa, para que sean comparables):
  1. TSTR       : entrena con todas las series sinteticas, prueba en la real.      (y una fila por semilla sintetica)
  2. TRTR       : entrena y prueba en real con validacion cruzada por bloques contiguos (--folds, con purga).
  3. TSTS       : entrena con las sinteticas menos una y prueba en la retenida (techo en el dominio sintetico).
  4. Regla      : sin aprendizaje, mediana de la ventana de ~1 h sobre la referencia >= --umbral-ms (linea base trivial).
Metricas: precision, recall, F1 (umbral de probabilidad --umbral-prob), precision media (area PR, independiente del umbral),
y por episodio: recall de episodios y falsas alarmas por dia. Brecha de utilidad = TRTR - TSTR (cuanto menor, mejor el sintetico).

Uso:
  python3 utilidad_tstr.py \\
      --sinteticos crudo12_s1/verdad_ciclos_*.csv crudo12_s2/verdad_ciclos_*.csv crudo12_s3/verdad_ciclos_*.csv \\
      --real-historial historial_traceroute_measurement126172435_probe1005718.csv \\
      --real-bloque-principal verificacion_1005718/bloque_principal_measurement126172435_probe1005718.json \\
      --real-excluidos verificacion_1005718/ciclos_excluidos_measurement126172435_probe1005718.csv \\
      --real-segmentos resultados_hdphmm_real/segmentos_caracterizados_measurement126172435_probe1005718_auto.csv \\
      --output-dir tstr --etiqueta crudo12

Solo usa numpy (la regresion logistica es propia). --modelo gbm usa scikit-learn si esta instalado.
No modifica ningun otro script.
=============================================================================
"""
from __future__ import print_function, division
import os
import sys
import csv
import json
import argparse
from datetime import datetime

import numpy as np

FMT = "%Y-%m-%d %H:%M:%S"
EPOCH = datetime(1970, 1, 1)


def a_seg(ts):
    return (datetime.strptime(ts.strip()[:19], FMT) - EPOCH).total_seconds()


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


def cargar_real(historial, bloque_json, excluidos_csv, segmentos_csv):
    """Serie real del bloque principal (RTT del ultimo salto por ciclo) + etiqueta plateada del HDP-HMM."""
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
    return t, y, lab, interval, n_seg, n_deg


# ============================================================================
# CARACTERISTICAS (causales)
# ============================================================================
def ventanas_causales(z, w):
    """Matriz (n, w): fila i = z[i-w+1..i]; el inicio se rellena con el primer valor."""
    pad = np.concatenate([np.full(w - 1, z[0]), z])
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(pad, w)
    except ImportError:
        idx = np.arange(len(z))[:, None] + np.arange(w)[None, :]
        return pad[idx]


def caracteristicas(y, interval, ventanas_h, ref_pct, umbral_ms):
    ref = float(np.percentile(y, ref_pct))
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
    return np.column_stack(cols), nombres, ref


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
    """Tramos contiguos de True: lista de (i0, i1) inclusivos."""
    out, i = [], 0
    n = len(b)
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
    """Area bajo la curva precision-recall (precision media)."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p)
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * ys) / ys.sum())


def metricas(y, p, dias, args):
    pred = p >= args.umbral_prob
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec = tp / float(tp + fp) if (tp + fp) else float("nan")
    rec = tp / float(tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec == prec and rec == rec and (prec + rec) > 0) else (0.0 if tp == 0 else float("nan"))
    # por episodio
    pred_f = np.zeros(len(pred), dtype=bool)
    for a, b in runs(pred):
        if b - a + 1 >= args.min_racha:
            pred_f[a:b + 1] = True
    ver = [(a, b) for a, b in runs(y == 1) if b - a + 1 >= args.min_racha]
    det = 0
    for a, b in ver:
        if pred_f[a:b + 1].mean() >= args.cobertura_ep:
            det += 1
    prs = runs(pred_f)
    fa = sum(1 for a, b in prs if not (y[a:b + 1] == 1).any())
    return {"prevalencia": float(y.mean()), "precision": prec, "recall": rec, "f1": f1, "ap": precision_media(y, p),
            "ep_verdad": len(ver), "ep_detectados": det,
            "ep_recall": (det / float(len(ver))) if ver else float("nan"),
            "ep_falsas_alarmas": fa, "ep_fa_por_dia": fa / dias if dias > 0 else float("nan")}


# ============================================================================
# VALIDACION POR BLOQUES
# ============================================================================
def cv_bloques(X, y, folds, purga, args):
    """Predicciones fuera de muestra con bloques contiguos; purga ciclos a cada lado del bloque de prueba."""
    n = len(y)
    p = np.zeros(n)
    lim = np.linspace(0, n, folds + 1).astype(int)
    for k in range(folds):
        a, b = lim[k], lim[k + 1]
        tr = np.ones(n, dtype=bool)
        tr[max(0, a - purga):min(n, b + purga)] = False
        if y[tr].min() == y[tr].max():
            p[a:b] = float(y[tr][0])
            continue
        p[a:b] = nuevo_modelo(args).fit(X[tr], y[tr]).proba(X[a:b])
    return p


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="TSTR minimo: utilidad del historial sintetico para aprendizaje supervisado")
    ap.add_argument("--sinteticos", nargs="+", required=True, help="verdad_ciclos_*.csv de cada serie sintetica (una por semilla)")
    ap.add_argument("--real-historial", required=True)
    ap.add_argument("--real-bloque-principal", required=True)
    ap.add_argument("--real-excluidos", default=None)
    ap.add_argument("--real-segmentos", required=True, help="segmentos_caracterizados_*.csv del HDP-HMM sobre la serie REAL")
    ap.add_argument("--ventanas-h", default="0.5,1,3,12", help="Ventanas causales en horas (default 0.5,1,3,12)")
    ap.add_argument("--ref-percentil", type=float, default=10.0, help="Percentil de referencia de cada serie (default 10)")
    ap.add_argument("--umbral-ms", type=float, default=2.0, help="Delta que define degradacion en la etiqueta sintetica (default 2)")
    ap.add_argument("--modelo", choices=["logreg", "gbm"], default="logreg")
    ap.add_argument("--umbral-prob", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5, help="Bloques de la validacion por bloques (TRTR y TSTS con una sola serie)")
    ap.add_argument("--purga-h", type=float, default=12.0, help="Horas descartadas a cada lado del bloque de prueba (default 12)")
    ap.add_argument("--min-racha", type=int, default=5, help="Ciclos minimos de un episodio (default 5 = min-dwell)")
    ap.add_argument("--cobertura-ep", type=float, default=0.5, help="Fraccion del episodio cubierta para contarlo detectado (default 0.5)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-dir", default="tstr")
    ap.add_argument("--etiqueta", default="")
    a = ap.parse_args()
    np.random.seed(a.seed)
    ventanas = [float(x) for x in a.ventanas_h.split(",") if x.strip()]

    # ---- datos
    t_r, y_r, lab_r, interval, n_seg, n_deg = cargar_real(a.real_historial, a.real_bloque_principal, a.real_excluidos, a.real_segmentos)
    if len(y_r) < 500:
        sys.exit("[ERROR] la serie real tiene solo {} ciclos.".format(len(y_r)))
    X_r, nombres, ref_r = caracteristicas(y_r, interval, ventanas, a.ref_percentil, a.umbral_ms)
    dias_r = len(y_r) * interval / 86400.0
    purga = int(round(a.purga_h * 3600.0 / interval))
    sint = []
    for pth in a.sinteticos:
        t_s, y_s, lab_s = cargar_sintetica(pth, a.umbral_ms)
        Xs, _, ref_s = caracteristicas(y_s, interval, ventanas, a.ref_percentil, a.umbral_ms)
        sint.append({"path": pth, "X": Xs, "y": lab_s, "dias": len(y_s) * interval / 86400.0, "ref": ref_s})

    filas = []   # (escenario, detalle, metricas)

    def agregar(esc, det, y, p, dias):
        m = metricas(y, p, dias, a)
        filas.append((esc, det, m))

    # ---- 1. TSTR
    Xtr = np.vstack([s["X"] for s in sint])
    ytr = np.concatenate([s["y"] for s in sint])
    mod = nuevo_modelo(a).fit(Xtr, ytr)
    agregar("TSTR", "entrena: {} serie(s) sintetica(s)".format(len(sint)), lab_r, mod.proba(X_r), dias_r)
    if len(sint) > 1:
        for i, s in enumerate(sint):
            m_i = nuevo_modelo(a).fit(s["X"], s["y"])
            agregar("TSTR (una semilla)", os.path.basename(os.path.dirname(os.path.abspath(s["path"]))) or "serie{}".format(i + 1),
                    lab_r, m_i.proba(X_r), dias_r)
    # ---- 2. TRTR
    if lab_r.min() == lab_r.max():
        print("[AVISO] la serie real no tiene ciclos de ambas clases segun el HDP-HMM: TRTR no se puede calcular.")
    else:
        agregar("TRTR", "validacion por bloques ({} bloques, purga {:g} h)".format(a.folds, a.purga_h),
                lab_r, cv_bloques(X_r, lab_r, a.folds, purga, a), dias_r)
    # ---- 3. TSTS
    if len(sint) > 1:
        for i, s in enumerate(sint):
            otros = [q for j, q in enumerate(sint) if j != i]
            m_i = nuevo_modelo(a).fit(np.vstack([q["X"] for q in otros]), np.concatenate([q["y"] for q in otros]))
            agregar("TSTS", "prueba en {}".format(os.path.basename(os.path.dirname(os.path.abspath(s["path"]))) or "serie{}".format(i + 1)),
                    s["y"], m_i.proba(s["X"]), s["dias"])
    else:
        s = sint[0]
        agregar("TSTS", "validacion por bloques en la unica serie sintetica", s["y"],
                cv_bloques(s["X"], s["y"], a.folds, purga, a), s["dias"])
    # ---- 4. Regla simple
    j = int(np.argmin([abs(h - 1.0) for h in ventanas]))
    k_med = nombres.index("med_{:g}h".format(ventanas[j]))
    regla = (X_r[:, k_med] >= a.umbral_ms).astype(float)
    agregar("Regla simple", "mediana {:g} h sobre la referencia >= {:g} ms".format(ventanas[j], a.umbral_ms), lab_r, regla, dias_r)

    # ---- informe
    def f(x, nd=3):
        return "-" if (x is None or x != x) else ("{:." + str(nd) + "f}").format(x)

    L = []
    P = L.append
    P("=" * 78)
    P("UTILIDAD PARA APRENDIZAJE SUPERVISADO (TSTR)" + ((" [" + a.etiqueta + "]") if a.etiqueta else ""))
    P("=" * 78)
    P("Serie real: {} ciclos ({:.1f} dias) | referencia (P{:g}) {:.2f} ms | segmentos HDP-HMM {} ({} de degradacion sostenida)".format(
        len(y_r), dias_r, a.ref_percentil, ref_r, n_seg, n_deg))
    P("Prevalencia de degradacion: real (plateada) {:.1%} | sinteticas {}".format(
        lab_r.mean(), ", ".join("{:.1%}".format(s["y"].mean()) for s in sint)))
    P("Series sinteticas: {} | modelo: {} | ventanas causales (h): {} | {} caracteristicas".format(
        len(sint), a.modelo, a.ventanas_h, len(nombres)))
    P("Umbral de probabilidad {:g} | episodio = racha >= {} ciclos, detectado si se cubre >= {:.0%}".format(
        a.umbral_prob, a.min_racha, a.cobertura_ep))
    P("")
    P("{:<19s}{:<46s}{:>7s}{:>7s}{:>7s}{:>7s}{:>13s}{:>8s}".format(
        "escenario", "detalle", "prec", "recall", "F1", "AP", "episodios", "FA/dia"))
    for esc, det, m in filas:
        P("{:<19s}{:<46s}{:>7s}{:>7s}{:>7s}{:>7s}{:>13s}{:>8s}".format(
            esc, det[:45], f(m["precision"], 2), f(m["recall"], 2), f(m["f1"], 2), f(m["ap"], 2),
            "{}/{}".format(m["ep_detectados"], m["ep_verdad"]), f(m["ep_fa_por_dia"], 2)))
    P("")
    d_tstr = [m for e, _, m in filas if e == "TSTR"][0]
    d_trtr = [m for e, _, m in filas if e == "TRTR"]
    if d_trtr:
        P("Brecha de utilidad (TRTR - TSTR): F1 {} | AP {} | recall de episodios {} | FA/dia {}".format(
            f(d_trtr[0]["f1"] - d_tstr["f1"], 3), f(d_trtr[0]["ap"] - d_tstr["ap"], 3),
            f(d_trtr[0]["ep_recall"] - d_tstr["ep_recall"], 3), f(d_trtr[0]["ep_fa_por_dia"] - d_tstr["ep_fa_por_dia"], 2)))
        P("  (positivo = entrenar con real rinde mas; cercano a 0 = el sintetico es util; negativo = el sintetico rinde mas)")
    reg = [m for e, _, m in filas if e == "Regla simple"][0]
    P("Aporte del aprendizaje sobre la regla simple (TSTR - regla): F1 {} | AP {}".format(
        f(d_tstr["f1"] - reg["f1"], 3), f(d_tstr["ap"] - reg["ap"], 3)))
    P("")
    P("--- Limitaciones ---")
    P("* La etiqueta real es PLATEADA (HDP-HMM), no verdad de campo: TSTR mide el acuerdo con ese detector, no la exactitud absoluta.")
    P("* Una sola sonda: el sintetico procede de la misma serie que se usa como prueba (misma distribucion de ruido); "
      "no demuestra generalizacion a otras sondas.")
    P("* La prevalencia difiere entre sintetico y real: se informa AP ademas de F1; el umbral de probabilidad no se recalibra.")
    P("* Los episodios al unir bloques de validacion se cortan en el limite del bloque (efecto menor).")
    P("=" * 78)
    texto = "\n".join(L)
    print(texto)

    os.makedirs(a.output_dir, exist_ok=True)
    suf = ("_" + a.etiqueta) if a.etiqueta else ""
    with open(os.path.join(a.output_dir, "tstr_resumen" + suf + ".txt"), "w") as fo:
        fo.write(texto + "\n")
    with open(os.path.join(a.output_dir, "tstr_resumen" + suf + ".csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["escenario", "detalle", "prevalencia", "precision", "recall", "f1", "ap", "ep_verdad", "ep_detectados",
                    "ep_recall", "ep_falsas_alarmas", "ep_fa_por_dia"])
        for esc, det, m in filas:
            w.writerow([esc, det, f(m["prevalencia"], 4), f(m["precision"], 4), f(m["recall"], 4), f(m["f1"], 4), f(m["ap"], 4),
                        m["ep_verdad"], m["ep_detectados"], f(m["ep_recall"], 4), m["ep_falsas_alarmas"], f(m["ep_fa_por_dia"], 4)])
    print("Archivos escritos en: {}".format(a.output_dir))


if __name__ == "__main__":
    main()
