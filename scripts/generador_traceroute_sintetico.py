#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generador_traceroute_sintetico.py - Generador de historiales de traceroute sinteticos con el formato de RIPE Atlas.

Produce el mismo CSV que consumen verificar_caminos.py, hdphmm_ripe_atlas_bnpy.py y el visor:
    timestamp,hop,ip,asn,rtt_ms            (una fila por salto; saltos mudos = fila con ip, asn y rtt vacios)

ALCANCE (v1). Es independiente del motor de failover y de TimescaleDB: no usa peer, jitter ni DNS, ni horario pico,
ni ciclos de confirmacion. Solo modela la medicion de traceroute hacia UN destino unico (sin anycast).
Nota: los valores del perfil por defecto son iniciales y aproximados (sondas 6835 y 1005718); hay que calibrarlos.

ARQUITECTURA POR CAPAS
  1. Reloj ........ cadencia fija + jitter; huecos de medicion opcionales (procesos de Poisson).
  2. Rutas ........ proceso semi-Markov sobre rutas (permanencia lognormal); cada ruta tiene sus saltos, saltos mudos y
                    variantes de interfaz (IP/ASN alternante) con cadena de Markov de dos estados por salto.
  3. Regimen RTT .. del ULTIMO salto: nivel por tramos (semi-Markov, permanencia lognormal): base / degradacion
                    moderada / degradacion severa (escalon de delta_ms desde el salto de localizacion hacia el destino).
                    + fondo AR(1) exacto (phi = exp(-dt/tau)) de cola positiva con piso
                    + spikes (Bernoulli por ciclo, magnitud lognormal, corridas cortas).
  4. Observacion .. RTT por salto (acumulado + ruido), picos propios en saltos intermedios, saltos mudos, y CICLOS SIN
                    DESTINO en tres modos (A: ultimo salto reemplazado con RTT de segundos; B: desvio y corte;
                    C: corte en el salto 3-4 con la IP del salto 2), con tasa de Poisson modulada por Markov
                    (calma / inestable) y racimos intermitentes.
  5. Verdad ...... verdad_ciclos_*.csv y verdad_episodios_*.csv.
  6. Perfil ....... JSON (se fusiona sobre el perfil por defecto; basta con indicar lo que cambia).

DIRECCIONES: IPv6 de documentacion (2001:db8::/32 y 3fff::/20) y ASN de documentacion 65536-65551 (RFC 5398).
verificar_caminos.py trata esos prefijos como IP publicas (es_privada).

SALIDAS en --output-dir (default sintetico_m<M>_p<P>):
  historial_traceroute_measurement<M>_probe<P>.csv     <- entrada de verificar_caminos / hdp-hmm
  verdad_ciclos_measurement<M>_probe<P>.csv            <- una fila por ciclo con el estado latente
  verdad_episodios_measurement<M>_probe<P>.csv         <- degradaciones, spikes, rutas alternativas, racimos, etc.
  perfil_usado_measurement<M>_probe<P>.json            <- perfil efectivo (reproducibilidad)

USO
  python3 generador_traceroute_sintetico.py --start "2026-07-01 12:00:00" --stop "2026-10-01 12:00:00" \\
        --interval 900 --measurement-id 900000001 --probe-id 1 --seed 42 [--perfil mi_perfil.json]
  python3 generador_traceroute_sintetico.py --escribir-perfil-ejemplo perfil_ejemplo.json
"""
from __future__ import print_function, division
import argparse
import copy
import csv
import ipaddress
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np

FMT_TS = "%Y-%m-%d %H:%M:%S"
ASN_DOC_MIN, ASN_DOC_MAX = 65536, 65551
REDES_DOC = [ipaddress.ip_network("2001:db8::/32"), ipaddress.ip_network("3fff::/20")]


# ============================================================================
# PERFIL POR DEFECTO (valores iniciales; calibrar con sondas reales)
# ============================================================================
def _saltos_comunes():
    """Saltos 1-5 compartidos por las rutas (red del proveedor local)."""
    return [
        {"asn": 65536, "ip": "2001:db8:100:4000::1", "rtt_ms": 0.4, "cv": 0.25},
        {"asn": 65536, "ip": "2001:db8:100:34::1", "rtt_ms": 1.6, "cv": 0.2},
        {"asn": 65536, "mudo": True},
        {"asn": 65536, "mudo": True},
        {"asn": 65536, "ip": "2001:db8:0:18::", "rtt_ms": 1.2, "cv": 0.15},
    ]


def perfil_defecto():
    r0 = {"id": "R0", "rtt_ultimo_ms": 29.5, "saltos": _saltos_comunes() + [
        {"asn": 65537, "rtt_ms": 1.4, "cv": 0.15,
         "alt": {"asn": 65536, "ip": "2001:db8:0:1005::", "p_entrada": 0.0007, "p_salida": 0.6}},
        {"asn": 65537, "rtt_ms": 29.2, "cv": 0.01},
        {"asn": 65537},
    ]}
    r1 = {"id": "R1", "rtt_ultimo_ms": 42.0, "saltos": _saltos_comunes() + [
        {"asn": 65538, "rtt_ms": 3.0, "cv": 0.1},
        {"asn": 65538, "rtt_ms": 20.0, "cv": 0.05},
        {"asn": 65538, "rtt_ms": 35.0, "cv": 0.05},
        {"asn": 65538, "rtt_ms": 36.0, "cv": 0.05},
        {"asn": 65536, "rtt_ms": 38.0, "cv": 0.05},
        {"asn": 65537, "rtt_ms": 40.0, "cv": 0.03},
        {"asn": 65537, "rtt_ms": 41.0, "cv": 0.03},
        {"asn": 65537},
    ]}
    return {
        "nombre": "defecto (rutas y fallos ~ sonda 6835; deltas de RTT ~ sonda 1005718) - valores iniciales",
        "reloj": {
            "jitter_s": 5.0,
            "huecos": {"activo": False, "por_dia": 0.0, "duracion": {"mediana_min": 60.0, "sigma_ln": 0.8}},
        },
        "ruta": {
            "destino": {"ip": "3fff:0:0:100::80", "asn": 65537},
            "asn_local": 65536,
            "rutas": [r0, r1],
            "proceso": {
                "inicial": "R0",
                "permanencia": {"R0": {"mediana_min": 57600.0, "sigma_ln": 0.8},
                                "R1": {"mediana_min": 230.0, "sigma_ln": 0.3}},
                "transiciones": {"R0": {"R1": 1.0}, "R1": {"R0": 1.0}},
                "p_acople_rtt": 1.0,
            },
            "p_pico_intermedio": 0.002,
            "pico_intermedio_ms": {"mediana": 30.0, "sigma_ln": 0.8},
        },
        "regimen_rtt": {
            "base": {"permanencia": {"mediana_min": 7200.0, "sigma_ln": 1.0}},
            "degradacion": {
                "permanencia": {"mediana_min": 5760.0, "sigma_ln": 0.9},
                "p_retorno_base": 0.5,
                "clases": {
                    "moderada": {"peso": 0.6, "delta_ms": {"mediana": 5.0, "sigma_ln": 0.3}},
                    "severa": {"peso": 0.4, "delta_ms": {"mediana": 17.0, "sigma_ln": 0.25}},
                },
            },
            "salto_min_localizacion": 6,
            "salto_max_localizacion": 8,
            "fondo": {"tau_s": 600.0, "escala_ms": 0.15, "sigma_log": 0.8, "piso_margen_ms": 1.0},
            "spike": {"p_por_ciclo": 0.005, "p_continuar": 0.15, "magnitud_ms": {"mediana": 10.0, "sigma_ln": 0.9}},
        },
        "fallos": {
            "estados": {
                "calma": {"permanencia": {"mediana_min": 72000.0, "sigma_ln": 0.6},
                          "aislados_por_dia": 0.16, "mezcla": {"A": 0.92, "B": 0.08},
                          "racimos_por_dia": 0.0},
                "inestable": {"permanencia": {"mediana_min": 20000.0, "sigma_ln": 0.4},
                              "aislados_por_dia": 0.37, "mezcla": {"A": 0.71, "C": 0.29},
                              "racimos_por_dia": 0.1},
            },
            "estado_inicial": "calma",
            "racimo": {"duracion": {"mediana_min": 120.0, "sigma_ln": 0.8}, "p_fallo_en_racimo": 0.6},
            "modos": {
                "A": {"ip": "2001:db8:0:1005::", "asn": 65536,
                      "rtt_s": {"mu_ln": 3.06, "sigma_ln": 0.148}},
                "B": {"salto_fin": 5,
                      "desvio": [{"salto": 3, "ip": "2001:db8:0:1005::", "asn": 65536, "rtt_ms": 11.2},
                                 {"salto": 4, "ip": "2001:db8:0:1200::1", "asn": 65536, "rtt_ms": 28.5}]},
                "C": {"salto_fin": {"3": 0.8125, "4": 0.1875}, "copia_ip_del_salto": 2},
            },
        },
    }


def fusionar(base, extra):
    """Fusiona recursivamente 'extra' sobre 'base' (los diccionarios se mezclan; listas y escalares se reemplazan)."""
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            fusionar(base[k], v)
        else:
            base[k] = v
    return base


# ============================================================================
# UTILIDADES
# ============================================================================
def dur_ciclos(minutos, intervalo_s):
    return max(1, int(round(minutos * 60.0 / intervalo_s)))


def lognormal_mediana(rng, par, clave_med="mediana_min"):
    return float(rng.lognormal(math.log(par[clave_med]), par["sigma_ln"]))


def eleccion(rng, pesos):
    claves = list(pesos.keys())
    p = np.array([pesos[k] for k in claves], dtype=float)
    return claves[int(rng.choice(len(claves), p=p / p.sum()))]


def fmt_rtt(x):
    return repr(round(float(x), 3))


def ip_auto(asn, asn_local, idx_ruta, salto, variante):
    """IP IPv6 de documentacion determinista: 2001:db8 para el ASN local, 3fff::/20 para el resto."""
    g = asn - ASN_DOC_MIN + 1
    if asn == asn_local:
        return "2001:db8:%x:%x:%x::%x" % (g, idx_ruta, salto, variante + 1)
    return "3fff:%x:%x:%x::%x" % (g, idx_ruta, salto, variante + 1)


# ============================================================================
# 2. RUTAS
# ============================================================================
class Ruta(object):
    def __init__(self, spec, idx, destino, asn_local):
        self.id = spec["id"]
        self.idx = idx
        self.rtt_ultimo_ms = float(spec["rtt_ultimo_ms"])
        self.saltos = []
        n = len(spec["saltos"])
        self.n = n
        for h, sp in enumerate(spec["saltos"], 1):
            s = dict(sp)
            if h == n:
                s["asn"], s["ip"] = destino["asn"], destino["ip"]
            else:
                s.setdefault("ip", None if s.get("mudo") else ip_auto(s["asn"], asn_local, idx, h, 0))
                if s.get("alt"):
                    a = dict(s["alt"])
                    a.setdefault("ip", ip_auto(a["asn"], asn_local, idx, h, 1))
                    s["alt"] = a
            s.setdefault("rtt_ms", 1.0)
            s.setdefault("cv", 0.15)
            s.setdefault("p_mudo", 0.0)
            self.saltos.append(s)


class ProcesoRuta(object):
    """Semi-Markov sobre rutas (permanencia lognormal) + variantes de interfaz por salto."""

    def __init__(self, cfg, intervalo_s, rng, perfil_ruta):
        self.rng = rng
        self.intervalo_s = intervalo_s
        dest, asn_local = perfil_ruta["destino"], perfil_ruta["asn_local"]
        self.rutas = {}
        for i, sp in enumerate(perfil_ruta["rutas"]):
            self.rutas[sp["id"]] = Ruta(sp, i, dest, asn_local)
        self.cfg = cfg
        self.actual = self.rutas[cfg["inicial"]]
        self.primera = self.actual.id
        self.restan = self._dur(self.actual.id)
        self.base_efectiva = self.actual.rtt_ultimo_ms
        self.alt_activa = {}  # (ruta_id, salto) -> bool

    def _dur(self, rid):
        return dur_ciclos(lognormal_mediana(self.rng, self.cfg["permanencia"][rid]), self.intervalo_s)

    def avanzar(self):
        cambio = False
        if self.restan <= 0:
            destinos = self.cfg["transiciones"].get(self.actual.id)
            if destinos:
                nuevo = self.rutas[eleccion(self.rng, destinos)]
                self.actual = nuevo
                cambio = True
                if self.rng.random() < self.cfg.get("p_acople_rtt", 1.0):
                    self.base_efectiva = nuevo.rtt_ultimo_ms
            self.restan = self._dur(self.actual.id)
        self.restan -= 1
        # variantes de interfaz
        activos = []
        for h, s in enumerate(self.actual.saltos, 1):
            a = s.get("alt")
            if not a:
                continue
            k = (self.actual.id, h)
            if self.alt_activa.get(k, False):
                if self.rng.random() < a["p_salida"]:
                    self.alt_activa[k] = False
            else:
                if self.rng.random() < a["p_entrada"]:
                    self.alt_activa[k] = True
            if self.alt_activa.get(k, False):
                activos.append(h)
        return cambio, activos


# ============================================================================
# 3. REGIMEN DE RTT DEL ULTIMO SALTO
# ============================================================================
class RegimenRTT(object):
    def __init__(self, cfg, intervalo_s, rng, n_ref):
        self.cfg, self.intervalo_s, self.rng = cfg, intervalo_s, rng
        self.estado = "base"
        self.clase, self.delta, self.k_loc = None, 0.0, None
        self.epi_id = 0
        self.restan = dur_ciclos(lognormal_mediana(rng, cfg["base"]["permanencia"]), intervalo_s)
        self.kmin = int(cfg.get("salto_min_localizacion", 6))
        self.kmax = int(cfg.get("salto_max_localizacion") or n_ref)
        if self.kmax < self.kmin:
            self.kmax = self.kmin

    def _nueva_degradacion(self):
        dg = self.cfg["degradacion"]
        pesos = dict((k, v["peso"]) for k, v in dg["clases"].items())
        self.clase = eleccion(self.rng, pesos)
        self.delta = lognormal_mediana(self.rng, dg["clases"][self.clase]["delta_ms"], "mediana")
        self.k_loc = int(self.rng.integers(self.kmin, self.kmax + 1))
        self.epi_id += 1
        self.estado = "degradacion"
        self.restan = dur_ciclos(lognormal_mediana(self.rng, dg["permanencia"]), self.intervalo_s)

    def avanzar(self):
        if self.restan <= 0:
            if self.estado == "base":
                self._nueva_degradacion()
            elif self.rng.random() < self.cfg["degradacion"]["p_retorno_base"]:
                self.estado, self.clase, self.delta, self.k_loc = "base", None, 0.0, None
                self.restan = dur_ciclos(lognormal_mediana(self.rng, self.cfg["base"]["permanencia"]),
                                         self.intervalo_s)
            else:
                self._nueva_degradacion()
        self.restan -= 1
        return self.estado, self.clase, self.delta, self.k_loc, self.epi_id


class Fondo(object):
    """AR(1) exacto con marginal lognormal centrada (cola positiva): fondo = escala*(exp(s*z) - exp(s^2/2))."""

    def __init__(self, cfg, intervalo_s, rng):
        self.rng = rng
        self.phi = math.exp(-intervalo_s / float(cfg["tau_s"]))
        self.escala, self.s = float(cfg["escala_ms"]), float(cfg["sigma_log"])
        self.margen = float(cfg["piso_margen_ms"])
        self.z = float(rng.normal())

    def muestra(self):
        self.z = self.phi * self.z + math.sqrt(1.0 - self.phi ** 2) * float(self.rng.normal())
        return self.escala * (math.exp(self.s * self.z) - math.exp(self.s ** 2 / 2.0))


class Spikes(object):
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.restan = 0

    def avanzar(self):
        """Devuelve la magnitud (ms) a sumar al ultimo salto en este ciclo (0 si no hay spike)."""
        if self.restan <= 0 and self.rng.random() < self.cfg["p_por_ciclo"]:
            extra = 0
            pc = self.cfg.get("p_continuar", 0.0)
            if pc > 0:
                extra = int(self.rng.geometric(1.0 - pc)) - 1
            self.restan = 1 + extra
        if self.restan > 0:
            self.restan -= 1
            return lognormal_mediana(self.rng, self.cfg["magnitud_ms"], "mediana")
        return 0.0


# ============================================================================
# 4. CICLOS SIN DESTINO: Poisson modulado por Markov + racimos
# ============================================================================
class ProcesoFallos(object):
    def __init__(self, cfg, intervalo_s, rng):
        self.cfg, self.intervalo_s, self.rng = cfg, intervalo_s, rng
        self.estado = cfg["estado_inicial"]
        self.restan = self._dur_estado()
        self.restan_racimo = 0

    def _dur_estado(self):
        return dur_ciclos(lognormal_mediana(self.rng, self.cfg["estados"][self.estado]["permanencia"]),
                          self.intervalo_s)

    def _p(self, por_dia):
        return 1.0 - math.exp(-por_dia * self.intervalo_s / 86400.0)

    def avanzar(self):
        """Devuelve (estado, fallo) con fallo = None o {'modo': 'A'|'B'|'C', 'racimo': bool}."""
        if self.restan <= 0:
            otros = [k for k in self.cfg["estados"] if k != self.estado]
            if otros:
                self.estado = otros[0]
            self.restan = self._dur_estado()
        self.restan -= 1
        est = self.cfg["estados"][self.estado]
        fallo = None
        if self.restan_racimo <= 0 and self.rng.random() < self._p(est["racimos_por_dia"]):
            self.restan_racimo = dur_ciclos(lognormal_mediana(self.rng, self.cfg["racimo"]["duracion"]),
                                            self.intervalo_s)
        if self.restan_racimo > 0:
            self.restan_racimo -= 1
            if self.rng.random() < self.cfg["racimo"]["p_fallo_en_racimo"]:
                fallo = {"modo": "C", "racimo": True}
        elif self.rng.random() < self._p(est["aislados_por_dia"]):
            fallo = {"modo": eleccion(self.rng, est["mezcla"]), "racimo": False}
        return self.estado, fallo, self.restan_racimo > 0


class Huecos(object):
    def __init__(self, cfg, intervalo_s, rng):
        self.cfg, self.intervalo_s, self.rng = cfg, intervalo_s, rng
        self.restan = 0

    def avanzar(self):
        if not self.cfg.get("activo"):
            return False
        if self.restan <= 0:
            p = 1.0 - math.exp(-self.cfg["por_dia"] * self.intervalo_s / 86400.0)
            if self.rng.random() < p:
                self.restan = dur_ciclos(lognormal_mediana(self.rng, self.cfg["duracion"]), self.intervalo_s)
        if self.restan > 0:
            self.restan -= 1
            return True
        return False


# ============================================================================
# 5. EPISODIOS (verdad de terreno)
# ============================================================================
class Episodios(object):
    def __init__(self):
        self.abiertos = {}
        self.cerrados = []

    def actualizar(self, cat, clave, idx, ts, magnitud=None, detalle=""):
        ab = self.abiertos.get(cat)
        if ab is not None and ab["clave"] != clave:
            self._cerrar(cat)
            ab = None
        if clave is None:
            return
        if ab is None:
            self.abiertos[cat] = {"clave": clave, "i0": idx, "t0": ts, "i1": idx, "t1": ts,
                                  "mag": magnitud, "det": detalle}
        else:
            ab["i1"], ab["t1"] = idx, ts
            if magnitud is not None and (ab["mag"] is None or magnitud > ab["mag"]):
                ab["mag"] = magnitud

    def _cerrar(self, cat):
        ab = self.abiertos.pop(cat)
        self.cerrados.append((ab["i0"], ab["clave"][0], ab["t0"], ab["t1"], ab["i1"] - ab["i0"] + 1,
                              ab["mag"], ab["det"]))

    def finalizar(self):
        for cat in list(self.abiertos.keys()):
            self._cerrar(cat)
        return sorted(self.cerrados, key=lambda e: (e[0], e[1]))


# ============================================================================
# 4b. EMISION DE FILAS DE UN CICLO
# ============================================================================
def emitir_ciclo(ts, ruta, alt_saltos, k_loc, delta, base_ult, fondo, spike, fallo, perfil, rng):
    """Devuelve (filas, rtt_ultimo_emitido_o_None, hop_final)."""
    saltos = ruta.saltos
    n = ruta.n
    modo = fallo["modo"] if fallo else None
    fcfg = perfil["fallos"]["modos"]
    rcfg = perfil["ruta"]
    limite = n
    h_c = None
    if modo == "B" and n < fcfg["B"]["salto_fin"]:
        modo = "A"  # ruta demasiado corta para el desvio: se degrada a modo A
    if modo == "C":
        h_c = int(eleccion(rng, fcfg["C"]["salto_fin"]))
        limite = min(h_c, n)
    elif modo == "B":
        limite = fcfg["B"]["salto_fin"]
    desv = dict((d["salto"], d) for d in fcfg["B"]["desvio"]) if modo == "B" else {}
    k_eff = min(k_loc, n) if k_loc else None
    filas = []
    rtt_ult = None
    for h in range(1, limite + 1):
        sp = saltos[h - 1]
        es_ult = (h == n)
        if modo == "C" and h == limite:
            ref = saltos[fcfg["C"]["copia_ip_del_salto"] - 1]
            rtt = max(0.05, ref["rtt_ms"] * (1.0 + ref["cv"] * float(rng.normal())))
            filas.append((ts, h, ref["ip"], ref["asn"], rtt))
            continue
        if modo == "B" and h in desv:
            d = desv[h]
            filas.append((ts, h, d["ip"], d["asn"], d["rtt_ms"] * (1.0 + 0.02 * float(rng.normal()))))
            continue
        if modo == "A" and es_ult:
            a = fcfg["A"]
            rtt = 1000.0 * float(rng.lognormal(a["rtt_s"]["mu_ln"], a["rtt_s"]["sigma_ln"]))
            filas.append((ts, h, a["ip"], a["asn"], rtt))
            continue
        if es_ult:
            piso = max(0.1, base_ult - perfil["regimen_rtt"]["fondo"]["piso_margen_ms"])
            rtt_ult = max(piso, base_ult + delta + fondo + spike)
            filas.append((ts, h, sp["ip"], sp["asn"], rtt_ult))
            continue
        if sp.get("mudo") or (sp["p_mudo"] > 0 and rng.random() < sp["p_mudo"]):
            filas.append((ts, h, None, None, None))
            continue
        ip, asn = sp["ip"], sp["asn"]
        if h in alt_saltos:
            ip, asn = sp["alt"]["ip"], sp["alt"]["asn"]
        rtt = max(0.05, sp["rtt_ms"] * (1.0 + sp["cv"] * float(rng.normal())))
        if k_eff and h >= k_eff:
            rtt += delta
        if rng.random() < rcfg["p_pico_intermedio"]:
            rtt += float(rng.lognormal(math.log(rcfg["pico_intermedio_ms"]["mediana"]),
                                       rcfg["pico_intermedio_ms"]["sigma_ln"]))
        filas.append((ts, h, ip, asn, rtt))
    return filas, rtt_ult, limite


# ============================================================================
# GENERACION
# ============================================================================
def validar_perfil(perfil, max_hops):
    for r in perfil["ruta"]["rutas"]:
        if len(r["saltos"]) > max_hops:
            raise ValueError("La ruta %s tiene %d saltos y --max-hops es %d" % (r["id"], len(r["saltos"]), max_hops))
        for s in r["saltos"]:
            a = s.get("asn")
            if a is not None and not (ASN_DOC_MIN <= int(a) <= ASN_DOC_MAX):
                raise ValueError("ASN %s fuera del rango de documentacion %d-%d" % (a, ASN_DOC_MIN, ASN_DOC_MAX))
            ip = s.get("ip")
            if ip and not any(ipaddress.ip_address(ip) in n for n in REDES_DOC):
                raise ValueError("IP %s fuera de 2001:db8::/32 y 3fff::/20" % ip)


def generar(perfil, inicio, fin, intervalo_s, max_hops, semilla, mid, pid, out_dir):
    validar_perfil(perfil, max_hops)
    ss = np.random.SeedSequence(semilla)
    nombres = ["reloj", "huecos", "ruta", "regimen", "fondo", "spike", "fallos", "obs"]
    rngs = dict(zip(nombres, [np.random.default_rng(s) for s in ss.spawn(len(nombres))]))

    pr = ProcesoRuta(perfil["ruta"]["proceso"], intervalo_s, rngs["ruta"], perfil["ruta"])
    n_ref = max(r.n for r in pr.rutas.values())
    reg = RegimenRTT(perfil["regimen_rtt"], intervalo_s, rngs["regimen"], n_ref)
    fon = Fondo(perfil["regimen_rtt"]["fondo"], intervalo_s, rngs["fondo"])
    spk = Spikes(perfil["regimen_rtt"]["spike"], rngs["spike"])
    fal = ProcesoFallos(perfil["fallos"], intervalo_s, rngs["fallos"])
    hue = Huecos(perfil["reloj"]["huecos"], intervalo_s, rngs["huecos"])
    epi = Episodios()

    os.makedirs(out_dir, exist_ok=True)
    sfx = "measurement%s_probe%s" % (mid, pid)
    f_hist = open(os.path.join(out_dir, "historial_traceroute_%s.csv" % sfx), "w", newline="")
    f_ciclos = open(os.path.join(out_dir, "verdad_ciclos_%s.csv" % sfx), "w", newline="")
    w_h = csv.writer(f_hist, lineterminator="\n")
    w_c = csv.writer(f_ciclos, lineterminator="\n")
    w_h.writerow(["timestamp", "hop", "ip", "asn", "rtt_ms"])
    w_c.writerow(["idx", "timestamp", "hueco", "ruta_id", "regimen", "delta_ms", "salto_localizacion", "nivel_ultimo_ms",
                  "spike_ms", "sin_destino", "modo_fallo", "en_racimo", "estado_fallos", "saltos_variante_alt",
                  "n_saltos_emitidos", "rtt_ultimo_ms"])

    n_ciclos = int((fin - inicio).total_seconds() // intervalo_s)
    jit = float(perfil["reloj"]["jitter_s"])
    ts_prev = None
    cont = {"sin_destino": 0, "huecos": 0}
    for i in range(n_ciclos):
        nominal = inicio + timedelta(seconds=i * intervalo_s)
        j = float(np.clip(rngs["reloj"].normal(0.0, jit), -intervalo_s / 4.0, intervalo_s / 4.0)) if jit > 0 else 0.0
        ts_dt = nominal + timedelta(seconds=int(round(j)))
        if ts_prev is not None and ts_dt <= ts_prev:
            ts_dt = ts_prev + timedelta(seconds=1)
        en_hueco = hue.avanzar()
        cambio_ruta, alt_saltos = pr.avanzar()
        estado, clase, delta, k_loc, epi_id = reg.avanzar()
        fondo = fon.muestra()
        spike = spk.avanzar()
        est_f, fallo, en_racimo = fal.avanzar()
        ruta = pr.actual
        ts_txt = (nominal if en_hueco else ts_dt).strftime(FMT_TS)
        k_eff = min(k_loc, ruta.n) if (k_loc and delta > 0) else None

        filas, rtt_ult, h_fin = [], None, 0
        if not en_hueco:
            ts_prev = ts_dt
            filas, rtt_ult, h_fin = emitir_ciclo(ts_txt, ruta, alt_saltos, k_loc if delta > 0 else None, delta,
                                                 pr.base_efectiva, fondo, spike, fallo, perfil, rngs["obs"])
            for (t, h, ip, asn, rtt) in filas:
                if ip is None:
                    w_h.writerow([t, h, "", "", ""])
                else:
                    w_h.writerow([t, h, ip, "Unknown" if asn is None else asn, fmt_rtt(rtt)])
        sin_dest = bool(fallo) and not en_hueco
        modo = fallo["modo"] if fallo else ""
        if fallo and fallo["modo"] == "B" and ruta.n < perfil["fallos"]["modos"]["B"]["salto_fin"]:
            modo = "A"
        cont["sin_destino"] += int(sin_dest)
        cont["huecos"] += int(en_hueco)
        nivel = pr.base_efectiva + delta
        w_c.writerow([i, ts_txt, int(en_hueco), ruta.id, "base" if estado == "base" else clase,
                      fmt_rtt(delta), k_eff or "", fmt_rtt(nivel), fmt_rtt(spike) if spike > 0 else "0",
                      int(sin_dest), modo if sin_dest else "", int(en_racimo), est_f,
                      " ".join(str(h) for h in alt_saltos), len(filas) and h_fin or "",
                      "" if (rtt_ult is None or sin_dest) else fmt_rtt(rtt_ult)])

        # episodios
        epi.actualizar("regimen", ("degradacion_%s" % clase, epi_id) if delta > 0 else None, i, ts_txt, delta,
                       "salto_localizacion=%s" % k_eff if k_eff else "")
        epi.actualizar("spike", ("spike", 0) if spike > 0 else None, i, ts_txt, spike)
        epi.actualizar("ruta", ("ruta_alternativa", ruta.id) if ruta.id != pr.primera else None, i, ts_txt, None,
                       "ruta=%s" % ruta.id)
        epi.actualizar("racimo", ("racimo_fallos", 0) if en_racimo else None, i, ts_txt)
        epi.actualizar("inestable", ("periodo_inestable", 0) if est_f == "inestable" else None, i, ts_txt)
        epi.actualizar("sin_destino", ("sin_destino", modo) if sin_dest else None, i, ts_txt, None,
                       "modo=%s" % modo)
        epi.actualizar("hueco", ("hueco", 0) if en_hueco else None, i, ts_txt)

    f_hist.close()
    f_ciclos.close()
    with open(os.path.join(out_dir, "verdad_episodios_%s.csv" % sfx), "w", newline="") as fe:
        we = csv.writer(fe, lineterminator="\n")
        we.writerow(["tipo", "inicio", "fin", "n_ciclos", "magnitud_ms", "detalle"])
        for (_, tipo, t0, t1, n, mag, det) in epi.finalizar():
            we.writerow([tipo, t0, t1, n, "" if mag is None else fmt_rtt(mag), det])
    with open(os.path.join(out_dir, "perfil_usado_%s.json" % sfx), "w") as fp:
        json.dump(perfil, fp, indent=2, ensure_ascii=False)
    return n_ciclos, cont


def parse_ts(s):
    """Acepta 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DDTHH:MM:SS' o solo 'YYYY-MM-DD' (UTC)."""
    s = s.strip().replace("T", " ")[:19]
    for f in (FMT_TS, "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    raise ValueError("fecha invalida: %r (use 'YYYY-MM-DD HH:MM:SS')" % s)


def main():
    ap = argparse.ArgumentParser(description="Generador de historiales de traceroute sinteticos (formato RIPE Atlas).")
    ap.add_argument("--measurement-id", default="900000001", help="Etiqueta para nombrar los archivos (no consulta RIPE)")
    ap.add_argument("--probe-id", default="1", help="Etiqueta para nombrar los archivos (no consulta RIPE)")
    ap.add_argument("--start", help='Inicio UTC, "YYYY-MM-DD HH:MM:SS"')
    ap.add_argument("--stop", help="Fin UTC (exclusivo)")
    ap.add_argument("--interval", type=float, default=900.0, help="Cadencia en segundos (default 900)")
    ap.add_argument("--max-hops", type=int, default=32, help="Tope de saltos por ciclo (default 32)")
    ap.add_argument("--perfil", default=None, help="JSON que se fusiona sobre el perfil por defecto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--escribir-perfil-ejemplo", metavar="RUTA", default=None,
                    help="Escribe el perfil por defecto en RUTA y termina")
    a = ap.parse_args()

    if a.escribir_perfil_ejemplo:
        with open(a.escribir_perfil_ejemplo, "w") as f:
            json.dump(perfil_defecto(), f, indent=2, ensure_ascii=False)
        print("Perfil por defecto escrito en: %s" % a.escribir_perfil_ejemplo)
        return
    if not a.start or not a.stop:
        ap.error("--start y --stop son obligatorios")
    if a.interval <= 0:
        ap.error("--interval debe ser > 0")
    try:
        inicio, fin = parse_ts(a.start), parse_ts(a.stop)
    except ValueError as e:
        ap.error(str(e))
    if fin <= inicio:
        ap.error("--stop debe ser posterior a --start")
    perfil = perfil_defecto()
    if a.perfil:
        with open(a.perfil) as f:
            fusionar(perfil, json.load(f))
    out = a.output_dir or "sintetico_m%s_p%s" % (a.measurement_id, a.probe_id)
    try:
        n, cont = generar(perfil, inicio, fin, a.interval, a.max_hops, a.seed, a.measurement_id, a.probe_id, out)
    except ValueError as e:
        ap.error(str(e))
    print("SINTETICO - measurement %s | probe %s | semilla %d | perfil: %s" % (
        a.measurement_id, a.probe_id, a.seed, perfil.get("nombre", "")))
    print("Ciclos generados: %d (cadencia %g s) | sin destino: %d (%.2f%%) | huecos: %d" % (
        n, a.interval, cont["sin_destino"], 100.0 * cont["sin_destino"] / max(1, n), cont["huecos"]))
    print("Salidas en: %s" % out)


if __name__ == "__main__":
    main()
