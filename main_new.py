# ============================================================
# framework_new.py
# User-friendly JAX-compatible thermo-hydraulic framework
# ============================================================

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable
import jax
import jax.numpy as jnp


# ============================================================
# Core variable representation
# ============================================================

@dataclass(frozen=True)
class VarSlice:
    index: int


class PortRole(Enum):
    INLET = "inlet"
    OUTLET = "outlet"


@dataclass(frozen=True)
class Port:
    name: str
    role: PortRole
    p: VarSlice
    h: VarSlice
    m: VarSlice


# ============================================================
# Parameter specification
# ============================================================

@dataclass(frozen=True)
class ParameterSpec:
    init: Callable | float
    trainable: bool = True


# ============================================================
# Component base class
# ============================================================

class Component:
    port_specs = {}          # name -> PortRole
    parameter_specs = {}     # name -> ParameterSpec

    def __init__(self, name: str):
        self.name = name
        self._ports: Dict[str, Port] = {}

    def __getattr__(self, item):
        if item in self._ports:
            return self._ports[item]
        raise AttributeError(item)

    def residual(self, x, params):
        raise NotImplementedError


# ============================================================
# Components
# ============================================================

class Compressor(Component):
    port_specs = {
        "inlet": PortRole.INLET,
        "outlet": PortRole.OUTLET,
    }

    parameter_specs = {
        "eta_is": ParameterSpec(init=0.7, trainable=True)
    }

    def residual(self, x, p):
        i, o = self.inlet, self.outlet
        return jnp.array([
            x[o.p.index] - x[i.p.index],
            x[o.h.index] - x[i.h.index],
            x[o.m.index] - x[i.m.index],
        ])


class HeatExchanger(Component):
    port_specs = {
        "hot_in": PortRole.INLET,
        "hot_out": PortRole.OUTLET,
        "cold_in": PortRole.INLET,
        "cold_out": PortRole.OUTLET,
    }

    parameter_specs = {
        "nn_params": ParameterSpec(
            init=lambda rng: jax.random.normal(rng, (4,)),
            trainable=True
        )
    }

    def residual(self, x, p):
        Q = jnp.linalg.norm(p["nn_params"])

        hi, ho = self.hot_in, self.hot_out
        ci, co = self.cold_in, self.cold_out

        return jnp.array([
            x[ho.p.index] - x[hi.p.index],
            x[ho.h.index] - (x[hi.h.index] - Q / x[hi.m.index]),
            x[ho.m.index] - x[hi.m.index],

            x[co.p.index] - x[ci.p.index],
            x[co.h.index] - (x[ci.h.index] + Q / x[ci.m.index]),
            x[co.m.index] - x[ci.m.index],
        ])


# ============================================================
# Junctions
# ============================================================

class JunctionType(Enum):
    THROUGH = "through"
    MERGE = "merge"
    SPLIT = "split"


@dataclass(frozen=True)
class Junction:
    ports: tuple
    kind: JunctionType


def classify_junction(ports):
    ins = [p for p in ports if p.role == PortRole.INLET]
    outs = [p for p in ports if p.role == PortRole.OUTLET]

    if len(ins) == 1 and len(outs) == 1:
        return JunctionType.THROUGH
    if len(ins) > 1 and len(outs) == 1:
        return JunctionType.MERGE
    if len(ins) == 1 and len(outs) > 1:
        return JunctionType.SPLIT

    raise ValueError("Invalid junction")


def junction_residual(x, j):
    res = []

    res.append(sum(
        (+1 if p.role == PortRole.INLET else -1) * x[p.m.index]
        for p in j.ports
    ))

    if j.kind == JunctionType.THROUGH:
        i, o = j.ports
        res.append(x[o.h.index] - x[i.h.index])

    elif j.kind == JunctionType.MERGE:
        outs = [p for p in j.ports if p.role == PortRole.OUTLET][0]
        ins = [p for p in j.ports if p.role == PortRole.INLET]
        res.append(
            sum(x[p.m.index] * x[p.h.index] for p in ins)
            - x[outs.m.index] * x[outs.h.index]
        )

    elif j.kind == JunctionType.SPLIT:
        i = [p for p in j.ports if p.role == PortRole.INLET][0]
        for o in [p for p in j.ports if p.role == PortRole.OUTLET]:
            res.append(x[o.h.index] - x[i.h.index])

    return jnp.array(res)


# ============================================================
# Case (Boundary conditions)
# ============================================================

class Case:
    def __init__(self, builder):
        self.builder = builder
        self.bc: Dict[VarSlice, float] = {}

    def set_bc(self, var: VarSlice, value):
        self.bc[var] = value


# ============================================================
# NetworkBuilder
# ============================================================

class NetworkBuilder:
    def __init__(self):
        self._cursor = 0
        self.components = []
        self.junctions = []
        self._p_parent = {}
        self.parameter_registry = {}

    def _alloc(self):
        s = VarSlice(self._cursor)
        self._cursor += 1
        return s

    def add(self, comp: Component):
        self.components.append(comp)
        self.parameter_registry[comp.name] = comp.parameter_specs

        for pname, role in comp.port_specs.items():
            comp._ports[pname] = Port(
                name=f"{comp.name}.{pname}",
                role=role,
                p=self._alloc(),
                h=self._alloc(),
                m=self._alloc(),
            )

    def connect(self, *ports):
        p0 = ports[0].p
        for p in ports[1:]:
            self._p_parent[p.p] = p0
        self.junctions.append(
            Junction(ports, classify_junction(ports))
        )

    def finalize(self):
        def canon(v):
            while v in self._p_parent:
                v = self._p_parent[v]
            return v

        for c in self.components:
            for k, p in c._ports.items():
                c._ports[k] = Port(
                    p.name, p.role,
                    canon(p.p), p.h, p.m
                )

    def init_params(self, rng):
        params = {}
        for cname, specs in self.parameter_registry.items():
            params[cname] = {}
            for pname, spec in specs.items():
                params[cname][pname] = (
                    spec.init(rng) if callable(spec.init) else spec.init
                )
        return params

    def residual(self, x, params, case: Case):
        res = []

        for c in self.components:
            res.append(c.residual(x, params[c.name]))

        for j in self.junctions:
            res.append(junction_residual(x, j))

        for var, val in case.bc.items():
            res.append(x[var.index] - val)

        return jnp.concatenate(res)


builder = NetworkBuilder()

# Komponenten erzeugen (Ports entstehen automatisch!)
comp = Compressor("Comp")
hx   = HeatExchanger("HX")

# Komponenten registrieren
builder.add(comp)
builder.add(hx)

# Verbindungen (objektbasiert, keine Namen!)
builder.connect(comp.outlet, hx.cold_in)

# Struktur einfrieren
builder.finalize()


case = Case(builder)

# Physikalische Randbedingungen setzen
case.set_bc(comp.inlet.p, 5e5)   # Druck [Pa]
case.set_bc(comp.inlet.h, 3e5)   # Enthalpie
case.set_bc(comp.inlet.m, 0.1)   # Massenstrom

case.set_bc(hx.hot_in.p, 5e5)
case.set_bc(hx.hot_in.h, 3e5)
case.set_bc(hx.hot_in.m, 0.1)
