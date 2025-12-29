"""
Created on Thu Jan 19 07:14:57 2025

@author: Mirco Ganz

Accompany the PhD Thesis:

M. Ganz (2025) Methodology for Solving Direct and Inverse Steady-State Thermal-Hydraulic Network Problems
"""

from typing import Dict, List, Callable
import numpy as np
import scipy
from CoolProp.CoolProp import PropsSI
import multiprocessing as mp
from copy import deepcopy
from multiprocessing.dummy import freeze_support


class Variable:
    """
    Scalar variable associated with a port in the thermal-hydraulic network.

    A Variable represents a physical quantity (pressure, specific enthalpy,
    or mass flow rate) that may be either known or unknown. Unknown variables
    are resolved through tearing, equation solving, or numerical optimization.

    Parameters
    ----------
    name : str
        Human-readable variable name.
    var_type : str
        Physical variable type ('p', 'h', or 'm').
    port_type : str
        Direction of the port ('in' or 'out').
    port : Port
        Parent port to which this variable belongs.
    scale_factor : float, optional
        Scaling factor used during numerical optimization.
    value : float or None, optional
        Current value of the variable. If None, the variable is unknown.
    is_var : bool, optional
        Flag indicating whether this variable is part of the optimization vector.
    bounds : tuple of float, optional
        Lower and upper bounds for optimization.
    """

    __slots__ = (
        "name", "var_type", "port_type", "port",
        "scale_factor", "value", "initial_value", "is_var", "bounds"
    )

    def __init__(
        self,
        name: str,
        var_type: str,
        port_type: str,
        port,
        scale_factor: float = 1.0,
        value=None,
        is_var=False,
        bounds=(-np.inf, np.inf),
    ):
        self.name = name
        self.var_type = var_type
        self.port_type = port_type
        self.port = port
        self.scale_factor = scale_factor
        self.value = value
        self.initial_value = value
        self.is_var = is_var
        self.bounds = bounds

    @property
    def known(self) -> bool:
        """
        Return True if the variable has a known value.
        """
        return self.value is not None

    def set_value(self, v):
        """
        Assign a numerical value to the variable.

        Parameters
        ----------
        v : float
            Value to assign.
        """
        self.value = v

    def get_value(self):
        """
        Return the current numerical value of the variable.

        Returns
        -------
        float or None
            Current value of the variable. Returns ``None`` if the
            variable has not yet been assigned.
        """
        return self.value

    def reset(self):
        """
        Reset the variable to its initial value.
        """
        self.value = self.initial_value

    def __repr__(self):
        return f"{self.name}"


class Parameter:
    """
    Model parameter associated with a component.

    Parameters may be fixed or treated as optimization variables. Each
    parameter carries its own scaling and bounds for numerical robustness.

    Parameters
    ----------
    label : str
        Parameter name.
    value : float
        Current parameter value.
    scale_factor : float, optional
        Scaling factor used during optimization.
    initial_value : float, optional
        Initial value used for solver initialization.
    is_var : bool, optional
        Flag indicating whether the parameter is optimized.
    bounds : tuple of float, optional
        Lower and upper bounds for optimization.
    """

    __slots__ = ("label", "value", "initial_value", "scale_factor", "is_var", "bounds")

    def __init__(
        self,
        label,
        value,
        scale_factor=1.0,
        initial_value=None,
        is_var=False,
        bounds=(-np.inf, np.inf)
    ):
        self.label = label
        self.value = value
        self.initial_value = initial_value
        self.scale_factor = scale_factor
        self.is_var = is_var
        self.bounds = bounds

    def set_value(self, v):
        """
        Assign a new value to the parameter.
        """
        self.value = v

    def reset(self):
        """
        Reset parameter to its initial value.
        """
        self.value = self.initial_value


class Output:
    """
    Output quantity produced by a component model.

    Outputs are not part of the equation system but may be used for
    post-processing, monitoring, or objective functions.

    Parameters
    ----------
    label : str
        Output name.
    """

    def __init__(self, label: str):
        self.label = label
        self.value = None

    def set_value(self, value: float):
        """
        Assign a value to the output.

        Parameters
        ----------
        value : float
            Output value.
        """
        self.value = value


class PortSpec:
    """
    Specification of a component port.

    PortSpec defines the static topological properties of a port and is used
    to construct actual Port instances inside a component.

    Parameters
    ----------
    direction : str
        Port direction ('in' or 'out').
    """

    __slots__ = ("name", "direction")

    def __init__(self, direction: str):
        if direction not in ("in", "out"):
            raise ValueError("PortSpec.direction must be 'in' or 'out'")
        self.direction = direction


class Port:
    """
    Component port carrying thermo-hydraulic state variables.

    A Port is associated with a Component and may be connected to a Junction.
    Each port carries pressure, enthalpy, and mass flow variables.

    Parameters
    ----------
    spec : PortSpec
        Port specification.
    component : Component
        Parent component.
    """

    __slots__ = ("name", "port_type", "fluid", "component", "junction", "p", "h", "m")

    def __init__(self, name: str, spec: PortSpec, component):
        self.name = name
        self.port_type = spec.direction
        self.fluid = None
        self.component = component
        self.junction = None

        self.p = Variable(f"p({self.name})", "p", self.port_type, self, 1e-5)
        self.h = Variable(f"h({self.name})", "h", self.port_type, self, 1e-5)
        self.m = Variable(f"m({self.name})", "m", self.port_type, self, 1e1)

    def reset(self):
        """
        Reset all port variables to their initial values.
        """
        self.p.reset()
        self.h.reset()
        self.m.reset()


class Component:
    """
    Base class for all thermal-hydraulic components.

    A Component encapsulates a local physical model and a set of ports.
    Components are executed according to the tearing-derived execution order.

    Parameters
    ----------
    label : str
        Unique component identifier.
    port_specs : dict
        Dictionary mapping port names to PortSpec instances.
    model : callable
        Component model function with signature model(component).
    """

    modeling_type = "Generic"

    def __init__(self, label: str, port_specs: dict):
        self.label = label
        self.ports = {
            name: Port(name, spec, self) for name, spec in port_specs.items()
        }
        self.parameter = {}
        self.executed = False
        self.model = None
        self.outputs = {}

    def solve(self):
        """
        Execute the component model.
        """
        if self.model is None:
            raise RuntimeError(
                f"Component model of {self.label} is not yet assigned"
            )
        else:
            return self.model(self)

    def set_model(self, model: Callable):
        self.model = model

    def reset(self):
        """
        Reset component execution state and all port variables.
        """
        self.executed = False
        for p in self.ports.values():
            p.reset()

    def __repr__(self):
        return self.label


class PressureBasedComponent(Component):
    """
    Component whose causality is pressure-based.

    Typically used for compressors, pumps, or expansion devices.
    """

    modeling_type = "Pressure Based"


class MassFlowBasedComponent(Component):
    """
    Component whose causality is mass-flow-based.

    Typically used for heat exchangers and mixers.
    """

    modeling_type = "Mass Flow Based"


class BypassComponent(Component):
    """
    Bypass component with direct propagation of thermo-hydraulic state.
    """

    modeling_type = "Bypass"


class BalanceEquation:
    """
    Base class for all balance equations at a junction.

    A BalanceEquation represents a single algebraic conservation or equality
    relation between a set of variables belonging to the same fluid loop.
    Subclasses implement specific physical laws such as mass conservation,
    pressure equality, or enthalpy conservation.

    Parameters
    ----------
    variables : list
        List of Variable instances participating in the equation.
    fluid_loop : int
        Identifier of the fluid loop to which this equation belongs.
    """

    __slots__ = (
        "variables",
        "fluid_loop",
        "solved",
    )

    name = ""

    def __init__(self, variables, fluid_loop):

        if not isinstance(variables, (list, tuple)):
            raise TypeError(
                "BalanceEquation.variables must be a list or tuple of Variables."
            )

        self.variables = list(variables)
        self.fluid_loop = fluid_loop
        self.solved = False

    def is_solvable(self) -> bool:
        """
        Check whether the equation is solvable.

        An equation is solvable if exactly one participating variable
        is currently unknown.

        Returns
        -------
        bool
            True if the equation has exactly one unknown variable.
        """

        unknown_count = 0

        for v in self.variables:
            if not v.known:
                unknown_count += 1
                if unknown_count > 1:
                    return False

        return unknown_count == 1


class MassFlowBalance(BalanceEquation):
    """
    Mass flow conservation equation at a junction.

    Enforces conservation of mass according to:

        Σ(m_out) - Σ(m_in) = 0
    """

    __slots__ = ("solved",)

    name = "Mass Flow Balance"
    scale_factor = 1e1

    def solve(self):
        """
        Solve the mass flow balance for the single unknown variable.

        The unknown mass flow is computed as the difference between
        outgoing and incoming known mass flows.
        """

        m_total = 0.0
        unknown = None

        for v in self.variables:
            if v.known:
                if v.port_type == "out":
                    m_total += v.value
                else:
                    m_total -= v.value
            else:
                unknown = v

        if unknown is None:
            raise RuntimeError(
                "MassFlowBalance.solve() called with no unknown variable."
            )

        unknown.set_value(m_total)
        self.solved = True

    def residual(self):
        """
        Compute the residual of the mass flow balance.

        Returns
        -------
        float
            Scaled residual value.
        """

        res = 0.0
        for v in self.variables:
            if v.port_type == "out":
                res += v.value
            else:
                res -= v.value

        return res * self.scale_factor


class PressureEquality(BalanceEquation):
    """
    Pressure equality constraint between two connected ports.

    Enforces:
        p1 = p2
    """

    __slots__ = ("solved",)

    name = "Pressure Equality"
    scale_factor = 1e-5

    def solve(self):
        """
        Solve the pressure equality by propagating the known pressure
        to the unknown variable.
        """

        v1, v2 = self.variables

        if v1.known:
            v2.set_value(v1.value)
            self.solved = True
            return

        if v2.known:
            v1.set_value(v2.value)
            self.solved = True
            return

        raise RuntimeError(
            "PressureEquality.solve() called without a solvable unknown variable."
        )

    def residual(self):
        """
        Compute the residual of the pressure equality.

        Returns
        -------
        float
            Scaled residual value.
        """

        v1, v2 = self.variables
        return (v1.value - v2.value) * self.scale_factor


class EnthalpyEquality(BalanceEquation):
    """
    Enthalpy equality constraint between two connected ports.

    Enforces:
        h1 = h2
    """

    __slots__ = ("solved",)

    name = "Enthalpy Equality"
    scale_factor = 1e-5

    def solve(self):
        """
        Solve the enthalpy equality by propagating the known enthalpy
        to the unknown variable.
        """

        v1, v2 = self.variables

        if v1.known:
            v2.set_value(v1.value)
            self.solved = True
            return

        if v2.known:
            v1.set_value(v2.value)
            self.solved = True
            return

        raise RuntimeError(
            "EnthalpyEquality.solve() called with no unknown variable."
        )

    def residual(self):
        """
        Compute the residual of the enthalpy equality.

        Returns
        -------
        float
            Scaled residual value.
        """

        v1, v2 = self.variables
        return (v1.value - v2.value) * self.scale_factor


class EnthalpyFlowBalance(BalanceEquation):
    """
    Enthalpy flow balance equation at a junction.

    Enforces conservation of energy according to:

        Σ (s_i * m_i * h_i) = 0

    where:
        s_i = +1 for outgoing ports,
        s_i = -1 for incoming ports.
    """

    __slots__ = ("solved", "_pairs", "_multipliers")

    name = "Enthalpy Flow Balance"
    scale_factor = 1e-4

    def __init__(self, variables, fluid_loop):
        """
        Construct an enthalpy flow balance equation.

        Parameters
        ----------
        variables : list
            List of [m_var, h_var] pairs.
        fluid_loop : int
            Fluid loop identifier.
        """

        super().__init__(variables, fluid_loop)

        pairs = []
        multipliers = []

        for m_var, h_var in variables:
            pairs.append((m_var, h_var))
            multipliers.append(1 if m_var.port_type == "out" else -1)

        self._pairs = pairs
        self._multipliers = multipliers

    def solve(self):
        """
        Solve the enthalpy flow balance for exactly one unknown variable.

        The unknown may be either a mass flow or an enthalpy variable.
        """

        H_total = 0.0
        unknown_pair = None
        unknown_index = None

        for i, (m, h) in enumerate(self._pairs):
            if m.known and h.known:
                H_total += self._multipliers[i] * m.value * h.value
            else:
                unknown_pair = (m, h)
                unknown_index = i

        if unknown_pair is None:
            raise RuntimeError(
                "EnthalpyFlowBalance.solve() called with no unknown variable."
            )

        m_var, h_var = unknown_pair
        s = self._multipliers[unknown_index]

        if not m_var.known:
            m_var.set_value(-(H_total) / (h_var.value * s))
        elif not h_var.known:
            h_var.set_value(-(H_total) / (m_var.value * s))
        else:
            raise RuntimeError(
                "Unexpected state: both variables known but equation unsolved."
            )

        self.solved = True

    def residual(self):
        """
        Compute the residual of the enthalpy flow balance.

        Returns
        -------
        float
            Scaled residual value.
        """

        H_total = 0.0
        for (m, h), s in zip(self._pairs, self._multipliers):
            H_total += s * m.value * h.value

        return H_total * self.scale_factor


class Junction:
    """
    Junction node connecting multiple component ports belonging to the same fluid loop.

    A Junction represents a topological connection point in the thermal–hydraulic
    network. It is responsible for generating all local balance equations that
    enforce conservation laws and continuity conditions between connected ports.

    Responsibilities
    ----------------
    - Store connected ports
    - Store fluid loop identifier
    - Generate pressure, enthalpy, mass-flow and enthalpy-flow equations
    """

    __slots__ = ("jid", "fluid_loop", "ports", "equations")

    def __init__(self, jid: int, fluid_loop: int):
        """
        Parameters
        ----------
        jid : int
            Unique junction identifier.
        fluid_loop : int
            Fluid loop index associated with this junction.
        """
        self.jid = jid
        self.fluid_loop = fluid_loop
        self.ports: List[Port] = []
        self.equations = None

    def add_port(self, port: Port):
        """
        Attach a port to the junction.

        Parameters
        ----------
        port : Port
            Component port to be connected.
        """
        port.junction = self
        self.ports.append(port)

    def create_equations(self):
        """
        Generate all balance equations associated with this junction.

        The generated equation blocks preserve ordering compatibility with
        the tearing algorithm.

        Returns
        -------
        list
            Nested list of BalanceEquation instances grouped by equation type.
        """

        p_vars, h_vars, m_vars, mh_pairs = [], [], [], []

        in_ports = [p for p in self.ports if p.port_type == "in"]
        out_ports = [p for p in self.ports if p.port_type == "out"]

        # --------------------------------------------------
        # Collect inlet variables
        # --------------------------------------------------
        for p in in_ports:
            p_vars.append(p.p)
            h_vars.append(p.h)
            m_vars.append(p.m)
            mh_pairs.append([p.m, p.h])

        # --------------------------------------------------
        # Collect outlet variables
        # --------------------------------------------------
        for p in out_ports:
            p_vars.append(p.p)
            m_vars.append(p.m)
            mh_pairs.append([p.m, p.h])
            if len(in_ports) == 1:
                h_vars.append(p.h)

        # --------------------------------------------------
        # Pressure equalities
        # --------------------------------------------------
        pressure_eqs = []
        if p_vars:
            ref = p_vars[0]
            for v in p_vars[1:]:
                pressure_eqs.append(
                    PressureEquality([ref, v], self.fluid_loop)
                )

        # --------------------------------------------------
        # Enthalpy equalities (single-inlet case)
        # --------------------------------------------------
        enthalpy_eqs = []
        if len(in_ports) == 1 and h_vars:
            ref = h_vars[0]
            for v in h_vars[1:]:
                enthalpy_eqs.append(
                    EnthalpyEquality([ref, v], self.fluid_loop)
                )

        # --------------------------------------------------
        # Mass and enthalpy-flow balances
        # --------------------------------------------------
        mass_eq = MassFlowBalance(m_vars, self.fluid_loop)

        if len(in_ports) > 1:
            self.equations = [
                pressure_eqs,
                [EnthalpyFlowBalance(mh_pairs, self.fluid_loop)],
                [mass_eq],
            ]
        else:
            self.equations = [
                pressure_eqs,
                enthalpy_eqs,
                [mass_eq],
            ]

        return self.equations


class TripartiteGraph:
    """
    Tripartite graph representation of the thermal–hydraulic network.

    The graph consists of three disjoint node sets:

    - V : Variable nodes
    - U : Equation nodes
    - C : Component nodes

    And two edge sets:

    - E  : Undirected edges (Variable ↔ Equation)
    - Ed : Directed causal edges (Variable → Component, Component → Variable)

    This graph forms the basis for the tearing algorithm.
    """

    def __init__(self,
                 junctions: Dict[int, Junction],
                 components: Dict[str, Component]):
        """
        Parameters
        ----------
        junctions : dict[int, Junction]
            Mapping of junction IDs to Junction objects.
        components : dict[str, Component]
            Mapping of component labels to Component objects.
        """

        # --------------------------------------------------
        # Component nodes
        # --------------------------------------------------
        self.C = list(components.values())

        # --------------------------------------------------
        # Collect equation blocks
        # --------------------------------------------------
        equation_blocks = []
        for j in junctions.values():
            equation_blocks.append(j.equations)

        # --------------------------------------------------
        # Variable nodes
        # --------------------------------------------------
        self.V = []
        for blocks in equation_blocks:
            for eqs in blocks:
                for eq in eqs:
                    for v in eq.variables:
                        if isinstance(v, list):
                            # enthalpy-flow balance → store enthalpy variable
                            self.V.append(v[1])
                        else:
                            self.V.append(v)

        # --------------------------------------------------
        # Equation nodes
        # --------------------------------------------------
        self.U = [
            eq
            for blocks in equation_blocks
            for eqs in blocks
            for eq in eqs
        ]

        # --------------------------------------------------
        # Undirected edges (Variable ↔ Equation)
        # --------------------------------------------------
        self.E = []

        for u in self.U:
            if isinstance(u.variables[0], list):
                vars_flat = [x for pair in u.variables for x in pair]
            else:
                vars_flat = u.variables

            for v in vars_flat:
                self.E.append((v, u))

        # --------------------------------------------------
        # Directed edges: Variable → Component
        # --------------------------------------------------
        self.Ed = [
            (v, c)
            for c in self.C
            for v in self.V
            if v.port.component is c and (
                (
                    isinstance(c, PressureBasedComponent)
                    and (
                        (v.port_type == "in" and v.var_type in ("p", "h"))
                        or (v.port_type == "out" and v.var_type == "p")
                    )
                )
                or (
                    isinstance(c, MassFlowBasedComponent)
                    and v.port_type == "in"
                )
                or (
                    isinstance(c, BypassComponent)
                    and v.port_type == "in"
                )
            )
        ]

        # --------------------------------------------------
        # Directed edges: Component → Variable
        # --------------------------------------------------
        self.Ed.extend([
            (c, v)
            for c in self.C
            for v in self.V
            if v.port.component is c and (
                (
                    isinstance(c, PressureBasedComponent)
                    and (
                        v.var_type == "m"
                        or (v.var_type == "h" and v.port_type == "out")
                    )
                )
                or (
                    isinstance(c, MassFlowBasedComponent)
                    and v.port_type == "out"
                )
                or (
                    isinstance(c, BypassComponent)
                    and v.port_type == "out"
                )
            )
        ])


class BoundaryCondition:
    """
    Boundary condition applied to a port variable.

    A boundary condition constrains a single variable (pressure, enthalpy,
    or mass flow) at a port to a prescribed value. Boundary conditions may
    either be fixed or treated as optimization variables.

    This class acts as a lightweight wrapper around a :class:`Variable`
    instance and integrates seamlessly into the solver and optimization
    infrastructure.

    Parameters
    ----------
    var : Variable
        Variable to which the boundary condition is applied.
    value : float
        Boundary condition value.
    scale_factor : float, optional
        Scaling factor used when the boundary condition is treated as an
        optimization variable.
    initial_value : float, optional
        Initial value for optimization. If None, defaults to ``value``.
    is_var : bool, optional
        Flag indicating whether the boundary condition is an optimization
        variable.
    bounds : tuple of float, optional
        Lower and upper bounds for optimization.
    """

    __slots__ = ("var", "value", "initial_value", "scale_factor", "is_var", "bounds")

    def __init__(
        self,
        var,
        value,
        scale_factor: float = 1.0,
        initial_value=None,
        is_var: bool = False,
        bounds: tuple = (-np.inf, np.inf),
    ):
        self.var = var
        self.value = value
        self.initial_value = value
        self.scale_factor = scale_factor
        self.initial_value = initial_value
        self.is_var = is_var
        self.bounds = bounds

    def _apply(self):
        """
        Apply the boundary condition to the associated variable.

        This method directly sets the variable value and is called
        before each network evaluation.
        """
        self.var.set_value(self.value)

    def set_value(self, value):
        """
        Update the boundary condition value.

        Parameters
        ----------
        value : float
            New boundary condition value.
        """
        self.var.set_value(value)


class EqualityConstraint:
    """
    Equality constraint wrapper for the optimization problem.

    This class represents a user-defined equality constraint of the form

        g(x) = 0

    where the constraint function depends on the current state of the
    thermal–hydraulic network.

    The constraint is evaluated after the network has been solved for a
    given optimization vector.

    Parameters
    ----------
    network : Network
        Reference to the network instance.
    fun : callable
        User-defined function of the form ``fun(network)``.
        The function must return either a scalar or a NumPy array.
    scale_factor : float, optional
        Scaling factor applied to the constraint value.
    """

    __slots__ = ("fun", "network", "scale_factor")

    def __init__(self, network: "Network", fun: Callable, scale_factor: float = 1.0):
        if not callable(fun):
            raise TypeError("fun must be a callable python function")
        self.fun = fun
        self.network = network
        self.scale_factor = scale_factor

    def solve(self):
        """
        Evaluate the equality constraint.

        Returns
        -------
        float or numpy.ndarray
            Scaled constraint value.

        Raises
        ------
        TypeError
            If the constraint function does not return a valid type.
        """
        res = self.fun(self.network) * self.scale_factor
        if isinstance(res, (float, np.ndarray)):
            return res
        raise TypeError(
            "equality constraint solve method must return float or numpy array"
        )


class InequalityConstraint:
    """
    Inequality constraint wrapper for the optimization problem.

    This class represents a user-defined inequality constraint of the form

        g(x) >= 0

    where the constraint function depends on the current state of the
    thermal–hydraulic network.

    Parameters
    ----------
    network : Network
        Reference to the network instance.
    fun : callable
        User-defined function of the form ``fun(network)``.
        The function must return either a scalar or a NumPy array.
    scale_factor : float, optional
        Scaling factor applied to the constraint value.
    """

    __slots__ = ("fun", "network", "scale_factor")

    def __init__(self, network: "Network", fun: Callable, scale_factor: float = 1.0):
        if not callable(fun):
            raise TypeError("fun must be a callable python function")
        self.fun = fun
        self.network = network
        self.scale_factor = scale_factor

    def solve(self):
        """
        Evaluate the inequality constraint.

        Returns
        -------
        float or numpy.ndarray
            Scaled constraint value.

        Raises
        ------
        TypeError
            If the constraint function does not return a valid type.
        """
        res = self.fun(self.network) * self.scale_factor
        if isinstance(res, (float, np.ndarray)):
            return res
        raise TypeError(
            "inequality constraint solve method must return float or numpy array"
        )


class ObjectiveFun:
    """
    Objective function wrapper for the optimization problem.

    This class represents a scalar objective function of the form

        min f(x)

    where the objective depends on the current state of the
    thermal–hydraulic network.

    Parameters
    ----------
    network : Network
        Reference to the network instance.
    fun : callable
        User-defined objective function of the form ``fun(network)``.
        The function must return a scalar value.
    weight_factor : float, optional
        Weighting factor applied to the objective value.
    """

    __slots__ = ("fun", "network", "weight_factor")

    def __init__(self, network: "Network", fun: Callable, weight_factor: float = 1.0):
        if not callable(fun):
            raise TypeError("fun must be a callable python function")
        self.fun = fun
        self.network = network
        self.weight_factor = weight_factor

    def solve(self):
        """
        Evaluate the objective function.

        Returns
        -------
        float
            Weighted objective value.

        Raises
        ------
        TypeError
            If the objective function does not return a scalar.
        """
        res = self.fun(self.network) * self.weight_factor
        if isinstance(res, float):
            return res
        raise TypeError("objective function solve method must return float")


class Network:
    """
    Thermal–hydraulic network container.

    The Network class is the central orchestration layer of the framework.
    It manages components, junctions, fluid loops, boundary conditions,
    constraints, tearing, and the numerical solution process.

    Responsibilities
    ----------------
    - Component and junction registration
    - Fluid-loop management
    - Boundary condition handling
    - Loop breaker definition
    - Tripartite graph construction
    - Tearing algorithm execution
    - Solver orchestration
    """

    def __init__(self):
        """
        Initialize an empty thermal–hydraulic network.
        """
        self.components: Dict[str, Component] = {}
        self.junctions: Dict[int, Junction] = {}
        self.fluid_loops: Dict[int, str] = {}
        self.boundary_conditions: Dict[Variable, BoundaryCondition] = {}
        self.loop_breakers: Dict[int, int] = {}

        self._jid = 0  # internal junction counter

        # User-defined constraints and objectives
        self.econs = {}
        self.iecons = {}
        self.objs = {}

        # Solver structures (filled during initialization)
        self.Vt = []
        self.U = []
        self.exec_list = []
        self.res_equa = []
        self.no_design_equa = (False, 0)

    def add_component(self, comp: Component):
        """
        Register a component in the network.

        Parameters
        ----------
        comp : Component
            Component instance to be added.
        """
        self.components[comp.label] = comp

    def add_parameter(
        self,
        comp_label: str,
        param_label: str,
        value: float,
        scale_factor: float = 1.0,
        initial_value=None,
        is_var: bool = False,
        bounds: tuple = (-np.inf, np.inf),
    ):
        """
        Add a parameter to a component.

        Parameters
        ----------
        comp_label : str
            Component label.
        param_label : str
            Parameter name.
        value : float
            Parameter value.
        scale_factor : float, optional
            Scaling factor for optimization.
        initial_value : float, optional
            Initial value for optimization.
        is_var : bool, optional
            Flag indicating whether this parameter is optimized.
        bounds : tuple, optional
            Optimization bounds.
        """
        comp = self.components.get(comp_label)
        if comp is None:
            raise RuntimeError(f'Component "{comp_label}" not in Network!')

        comp.parameter[param_label] = Parameter(
            label=param_label,
            value=value,
            scale_factor=scale_factor,
            initial_value=initial_value,
            is_var=is_var,
            bounds=bounds,
        )

    def add_constraint(
        self,
        label: str,
        fun: Callable,
        ctype: str = "eq",
        scale_factor: float = 1.0,
        weight_factor: float = 1.0,
    ):
        """
        Add an equality, inequality, or objective function.

        Parameters
        ----------
        label : str
            Unique identifier.
        fun : callable
            Function of the form fun(network).
        ctype : {"eq", "ineq", "obj"}
            Constraint type.
        scale_factor : float, optional
            Scaling factor for constraints.
        weight_factor : float, optional
            Weight factor for objective functions.
        """
        if ctype == "eq":
            self.econs[label] = EqualityConstraint(self, fun, scale_factor)
        elif ctype == "ineq":
            self.iecons[label] = InequalityConstraint(self, fun, scale_factor)
        elif ctype == "obj":
            self.objs[label] = ObjectiveFun(self, fun, weight_factor)
        else:
            raise ValueError(f"Unknown constraint type '{ctype}'")

    def add_output(self, comp_label: str, output_label: str):
        """
        Register an output variable on a component.

        Parameters
        ----------
        comp_label : str
            Component label.
        output_label : str
            Output name.
        """
        comp = self.components.get(comp_label)
        if comp is None:
            raise RuntimeError(f'Component "{comp_label}" not in Network!')
        comp.outputs[output_label] = Output(output_label)

    def connect(self, cA: str, pA: str, cB: str, pB: str, fluid_loop: int):
        """
        Connect two component ports via a junction.

        Parameters
        ----------
        cA, cB : str
            Component labels.
        pA, pB : str
            Port names.
        fluid_loop : int
            Fluid loop index.
        """
        self._jid += 1

        portA = self.components[cA].ports[pA]
        portB = self.components[cB].ports[pB]

        portA.fluid = self.fluid_loops[fluid_loop]
        portB.fluid = self.fluid_loops[fluid_loop]

        junction = Junction(self._jid, fluid_loop)
        junction.add_port(portA)
        junction.add_port(portB)

        self.junctions[self._jid] = junction


    def set_bc(
        self,
        comp_label: str,
        port_name: str,
        var_type: str,
        value,
        fluid_loop: int,
        scale_factor: float = 1.0,
        initial_value=None,
        is_var: bool = False,
        bounds: tuple = (-np.inf, np.inf),
    ):
        """
        Apply a boundary condition to a port variable.

        Parameters
        ----------
        comp_label : str
            Component label.
        port_name : str
            Port name.
        var_type : {"p", "h", "m"}
            Variable type.
        value : float
            Boundary value.
        fluid_loop : int
            Fluid loop index.
        """
        port = self.components[comp_label].ports[port_name]
        port.fluid = self.fluid_loops[fluid_loop]

        var = getattr(port, var_type)
        bc = BoundaryCondition(
            var=var,
            value=value,
            scale_factor=scale_factor,
            initial_value=initial_value,
            is_var=is_var,
            bounds=bounds,
        )

        bc._apply()
        self.boundary_conditions[var] = bc

    def set_inital_values(self, x_init, x_bnds):
        """
        Set initial values and bounds for tearing variables.

        This method assigns user-defined initial values and bounds to
        the tearing variables identified by the tearing algorithm.
        It is typically used to improve convergence of the nonlinear
        solver or to provide physically meaningful starting points
        for inverse problems.

        Parameters
        ----------
        x_init : array-like
            Initial values for the tearing variables.
            The order must correspond to ``self.Vt``.
        x_bnds : array-like
            Bounds for the tearing variables.
            Each entry must be a tuple ``(lower, upper)`` and the order
            must correspond to ``self.Vt``.

        Notes
        -----
        - This method only affects tearing variables.
        - Parameter variables and boundary-condition variables are
          initialized separately.
        - No consistency checks on array length are performed in order
          to preserve the original framework behavior.
        """

        i = 0
        for var in self.Vt:
            var.initial_value = x_init[i]
            var.bounds = x_bnds[i]
            i += 1

    def set_loop_breaker(self, fluid_loop: int, junction_id: int):
        """
        Define a loop breaker by removing one mass balance equation
        in a closed fluid loop.

        Parameters
        ----------
        fluid_loop : int
            Fluid loop index.
        junction_id : int
            Junction ID at which the loop is broken.
        """
        if junction_id not in self.junctions:
            raise RuntimeError(f"Junction {junction_id} does not exist.")
        self.loop_breakers[fluid_loop] = junction_id

    def set_component_model(self, comp_label, model: Callable):
        self.components[comp_label].set_model(model)

    def print_tearing_variables(self):
        print("Tearing variables:")
        for var in self.Vt:
            print(f"{var} at component <{var.port.component.label}> , port <{var.port.name}>")

    def get_results(self):
        """
        Collect all solved port states, parameters, and outputs.

        Returns
        -------
        dict
            Structured results dictionary.
        """
        results = {
            "ports": {},
            "parameters": {},
            "outputs": {},
        }

        # Ports
        for cname, comp in self.components.items():
            results["ports"][cname] = {}
            for pname, port in comp.ports.items():
                results["ports"][cname][pname] = {
                    "p": port.p.get_value(),
                    "h": port.h.get_value(),
                    "m": port.m.get_value(),
                    "fluid": port.fluid,
                }

        # Parameters
        for cname, comp in self.components.items():
            results["parameters"][cname] = {
                pname: p.value for pname, p in comp.parameter.items()
            }

        # Outputs
        for cname, comp in self.components.items():
            results["outputs"][cname] = {
                oname: o.value for oname, o in comp.outputs.items()
            }

        return results

    def _apply_loop_breakers(self):
        """
        Apply user-defined loop breakers to the network.

        In closed fluid loops, the mass balance equations are linearly
        dependent, which leads to a singular system of equations.
        To avoid this, exactly one mass balance equation per closed
        fluid loop must be removed.

        This method removes the corresponding ``MassFlowBalance`` equation
        at a user-specified junction for each fluid loop.

        Notes
        -----
        - The user explicitly selects the loop breaker via
          :meth:`set_loop_breaker`.
        - Only mass flow balance equations are removed.
        - All other balance equations (pressure, enthalpy, enthalpy flow)
          remain untouched.
        - If no matching mass balance is found, an exception is raised
          to prevent silent model inconsistency.

        Raises
        ------
        RuntimeError
            If no mass flow balance equation is found at the specified
            junction for the given fluid loop.
        """

        for fluid_loop, jid in self.loop_breakers.items():

            junction = self.junctions[jid]
            removed = False

            for block in junction.equations:
                for eq in list(block):
                    if isinstance(eq, MassFlowBalance) and eq.fluid_loop == fluid_loop:
                        block.remove(eq)
                        removed = True
                        break
                if removed:
                    break

            if not removed:
                raise RuntimeError(
                    f"No MassFlowBalance found at Junction {jid} "
                    f"for fluid loop {fluid_loop}"
                )

    def initialize(self):
        """
        Initialize the network topology and solver structures.

        This method performs the following steps:

        1. Generate balance equations for all junctions.
        2. Apply user-defined loop breakers.
        3. Construct the tripartite graph representation.
        4. Execute the tearing algorithm.

        After calling this method, the network is ready for
        simulation or optimization.
        """

        # --------------------------------------------------
        # 1) Create junction balance equations
        # --------------------------------------------------
        for junction in self.junctions.values():
            junction.create_equations()

        # --------------------------------------------------
        # 2) Apply loop breakers (remove mass balances)
        # --------------------------------------------------
        self._apply_loop_breakers()

        # --------------------------------------------------
        # 3) Build tripartite graph
        # --------------------------------------------------
        self.tpg = TripartiteGraph(self.junctions, self.components)

        # --------------------------------------------------
        # 4) Execute tearing algorithm
        # --------------------------------------------------
        (
            self.Vt,
            self.exec_list,
            self.res_equa,
            self.no_design_equa,
        ) = self._tearing_alg(self.tpg)

        # print("Tearing variables:")
        # for var in self.Vt:
        #     print(f"{var} at {var.port.component.label}")

    def _tearing_alg(self, tpg: TripartiteGraph):
        """
        Execute the tearing algorithm.

        The tearing algorithm determines:

        - Vt: tearing variables
        - exec_list: execution order of equations and components
        - res_equa: residual equations
        - design_equa: number of required design equations

        Parameters
        ----------
        tpg : TripartiteGraph
            Tripartite graph representation of the network.

        Returns
        -------
        Vt : list of Variable
            Tearing variables.
        exec_list : list
            Ordered list of equations and components to execute.
        res_equa : list
            Residual equations.
        design_equa : tuple
            Flag and number of design equations required.
        """

        Vt, V, Vnew = [], [], []
        comp_exec = []
        comp_not_exec = tpg.C.copy()
        equa_solved = []
        exec_list = []
        res_equa = []

        # --------------------------------------------------
        # Main loop: continue until all components executed
        # --------------------------------------------------
        while len(comp_exec) != len(tpg.C):

            # ----------------------------------------------
            # Step 1: Select next component
            # ----------------------------------------------
            if any(c.modeling_type == "Pressure Based" for c in comp_not_exec):
                indices = [
                    i for i, c in enumerate(comp_not_exec)
                    if c.modeling_type == "Pressure Based"
                ]
                for i, index in enumerate(indices):
                    c = comp_not_exec[index]
                    if i == 0:
                        cid = index
                        n = len([e[0] for e in tpg.Ed if e[1] == c and e[0] not in V])
                    else:
                        n_new = len([e[0] for e in tpg.Ed if e[1] == c and e[0] not in V])
                        if n_new < n:
                            cid = index
                            n = n_new
            else:
                for i, c in enumerate(comp_not_exec):
                    n_new = len([e[0] for e in tpg.Ed if e[1] == c and e[0] not in V])
                    if i == 0:
                        c_old = c
                        n = n_new
                        cid = i
                    else:
                        if n_new < n:
                            c_old = c
                            n = n_new
                            cid = i
                        elif n_new == n:
                            if isinstance(c, PressureBasedComponent) and isinstance(
                                c_old, MassFlowBasedComponent
                            ):
                                c_old = c
                                cid = i

            # ----------------------------------------------
            # Step 2: Add tearing variables
            # ----------------------------------------------
            Vt += [e[0] for e in tpg.Ed if e[1] == comp_not_exec[cid] and e[0] not in V]
            V += [e[0] for e in tpg.Ed if e[1] == comp_not_exec[cid] and e[0] not in V]

            n_solved = len(equa_solved)
            n_executed = len(comp_exec)

            # ----------------------------------------------
            # Step 3: Propagate solvability
            # ----------------------------------------------
            while True:

                # 3A — Solve equations with one unknown
                for u in list(set(tpg.U).difference(set(equa_solved))):
                    if isinstance(u.variables[0], list):
                        flat = [x for pair in u.variables for x in pair]
                        if len(set(flat).difference(set(flat).intersection(set(V)))) == 1:
                            equa_solved.append(u)
                            exec_list.append(u)
                            V.extend(list(set(flat).difference(set(V))))
                    else:
                        if len(set(u.variables).difference(set(V))) == 1:
                            equa_solved.append(u)
                            exec_list.append(u)
                            V.extend(list(set(u.variables).difference(set(V))))

                # 3B — Find executable components
                comp_new = [
                    c for c in comp_not_exec
                    if c not in comp_exec
                    and all(v in V for v in [e[0] for e in tpg.Ed if e[1] == c])
                ]

                Vnew.extend([e[1] for e in tpg.Ed if e[0] in comp_new])

                # 3C — Identify residual equations
                for u in tpg.U:
                    v_list = []
                    for var in u.variables:
                        if isinstance(var, list):
                            v_list.extend(var)
                        else:
                            v_list.append(var)

                    if (
                        any(v in V and v in Vnew for v in v_list)
                        or (
                            any(v in Vnew for v in v_list)
                            and all(v in V or v in Vnew for v in v_list)
                            and any(v in Vt for v in v_list)
                        )
                    ):
                        res_equa.append(u)

                # 3D — Update working sets
                V.extend(Vnew)
                comp_exec.extend(comp_new)
                comp_not_exec = [c for c in comp_not_exec if c not in comp_exec]
                exec_list.extend(comp_new)
                Vnew = []

                if len(comp_exec) == n_executed and len(equa_solved) == n_solved:
                    break
                else:
                    n_executed = len(comp_exec)
                    n_solved = len(equa_solved)

        # --------------------------------------------------
        # Step 4: Design equation check
        # --------------------------------------------------
        if len(res_equa) < len(Vt):
            design_equa = (True, len(Vt) - len(res_equa))
        else:
            design_equa = (False, 0)

        return Vt, exec_list, res_equa, design_equa

    def solve_system(self, acc: float = 1e-6, max_iter: int = 500):
        """
        Solve the thermal–hydraulic network using SLSQP.

        This method assembles the optimization vector consisting of
        tearing variables, parameter variables and boundary-condition
        variables, and solves the resulting nonlinear constrained
        optimization problem.

        Parameters
        ----------
        acc : float, optional
            Convergence tolerance for the optimizer.
        max_iter : int, optional
            Maximum number of iterations.

        Returns
        -------
        OptimizeResult
            Result object returned by ``scipy.optimize.minimize``.
        """

        print("\n Start solver...")

        self._add_vars()

        vars_all = self.Vt + self.U
        n = len(vars_all)

        x0 = np.zeros(n)

        if n == len(self.Vt):

            for i, v in enumerate(vars_all):

                x0[i] = v.initial_value * v.scale_factor

            sol = scipy.optimize.root(self._solve_econs, x0)

            if sol.success:
                self._solve(sol["x"])
                print("\n Solver converged successfully")
            else:
                print("\n Solver did not converged successfully")

        else:

            bounds = []

            for i, v in enumerate(vars_all):
                x0[i] = v.initial_value * v.scale_factor
                bounds.append((
                    v.bounds[0] * v.scale_factor,
                    v.bounds[1] * v.scale_factor
                ))

            constraints = []

            if self.res_equa or self.econs:
                constraints.append({
                    "type": "eq",
                    "fun": self._solve_econs,
                    "jac": lambda x: grad_econ(x),
                })

            if self.iecons:
                constraints.append({
                    "type": "ineq",
                    "fun": self._solve_iecons,
                    "jac": lambda x: grad_iecon(x),
                })

            eps_vec = np.array([1e-6 * max(abs(x0[i]), 1.0) for i in range(n)])

            clones = [deepcopy(self) for _ in range(n + 1)]
            pool = mp.Pool(processes=min(mp.cpu_count(), n + 1))

            def grad_obj(x):
                return self._compute_fd_gradient(
                    clones, x, eps_vec, Network._solve_objs, pool
                )

            def grad_econ(x):
                return self._compute_fd_gradient(
                    clones, x, eps_vec, Network._solve_econs, pool, mode="econ"
                )

            def grad_iecon(x):
                return self._compute_fd_gradient(
                    clones, x, eps_vec, Network._solve_iecons, pool, mode="iecon"
                )

            sol = scipy.optimize.minimize(
                fun=self._solve_objs,
                x0=x0,
                bounds=bounds,
                jac=lambda x: grad_obj(x),
                constraints=constraints,
                method="SLSQP",
                options={
                    "ftol": acc,
                    "maxiter": max_iter,
                    "disp": True,
                },
                callback=self._callback,
            )

            pool.close()
            pool.join()

            if sol["success"]:
                self._solve(sol["x"])
                print("\n Solver converged successfully")
            else:
                print("\n Solver did not converged successfully")

        print(sol)

    def solve_system_notebook(self, acc=1e-6, max_iter=50):
        """
        Solve the network using a minimal, notebook-safe configuration.

        This solver variant disables:
        - numerical finite-difference gradients
        - multiprocessing

        It is intended for demonstration purposes (e.g. Jupyter notebooks)
        and produces identical solutions for well-posed problems, albeit
        with reduced performance.
        """

        print("\n Start solver...")

        self._add_vars()
        vars_all = self.Vt + self.U
        n = len(vars_all)

        x0 = np.zeros(n)

        if n == len(self.Vt):

            for i, v in enumerate(vars_all):

                x0[i] = v.initial_value * v.scale_factor

            sol = scipy.optimize.root(self._solve_econs, x0)

            if sol.success:
                self._solve(sol["x"])
                print("\n Solver converged successfully")
            else:
                print("\n Solver did not converged successfully")

        else:

            bounds = []

            for i, v in enumerate(vars_all):
                x0[i] = v.initial_value * v.scale_factor
                bounds.append((
                    v.bounds[0] * v.scale_factor,
                    v.bounds[1] * v.scale_factor
                ))

            constraints = []

            if self.res_equa or self.econs:
                constraints.append({
                    "type": "eq",
                    "fun": lambda x: self._solve_econs(x),
                })

            if self.iecons:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x: self._solve_iecons(x),
                })

            # -------------------------------------------------
            # Solve (SLSQP without gradients)
            # -------------------------------------------------
            sol = scipy.optimize.minimize(
                fun=lambda x: self._solve_objs(x),
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
                options={
                    "ftol": acc,
                    "maxiter": max_iter,
                    "disp": True,
                },
                callback=self._callback,
            )
            if sol["success"]:
                self._solve(sol["x"])
                print("\n Solver converged successfully")
            else:
                print("\n Solver did not converged successfully")

        print(sol)

    def _solve(self, x):
        """
        Execute a single network evaluation.

        This method assigns all optimization variables, applies fixed
        boundary conditions, and executes equations and components
        in tearing order.

        Parameters
        ----------
        x : ndarray
            Optimization variable vector.
        """

        idx = 0

        # 1) Assign tearing variables
        for v in self.Vt:
            v.set_value(x[idx] / v.scale_factor)
            idx += 1

        # 2) Assign parameter and BC variables
        for var in self.U:
            var.set_value(x[idx] / var.scale_factor)
            idx += 1

        # 3) Apply fixed boundary conditions
        for bc in self.boundary_conditions.values():
            if not bc.is_var:
                bc._apply()

        # 4) Execute in tearing order
        for item in self.exec_list:
            item.solve()

    def _solve_econs(self, x):
        """
        Evaluate equality constraints and residual equations.

        Parameters
        ----------
        x : ndarray
            Optimization variable vector.

        Returns
        -------
        ndarray
            Residual vector.
        """

        self._solve(x)

        res = []

        for eq in self.res_equa:
            res.append(eq.residual())

        for c in self.econs.values():
            val = c.solve()
            if isinstance(val, (list, tuple, np.ndarray)):
                res.extend(val)
            else:
                res.append(val)

        for comp in self.components.values():
            comp.reset()

        return np.asarray(res)

    def _solve_iecons(self, x):
        """
        Evaluate inequality constraints.

        Parameters
        ----------
        x : ndarray
            Optimization variable vector.

        Returns
        -------
        ndarray
            Inequality constraint vector.
        """

        self._solve(x)

        res = []

        for c in self.iecons.values():
            val = c.solve()
            if isinstance(val, (list, tuple, np.ndarray)):
                res.extend(val)
            else:
                res.append(val)

        for comp in self.components.values():
            comp.reset()

        return np.asarray(res)

    def _solve_objs(self, x):
        """
        Evaluate objective function.

        Parameters
        ----------
        x : ndarray
            Optimization variable vector.

        Returns
        -------
        float
            Objective function value.
        """

        self._solve(x)

        total = 0.0
        for obj in self.objs.values():
            total += obj.solve()

        for comp in self.components.values():
            comp.reset()

        return total

    def _add_vars(self):
        """
        Collect all optimization variables.

        This includes:
        - component parameters
        - boundary-condition variables
        """

        for comp in self.components.values():
            for p in comp.parameter.values():
                if p.is_var:
                    self.U.append(p)

        for bc in self.boundary_conditions.values():
            if bc.is_var:
                self.U.append(bc)

    def _callback(self, x):
        """
        Callback function for the optimizer.

        Parameters
        ----------
        x : ndarray
            Current optimization vector.
        """
        r = np.linalg.norm(self._solve_econs(x))
        f = self._solve_objs(x)
        print(
            f"Residual norm = {r:.3e}, Obj = {f:.3e}"
        )

    def _compute_fd_gradient(
        self,
        clones,
        x,
        eps_vec,
        eval_fun,
        pool,
        mode=None,
    ):
        """
        Compute finite-difference gradients in parallel.

        Parameters
        ----------
        clones : list of Network
            Deep copies of the network.
        x : ndarray
            Base optimization vector.
        eps_vec : ndarray
            Perturbation sizes.
        eval_fun : callable
            Evaluation function.
        pool : multiprocessing.Pool
            Worker pool.
        mode : {"econ", "iecon"}, optional
            Constraint evaluation mode.

        Returns
        -------
        ndarray
            Gradient matrix or vector.
        """

        f0 = eval_fun(clones[0], x)

        tasks = []
        for i, eps in enumerate(eps_vec):
            x_p = x.copy()
            x_p[i] += eps
            if mode is None:
                tasks.append((clones[i + 1], x_p))
            else:
                tasks.append((clones[i + 1], x_p, mode))

        if mode is None:
            f_fw = pool.map(_worker_eval_obj, tasks)
            f_fw = np.array(f_fw)
            grad = (f_fw - f0) / eps_vec
        else:
            f_fw = pool.map(_worker_eval_vec, tasks)
            f_fw = np.vstack(f_fw)
            f0 = np.asarray(f0)
            grad = (f_fw - f0) / eps_vec[:, None]
            grad = grad.T

        return grad


def _worker_eval_obj(args):
    """
    Worker function for parallel objective-function evaluation.

    This function is used in the finite-difference gradient computation
    of the objective function. Each worker operates on a deep copy of
    the network to ensure thread/process safety.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - clone : Network
            Deep copy of the network instance.
        - x : ndarray
            Optimization variable vector.

    Returns
    -------
    float
        Objective function value evaluated on the cloned network.
    """
    clone, x = args
    return clone._solve_objs(x)


def _worker_eval_vec(args):
    """
    Worker function for parallel constraint evaluation.

    This function is used in the finite-difference gradient computation
    of equality and inequality constraints. Each worker evaluates either
    the equality or inequality constraints on an independent clone of
    the network.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - clone : Network
            Deep copy of the network instance.
        - x : ndarray
            Optimization variable vector.
        - mode : {"econ", "iecon"}
            Evaluation mode:
            - "econ": equality constraints
            - "iecon": inequality constraints

    Returns
    -------
    numpy.ndarray
        Constraint vector evaluated on the cloned network.
    """
    clone, x, mode = args
    if mode == "econ":
        return clone._solve_econs(x)
    else:
        return clone._solve_iecons(x)