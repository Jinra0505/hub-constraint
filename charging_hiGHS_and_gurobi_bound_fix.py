from typing import Any, Dict, Tuple

from .assignment import aggregate_ev_energy_demand, aggregate_ev_energy_demand_components, aggregate_evtol_demand, aggregate_vt_departure_flow_by_class, compute_evtol_energy_demand

HAS_GUROBI = False
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

HAS_SCIPY = False
SCIPY_VERSION = None
SCIPY_IMPORT_ERROR = None
try:
    import numpy as np
    import scipy
    from scipy.optimize import linprog
except Exception as exc:
    HAS_SCIPY = False
    SCIPY_IMPORT_ERROR = repr(exc)
else:
    HAS_SCIPY = True
    SCIPY_VERSION = getattr(scipy, "__version__", "unknown")

LAST_SOLVER_USED = "unknown"
LAST_SHARED_SOLVER_USED = "unknown"
LAST_SHARED_POWER_SOLVER_USED = "unknown"


class LPFailed(RuntimeError):
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        super().__init__(f"Shared power HiGHS LP failed: {payload.get('message', 'unknown error')}")


def _hybrid_station_list(data: Dict[str, Any]) -> list[str]:
    """Return stations modeled in VT/shared-power subproblems.

    This model requires ``sets.hybrid_stations`` to be explicitly provided.
    No fallback to ``sets.stations`` is allowed for VT semantics.
    """
    hs = data.get("sets", {}).get("hybrid_stations")
    if not isinstance(hs, list) or not hs:
        raise ValueError("Missing required key path: sets.hybrid_stations for VT/shared-power modeling")
    return [str(x) for x in hs]




def _terminal_soc_target(storage_cfg: Dict[str, Any], config: Dict[str, Any], station: str) -> float:
    """Terminal SOC lower bound target in kWh for one station.

    Policies:
    - at_least_min (legacy)
    - at_least_init (default)
    - target (uses terminal_soc_target_kwh, scalar or by station mapping)
    """
    policy = str(config.get("terminal_soc_policy", "at_least_init"))
    b_min = float(storage_cfg.get("B_min", 0.0))
    b_init = float(storage_cfg.get("B_init", b_min))
    if policy == "at_least_min":
        return b_min
    if policy == "target":
        raw = config.get("terminal_soc_target_kwh", b_init)
        if isinstance(raw, dict):
            return float(raw.get(station, b_init))
        return float(raw)
    return b_init



def _resolve_time_value(raw: Any, t: int, default: float | None = None) -> float | None:
    if isinstance(raw, dict):
        if t in raw:
            return float(raw[t])
        ts = str(t)
        if ts in raw:
            return float(raw[ts])
        return None if default is None else float(default)
    if raw is None:
        return None if default is None else float(default)
    return float(raw)


def _effective_station_power_cap(data: Dict[str, Any], station: str, t: int) -> float:
    runtime_caps = data.get("diagnostics_runtime", {}).get("station_power_cap_effective", {})
    if station in runtime_caps:
        cap = _resolve_time_value(runtime_caps.get(station, {}), t, None)
        if cap is not None:
            return max(0.0, float(cap))
    station_cfg = data.get("parameters", {}).get("stations", {}).get(station, {})
    return max(0.0, float(_resolve_time_value(station_cfg.get("P_site", {}), t, 0.0) or 0.0))


def _vt_charge_power_upper_bound(
    data: Dict[str, Any],
    dep: str,
    t: int,
    e_dep: Dict[str, Dict[int, float]] | None,
) -> float:
    """Non-binding upper bound for VT charging power.

    Important: this is *not* the shared site cap. The actual site power coupling is
    enforced by the shared station constraint. Using the same ``P_site`` value both
    here and in the shared cap row creates a duplicate bottleneck and can push the
    marginal value onto variable bounds instead of the station-cap constraint, which
    suppresses the intended shadow price signal.

    Priority:
    1) explicit parameters.vt_charge_power_cap[dep][t] / scalar
    2) a safe dynamic upper bound from storage headroom and peak cycle demand
    3) a loose multiple of site power for numerical stability
    """
    params = data.get("parameters", {})
    cfg = data.get("config", {})
    explicit_block = params.get("vt_charge_power_cap", {})
    explicit = None
    if isinstance(explicit_block, dict) and dep in explicit_block:
        explicit = _resolve_time_value(explicit_block.get(dep), t)
    if explicit is not None:
        return max(0.0, float(explicit))

    storage_cfg = params.get("vertiport_storage", {}).get(dep, {})
    station_cfg = params.get("stations", {}).get(dep, {})
    dt = max(1.0e-9, float(data.get("meta", {}).get("delta_t", 1.0)))
    eta = max(1.0e-6, float(storage_cfg.get("eta_ch", 1.0)))
    b_min = float(storage_cfg.get("B_min", 0.0))
    b_max = float(storage_cfg.get("B_max", b_min))
    site_cap = float(_resolve_time_value(station_cfg.get("P_site", {}), t, 0.0) or 0.0)

    e_dep_map = (e_dep or {}).get(dep, {})
    peak_cycle_kwh = max((float(v) for v in e_dep_map.values()), default=0.0)
    dynamic_ub = (max(0.0, b_max - b_min) + max(0.0, peak_cycle_kwh)) / eta / dt

    factor = float(cfg.get("vt_charge_power_ub_site_factor", 2.0) or 2.0)
    floor_kw = float(cfg.get("vt_charge_power_ub_floor_kw", 1.0) or 1.0)
    return max(floor_kw, dynamic_ub, factor * site_cap)


def compute_station_loads_from_flows(
    data: Dict[str, Any],
    itineraries: list[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: list[int],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Aggregate EV/eVTOL station loads directly from itinerary flows.

    Includes EV access charging loads from multimodal EV_to_eVTOL itineraries.

    Returns a dictionary with:
    - E_ev_req[s][t], P_ev_req_kw[s][t]
    - E_ev_pure_req[s][t], E_ev_access_req[s][t]
    - P_ev_pure_req_kw[s][t], P_ev_access_req_kw[s][t]
    - E_vt_req[s][t], P_vt_req_kw_energy[s][t], P_vt_req_kw_grid[s][t]
    - P_total_req[s][t]
    """
    ev_stations = [str(x) for x in data.get("sets", {}).get("ev_stations", [])]
    hybrid_stations = _hybrid_station_list(data)
    delta_t = float(data["meta"]["delta_t"])

    d_vt_route = aggregate_evtol_demand(flows, itineraries, times)
    e_vt_dep = compute_evtol_energy_demand(d_vt_route, itineraries, times)
    e_ev_components = aggregate_ev_energy_demand_components(itineraries, flows, times)
    e_ev_station = e_ev_components["total"]
    e_ev_pure = e_ev_components["pure_ev"]
    e_ev_access = e_ev_components["access_ev"]
    vt_departure_flow_by_class = aggregate_vt_departure_flow_by_class(itineraries, flows, times)

    E_ev_req = {s: {t: 0.0 for t in times} for s in ev_stations}
    E_ev_pure_req = {s: {t: 0.0 for t in times} for s in ev_stations}
    E_ev_access_req = {s: {t: 0.0 for t in times} for s in ev_stations}
    E_vt_req = {s: {t: 0.0 for t in times} for s in hybrid_stations}
    P_ev_req = {s: {t: 0.0 for t in times} for s in ev_stations}
    P_ev_pure_req = {s: {t: 0.0 for t in times} for s in ev_stations}
    P_ev_access_req = {s: {t: 0.0 for t in times} for s in ev_stations}
    P_vt_req_energy = {s: {t: 0.0 for t in times} for s in hybrid_stations}
    P_vt_req_grid = {s: {t: 0.0 for t in times} for s in hybrid_stations}
    P_total_req = {s: {t: 0.0 for t in times} for s in ev_stations}

    for s in ev_stations:
        for t in times:
            E_ev_req[s][t] = float(e_ev_station.get(s, {}).get(t, 0.0))
            E_ev_pure_req[s][t] = float(e_ev_pure.get(s, {}).get(t, 0.0))
            E_ev_access_req[s][t] = float(e_ev_access.get(s, {}).get(t, 0.0))
            P_ev_req[s][t] = E_ev_req[s][t] / delta_t if delta_t > 0 else 0.0
            P_ev_pure_req[s][t] = E_ev_pure_req[s][t] / delta_t if delta_t > 0 else 0.0
            P_ev_access_req[s][t] = E_ev_access_req[s][t] / delta_t if delta_t > 0 else 0.0
            P_total_req[s][t] = P_ev_req[s][t]

    for s in hybrid_stations:
        for t in times:
            E_vt_req[s][t] = float(e_vt_dep.get(s, {}).get(t, 0.0))
            eta = 1.0
            if s in data["parameters"].get("vertiport_storage", {}):
                eta = max(1e-6, float(data["parameters"]["vertiport_storage"][s].get("eta_ch", 1.0)))
            P_vt_req_energy[s][t] = E_vt_req[s][t] / delta_t if delta_t > 0 else 0.0
            P_vt_req_grid[s][t] = E_vt_req[s][t] / (eta * delta_t) if delta_t > 0 else 0.0
            P_total_req.setdefault(s, {tt: 0.0 for tt in times})[t] = P_ev_req.get(s, {}).get(t, 0.0) + P_vt_req_grid[s][t]

    return {
        "E_ev_req": E_ev_req,
        "E_ev_pure_req": E_ev_pure_req,
        "E_ev_access_req": E_ev_access_req,
        "E_vt_req": E_vt_req,
        "P_ev_req": P_ev_req,
        "P_ev_req_kw": P_ev_req,
        "P_ev_pure_req_kw": P_ev_pure_req,
        "P_ev_access_req_kw": P_ev_access_req,
        "P_vt_req": P_vt_req_grid,
        "P_vt_req_kw_energy": P_vt_req_energy,
        "P_vt_req_kw_grid": P_vt_req_grid,
        "P_total_req": P_total_req,
        "vt_departure_flow_by_class": vt_departure_flow_by_class,
    }


def _solve_shared_power_core_gurobi(
    data: Dict[str, Any],
    times: list[int],
    e_dep: Dict[str, Dict[int, float]],
    ev_energy: Dict[str, Dict[int, float]],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
]:
    global LAST_SHARED_SOLVER_USED, LAST_SHARED_POWER_SOLVER_USED
    LAST_SHARED_SOLVER_USED = "gurobi"
    LAST_SHARED_POWER_SOLVER_USED = "gurobi"

    stations = _hybrid_station_list(data)
    delta_t = data["meta"]["delta_t"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    if storage_params is None and e_dep:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    model = gp.Model("shared_power_inventory")
    model.setParam("OutputFlag", 0)

    B = {}
    P_vt = {}
    shed_vt = {}
    for dep in e_dep:
        if dep not in storage_params:
            raise ValueError(f"Missing required key path: parameters.vertiport_storage.{dep}")
        for t in times:
            B[dep, t] = model.addVar(
                lb=storage_params[dep]["B_min"],
                ub=storage_params[dep]["B_max"],
                name=f"B_{dep}_{t}",
            )
            P_vt[dep, t] = model.addVar(
                lb=0.0,
                ub=_vt_charge_power_upper_bound(data, dep, t, e_dep),
                name=f"P_vt_{dep}_{t}",
            )
            shed_vt[dep, t] = model.addVar(lb=0.0, name=f"shed_vt_{dep}_{t}")

    shed_ev = {}
    for s in stations:
        for t in times:
            shed_ev[s, t] = model.addVar(lb=0.0, name=f"shed_ev_{s}_{t}")

    voll_ev_cfg = data.get("config", {}).get("voll_ev_per_kwh")
    voll_vt_cfg = data.get("config", {}).get("voll_vt_per_kwh")
    voll_ev_per_kwh = float(voll_ev_cfg) if voll_ev_cfg is not None else 50.0
    voll_vt_per_kwh = float(voll_vt_cfg) if voll_vt_cfg is not None else 200.0
    model.setObjective(
        gp.quicksum(prices[dep][t] * P_vt[dep, t] * delta_t for dep in e_dep for t in times)
        + gp.quicksum((voll_ev_per_kwh - prices[s][t]) * shed_ev[s, t] * delta_t for s in stations for t in times)
        + gp.quicksum(voll_vt_per_kwh * shed_vt[dep, t] for dep in e_dep for t in times),
        GRB.MINIMIZE,
    )

    for dep in e_dep:
        model.addConstr(B[dep, times[0]] == storage_params[dep]["B_init"], name=f"B_init_{dep}")
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                model.addConstr(
                    B[dep, times[idx + 1]]
                    == B[dep, t] + eta * P_vt[dep, t] * delta_t - (e_dep[dep][t] - shed_vt[dep, t]),
                    name=f"B_bal_{dep}_{t}",
                )
        t_last = times[-1]
        b_terminal_target = _terminal_soc_target(storage_params[dep], data.get("config", {}), dep)
        model.addConstr(
            B[dep, t_last] + eta * P_vt[dep, t_last] * delta_t - (e_dep[dep][t_last] - shed_vt[dep, t_last])
            >= b_terminal_target,
            name=f"B_terminal_{dep}_{t_last}",
        )

    power_constraints = {}
    for s in stations:
        for t in times:
            p_ev_req = ev_energy.get(s, {}).get(t, 0.0) / delta_t
            p_vt_sum = gp.quicksum(P_vt[dep, t] for dep in e_dep if dep == s)
            power_constraints[s, t] = model.addConstr(
                p_vt_sum + p_ev_req - shed_ev[s, t] <= _effective_station_power_cap(data, s, t),
                name=f"P_shared_{s}_{t}",
            )
            model.addConstr(shed_ev[s, t] <= p_ev_req, name=f"shed_ev_ub_{s}_{t}")

    for dep in e_dep:
        for t in times:
            model.addConstr(shed_vt[dep, t] <= e_dep[dep][t], name=f"shed_vt_ub_{dep}_{t}")

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"Shared power LP did not solve to optimality: status={model.Status}")

    B_out = {dep: {t: float(B[dep, t].X) for t in times} for dep in e_dep}
    P_out = {dep: {t: float(P_vt[dep, t].X) for t in times} for dep in e_dep}
    shed_ev_out = {s: {t: float(shed_ev[s, t].X) for t in times} for s in stations}
    shed_vt_out = {dep: {t: float(shed_vt[dep, t].X) for t in times} for dep in e_dep}
    shadow_prices = {
        s: {t: max(0.0, -float(power_constraints[s, t].Pi)) for t in times}
        for s in stations
    }

    residuals = {"INV1": 0.0, "INV2": 0.0, "INV3": 0.0, "INV4": 0.0}
    for dep in e_dep:
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = B_out[dep][times[idx + 1]]
                rhs = B_out[dep][t] + eta * P_out[dep][t] * delta_t - (float(e_dep.get(dep, {}).get(t, 0.0)) - shed_vt_out[dep][t])
                residuals["INV1"] = max(residuals["INV1"], abs(lhs - rhs))
            residuals["INV2"] = max(
                residuals["INV2"],
                max(storage_params[dep]["B_min"] - B_out[dep][t], B_out[dep][t] - storage_params[dep]["B_max"], 0.0),
            )
    for s in stations:
        for t in times:
            p_ev_req = ev_energy.get(s, {}).get(t, 0.0) / delta_t
            p_vt_sum = sum(P_out.get(dep, {}).get(t, 0.0) for dep in e_dep if dep == s)
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, p_vt_sum + p_ev_req - shed_ev_out[s][t] - _effective_station_power_cap(data, s, t)),
            )
            residuals["INV4"] = max(residuals["INV4"], max(0.0, shed_ev_out[s][t]))

    return B_out, P_out, shed_ev_out, shed_vt_out, shadow_prices, residuals



def solve_shared_power_inventory_highs(
    data: Dict[str, Any],
    times: list[int],
    e_dep: Dict[str, Dict[int, float]],
    ev_energy: Dict[str, Dict[int, float]],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
    Dict[str, Any],
]:
    global LAST_SHARED_SOLVER_USED, LAST_SHARED_POWER_SOLVER_USED
    if not HAS_SCIPY:
        raise RuntimeError(f"SciPy/HiGHS is required for LP-dual shared-power pricing, import failed: {SCIPY_IMPORT_ERROR}")
    LAST_SHARED_SOLVER_USED = "highs"
    LAST_SHARED_POWER_SOLVER_USED = "highs"

    stations = _hybrid_station_list(data)
    delta_t = float(data["meta"]["delta_t"])
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    if storage_params is None and e_dep:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    deps = [str(dep) for dep in storage_params.keys()]
    if not times:
        raise RuntimeError("Time set is empty")
    t_terminal = times[-1] + 1
    times_ext = list(times) + [t_terminal]

    voll_ev_cfg = data.get("config", {}).get("voll_ev_per_kwh")
    voll_vt_cfg = data.get("config", {}).get("voll_vt_per_kwh")
    voll_ev_per_kwh = float(voll_ev_cfg) if voll_ev_cfg is not None else 50.0
    voll_vt_per_kwh = float(voll_vt_cfg) if voll_vt_cfg is not None else 200.0

    var_idx: Dict[tuple[str, str, int], int] = {}
    bounds = []
    c = []

    def add_var(kind: str, key: str, t: int, lb: float, ub: float, coeff: float) -> None:
        idx = len(c)
        var_idx[(kind, key, t)] = idx
        c.append(float(coeff))
        bounds.append((float(lb), float(ub)))

    p_ev_req_kw = {s: {t: float(ev_energy.get(s, {}).get(t, 0.0)) / max(1.0e-9, delta_t) for t in times} for s in stations}

    for dep in deps:
        if dep not in storage_params:
            raise ValueError(f"Missing required key path: parameters.vertiport_storage.{dep}")
        for tau in times_ext:
            add_var("B", dep, tau, storage_params[dep]["B_min"], storage_params[dep]["B_max"], 0.0)
        for t in times:
            add_var("P", dep, t, 0.0, _vt_charge_power_upper_bound(data, dep, t, e_dep), prices[dep][t] * delta_t)
            add_var("SVT", dep, t, 0.0, float(e_dep.get(dep, {}).get(t, 0.0)), voll_vt_per_kwh)

    for s in stations:
        for t in times:
            add_var("SEV", s, t, 0.0, p_ev_req_kw[s][t], (voll_ev_per_kwh - prices[s][t]) * delta_t)

    n = len(c)
    A_eq = []
    b_eq = []

    for dep in deps:
        row = [0.0] * n
        row[var_idx[("B", dep, times_ext[0])]] = 1.0
        A_eq.append(row)
        b_eq.append(float(storage_params[dep]["B_init"]))

        eta = float(storage_params[dep]["eta_ch"])
        for idx, t in enumerate(times):
            t_next = times_ext[idx + 1]
            row = [0.0] * n
            row[var_idx[("B", dep, t_next)]] = 1.0
            row[var_idx[("B", dep, t)]] = -1.0
            row[var_idx[("P", dep, t)]] = -eta * delta_t
            row[var_idx[("SVT", dep, t)]] = -1.0
            A_eq.append(row)
            b_eq.append(-float(e_dep.get(dep, {}).get(t, 0.0)))

    A_ub = []
    b_ub = []
    # Terminal SOC floor to avoid free depletion of initial storage.
    for dep in deps:
        b_terminal_target = _terminal_soc_target(storage_params[dep], data.get("config", {}), dep)
        row = [0.0] * n
        row[var_idx[("B", dep, t_terminal)]] = -1.0
        A_ub.append(row)
        b_ub.append(-float(b_terminal_target))

    cap_rows = []
    for s in stations:
        for t in times:
            row = [0.0] * n
            for dep in deps:
                if dep == s:
                    row[var_idx[("P", dep, t)]] += 1.0
            row[var_idx[("SEV", s, t)]] = -1.0
            A_ub.append(row)
            b_ub.append(float(_effective_station_power_cap(data, s, t)) - p_ev_req_kw[s][t])
            cap_rows.append((s, t, len(A_ub) - 1))

    A_ub_np = np.array(A_ub, dtype=float) if A_ub else None
    b_ub_np = np.array(b_ub, dtype=float) if b_ub else None
    A_eq_np = np.array(A_eq, dtype=float) if A_eq else None
    b_eq_np = np.array(b_eq, dtype=float) if b_eq else None

    shape_payload = {
        "n_vars": n,
        "A_ub_shape": tuple(A_ub_np.shape) if A_ub_np is not None else None,
        "A_eq_shape": tuple(A_eq_np.shape) if A_eq_np is not None else None,
        "n_bounds": len(bounds),
    }

    try:
        res = linprog(
            c=np.array(c, dtype=float),
            A_ub=A_ub_np,
            b_ub=b_ub_np,
            A_eq=A_eq_np,
            b_eq=b_eq_np,
            bounds=bounds,
            method="highs",
        )
    except Exception as exc:
        raise LPFailed({
            "status": None,
            "message": repr(exc),
            "fun": None,
            "nit": None,
            "max_ub_violation": None,
            "max_eq_violation": None,
            **shape_payload,
        }) from exc

    if not res.success:
        x = getattr(res, "x", None)
        max_ub_violation = None
        max_eq_violation = None
        if x is not None:
            try:
                if A_ub_np is not None and b_ub_np is not None:
                    max_ub_violation = float(np.max(A_ub_np @ x - b_ub_np))
                if A_eq_np is not None and b_eq_np is not None:
                    max_eq_violation = float(np.max(np.abs(A_eq_np @ x - b_eq_np)))
            except Exception:
                max_ub_violation = None
                max_eq_violation = None
        raise LPFailed({
            "status": getattr(res, "status", None),
            "message": getattr(res, "message", "linprog failed"),
            "fun": getattr(res, "fun", None),
            "nit": getattr(res, "nit", None),
            "max_ub_violation": max_ub_violation,
            "max_eq_violation": max_eq_violation,
            **shape_payload,
        })

    x = res.x
    B_out = {dep: {tau: float(x[var_idx[("B", dep, tau)]]) for tau in times_ext} for dep in deps}
    P_out = {dep: {t: float(x[var_idx[("P", dep, t)]]) for t in times} for dep in deps}
    shed_vt_out = {dep: {t: float(x[var_idx[("SVT", dep, t)]]) for t in times} for dep in deps}
    shed_ev_out = {s: {t: float(x[var_idx[("SEV", s, t)]]) for t in times} for s in stations}

    # For scipy.optimize.linprog(method="highs"), "ineqlin.marginals" are derivatives of
    # objective value with respect to RHS b in A_ub x <= b. For a binding capacity row,
    # relaxing b by +1 kW should weakly decrease objective, so marginal is typically <= 0.
    # Scarcity adder (nonnegative, $/kW) is therefore -marginal.
    shadow_prices = {s: {t: None for t in times} for s in stations}
    cap_binding_flags = {s: {t: False for t in times} for s in stations}
    dual_trace = {s: {t: {"label": f"cap_constraint[{s},{t}]", "dual_raw": None, "mu_kw": None} for t in times} for s in stations}
    marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
    if marg is None:
        raise LPFailed({
            "status": getattr(res, "status", None),
            "message": "HiGHS solved shared-power LP but did not return inequality marginals; cannot build LP-dual scarcity prices.",
            "fun": getattr(res, "fun", None),
            "nit": getattr(res, "nit", None),
            **shape_payload,
        })
    if len(marg) < len(A_ub):
        raise LPFailed({
            "status": getattr(res, "status", None),
            "message": f"HiGHS returned insufficient marginals ({len(marg)}) for A_ub rows ({len(A_ub)}).",
            "fun": getattr(res, "fun", None),
            "nit": getattr(res, "nit", None),
            **shape_payload,
        })
    for s, t, ridx in cap_rows:
        m = float(marg[ridx])
        mu = max(0.0, -m)
        shadow_prices[s][t] = mu
        dual_trace[s][t] = {
            "label": f"cap_constraint[{s},{t}]",
            "dual_raw": m,
            "mu_kw": mu,
        }

    objective_components = {s: {t: {"energy_cost_term": 0.0, "ev_shed_penalty": 0.0, "vt_shed_penalty": 0.0} for t in times} for s in stations}
    total_energy_cost = 0.0
    total_ev_shed_penalty = 0.0
    total_vt_shed_penalty = 0.0
    for s in stations:
        for t in times:
            p_vt_sum = sum(P_out.get(dep, {}).get(t, 0.0) for dep in deps if dep == s)
            p_ev_served = max(0.0, p_ev_req_kw[s][t] - shed_ev_out[s][t])
            energy_cost_term = float(prices[s][t]) * (p_ev_served + p_vt_sum) * delta_t
            ev_pen = voll_ev_per_kwh * shed_ev_out[s][t] * delta_t
            vt_pen = voll_vt_per_kwh * sum(shed_vt_out.get(dep, {}).get(t, 0.0) for dep in deps if dep == s)
            objective_components[s][t] = {
                "energy_cost_term": energy_cost_term,
                "ev_shed_penalty": ev_pen,
                "vt_shed_penalty": vt_pen,
            }
            total_energy_cost += energy_cost_term
            total_ev_shed_penalty += ev_pen
            total_vt_shed_penalty += vt_pen

    residuals = {"INV1": 0.0, "INV2": 0.0, "INV3": 0.0, "INV4": 0.0}
    for dep in deps:
        eta = float(storage_params[dep]["eta_ch"])
        for idx, t in enumerate(times):
            t_next = times_ext[idx + 1]
            lhs = B_out[dep][t_next]
            rhs = B_out[dep][t] + eta * P_out[dep][t] * delta_t - (float(e_dep.get(dep, {}).get(t, 0.0)) - shed_vt_out[dep][t])
            residuals["INV1"] = max(residuals["INV1"], abs(lhs - rhs))
        for tau in times_ext:
            residuals["INV2"] = max(
                residuals["INV2"],
                max(storage_params[dep]["B_min"] - B_out[dep][tau], B_out[dep][tau] - storage_params[dep]["B_max"], 0.0),
            )
    for s in stations:
        for t in times:
            p_vt_sum = sum(P_out.get(dep, {}).get(t, 0.0) for dep in deps if dep == s)
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, p_vt_sum + p_ev_req_kw[s][t] - shed_ev_out[s][t] - _effective_station_power_cap(data, s, t)),
            )
            residuals["INV4"] = max(residuals["INV4"], max(0.0, shed_ev_out[s][t]))

    lp_diag = {
        "solver": "highs",
        "fallback_used": False,
        "dual_available": True,
        "objective_components": objective_components,
        "objective_totals": {
            "energy_cost_term": total_energy_cost,
            "ev_shed_penalty": total_ev_shed_penalty,
            "vt_shed_penalty": total_vt_shed_penalty,
            "total_objective": total_energy_cost + total_ev_shed_penalty + total_vt_shed_penalty,
        },
        "dual_trace": dual_trace,
    }
    binding_cap_total_count = 0
    binding_cap_with_positive_dual_count = 0
    cap_binding_flags = {s: {t: False for t in times} for s in stations}
    for s in stations:
        for t in times:
            p_vt_sum = sum(P_out.get(dep, {}).get(t, 0.0) for dep in deps if dep == s)
            lhs = p_vt_sum + p_ev_req_kw[s][t] - shed_ev_out[s][t]
            cap = _effective_station_power_cap(data, s, t)
            bind = abs(lhs - cap) <= 1.0e-6
            cap_binding_flags[s][t] = bind
            if bind:
                binding_cap_total_count += 1
                mu = shadow_prices.get(s, {}).get(t)
                if mu is not None and float(mu) > 1.0e-9:
                    binding_cap_with_positive_dual_count += 1
    lp_diag["cap_binding_flags"] = cap_binding_flags
    lp_diag["binding_cap_total_count"] = binding_cap_total_count
    lp_diag["binding_cap_with_positive_dual_count"] = binding_cap_with_positive_dual_count
    zero_dual_binding_rows: list[dict[str, Any]] = []
    for s in stations:
        for t in times:
            if cap_binding_flags[s][t] and float(shadow_prices[s][t] or 0.0) <= 1.0e-9:
                zero_dual_binding_rows.append({
                    "station": s,
                    "time": int(t),
                    "dual_raw": dual_trace[s][t]["dual_raw"],
                    "mu_kw": float(shadow_prices[s][t] or 0.0),
                })
    if zero_dual_binding_rows:
        raise LPFailed({
            "status": getattr(res, "status", None),
            "message": "Binding shared-power cap rows had zero/nonpositive LP dual; cannot validate scarcity pricing channel.",
            "fun": getattr(res, "fun", None),
            "nit": getattr(res, "nit", None),
            "binding_rows_without_positive_dual": zero_dual_binding_rows,
            **shape_payload,
        })
    lp_diag["shared_power_price_signal_check"] = {
        "solver_used": "highs",
        "lp_dual_available": True,
        "shared_power_fallback_used": False,
        "binding_cap_total_count": binding_cap_total_count,
        "binding_cap_with_positive_dual_count": binding_cap_with_positive_dual_count,
        "note": "LP-dual-only mode: scarcity adders are derived strictly from HiGHS cap-row marginals.",
    }

    return B_out, P_out, shed_ev_out, shed_vt_out, shadow_prices, residuals, lp_diag


def _solve_shared_power_core(

    data: Dict[str, Any],
    times: list[int],
    e_dep: Dict[str, Dict[int, float]],
    ev_energy: Dict[str, Dict[int, float]],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
    Dict[str, Any],
]:
    global LAST_SHARED_SOLVER_USED, LAST_SHARED_POWER_SOLVER_USED
    solver_pref = str(data.get("config", {}).get("shared_power_solver", "highs")).lower()
    if solver_pref != "highs":
        raise ValueError("shared_power_solver must be 'highs' (LP-only mode)")
    if not HAS_SCIPY:
        raise LPFailed({
            "status": None,
            "message": f"SciPy is not available for requested HiGHS solver: {SCIPY_IMPORT_ERROR}",
            "fun": None,
            "nit": None,
            "max_ub_violation": None,
            "max_eq_violation": None,
            "n_vars": None,
            "A_ub_shape": None,
            "A_eq_shape": None,
            "n_bounds": None,
        })
    return solve_shared_power_inventory_highs(data, times, e_dep, ev_energy)

def solve_shared_power_lp(
    data: Dict[str, Any],
    itineraries: list[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: list[int],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
]:
    d_vt_route = aggregate_evtol_demand(flows, itineraries, times)
    e_dep = compute_evtol_energy_demand(d_vt_route, itineraries, times)
    ev_energy = aggregate_ev_energy_demand(itineraries, flows, times)
    return _solve_shared_power_core(data, times, e_dep, ev_energy)


def solve_shared_power_inventory_lp(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]],
    ev_energy: Dict[str, Dict[int, float]],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
]:
    """Solve shared station-power and storage inventory on hybrid stations only.

    Inputs are zero-filled onto the hybrid set so that stations with zero current
    demand remain explicitly modeled (important for terminal SOC accounting).
    """
    times = data["sets"]["time"]
    stations = set(_hybrid_station_list(data))
    storage_params = data.get("parameters", {}).get("vertiport_storage", {})
    if not set(str(k) for k in storage_params.keys()).issuperset(stations):
        missing = sorted(stations - set(str(k) for k in storage_params.keys()))
        raise ValueError(f"Missing required key path: parameters.vertiport_storage for hybrid stations {missing}")
    e_dep_full = {s: {t: float(e_dep.get(s, {}).get(t, 0.0)) for t in times} for s in stations}
    ev_energy_full = {s: {t: float(ev_energy.get(s, {}).get(t, 0.0)) for t in times} for s in stations}
    diagnostics_ref = data.setdefault("diagnostics_runtime", {})
    B_out, P_out, shed_ev_out, shed_vt_out, shadow_prices, residuals, lp_diag = _solve_shared_power_core(data, times, e_dep_full, ev_energy_full)
    diagnostics_ref["lp_ok"] = True
    diagnostics_ref["lp_failure"] = None

    key_mismatch = 0.0
    for dep in B_out.keys():
        if dep not in stations:
            key_mismatch = 1.0
            raise ValueError(f"Shared-power output key mismatch: dep={dep} not in stations")
    for dep in P_out.keys():
        if dep not in stations:
            key_mismatch = 1.0
            raise ValueError(f"Shared-power output key mismatch: dep={dep} not in stations")
    residuals["KEY_MISMATCH"] = key_mismatch
    return B_out, P_out, shed_ev_out, shed_vt_out, shadow_prices, residuals, lp_diag


def solve_charging(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]] | None = None,
    d_dep: Dict[str, Dict[int, float]] | None = None,
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[str, Dict[int, float]]],
    Dict[str, Dict[str, Dict[int, int]]],
    Dict[str, float],
    Dict[str, Dict[int, float]] | None,
    Dict[str, Dict[int, float]] | None,
    Dict[str, float] | None,
    Dict[str, Dict[int, float]] | None,
]:
    global LAST_SOLVER_USED

    if not HAS_GUROBI:
        return _solve_charging_pulp(data, e_dep, d_dep)

    LAST_SOLVER_USED = "gurobi"
    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = _hybrid_station_list(data)
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    # d_dep kept for API compatibility/future vehicle turnover constraints; LP here uses e_dep only.
    include_vt = (e_dep is not None) and any(
        val > 0.0 for dep in (e_dep or {}) for val in e_dep[dep].values()
    )
    if include_vt and storage_params is None:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    model = gp.Model("charging")
    model.setParam("OutputFlag", 0)

    E = {}
    p_ch = {}
    y = {}
    for m in vehicles:
        for t in times:
            E[m, t] = model.addVar(
                lb=charging_params[m]["E_min"],
                ub=charging_params[m]["E_max"],
                name=f"E_{m}_{t}",
            )
        for s in stations:
            for t in times:
                p_ch[m, s, t] = model.addVar(lb=0.0, ub=charging_params[m]["P_max"], name=f"p_{m}_{s}_{t}")
                y[m, s, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{m}_{s}_{t}")

    B_vt = {}
    P_vt = {}
    if include_vt:
        for dep in e_dep:
            if dep not in storage_params:
                raise ValueError(f"Missing required key path: parameters.vertiport_storage.{dep}")
            for t in times:
                B_vt[dep, t] = model.addVar(
                    lb=storage_params[dep]["B_min"],
                    ub=storage_params[dep]["B_max"],
                    name=f"B_vt_{dep}_{t}",
                )
                P_vt[dep, t] = model.addVar(
                    lb=0.0,
                    ub=_vt_charge_power_upper_bound(data, dep, t, e_dep),
                    name=f"P_vt_{dep}_{t}",
                )

    model.setObjective(
        gp.quicksum(prices[s][t] * p_ch[m, s, t] * delta_t for m in vehicles for s in stations for t in times)
        + gp.quicksum(prices[dep][t] * P_vt[dep, t] * delta_t for dep in e_dep or {} for t in times),
        GRB.MINIMIZE,
    )

    for m in vehicles:
        model.addConstr(E[m, times[0]] == charging_params[m]["E_init"], name=f"E_init_{m}")
        eta = charging_params[m]["eta_ch"]
        E_res = charging_params[m]["E_res"]
        for idx, t in enumerate(times):
            model.addConstr(E[m, t] >= E_res, name=f"E_res_{m}_{t}")
            if idx < len(times) - 1:
                model.addConstr(
                    E[m, times[idx + 1]]
                    == E[m, t] - e_fly_or_drive[m][t] + gp.quicksum(eta * p_ch[m, s, t] * delta_t for s in stations),
                    name=f"E_dyn_{m}_{t}",
                )
            model.addConstr(gp.quicksum(y[m, s, t] for s in stations) <= 1, name=f"y_one_{m}_{t}")
            for s in stations:
                model.addConstr(p_ch[m, s, t] <= charging_params[m]["P_max"] * y[m, s, t], name=f"p_link_{m}_{s}_{t}")
                model.addConstr(y[m, s, t] <= avail[m][s][t], name=f"y_avail_{m}_{s}_{t}")

    for s in stations:
        cap = station_params[s]["cap_stall"]
        for t in times:
            vt_power = gp.quicksum(P_vt[dep, t] for dep in e_dep or {} if dep == s) if include_vt else 0.0
            model.addConstr(
                gp.quicksum(p_ch[m, s, t] for m in vehicles) + vt_power <= _effective_station_power_cap(data, s, t),
                name=f"P_site_{s}_{t}",
            )
            model.addConstr(gp.quicksum(y[m, s, t] for m in vehicles) <= cap, name=f"cap_{s}_{t}")

    if include_vt:
        for dep in e_dep:
            model.addConstr(B_vt[dep, times[0]] == storage_params[dep]["B_init"], name=f"B_init_{dep}")
            eta = storage_params[dep]["eta_ch"]
            for idx, t in enumerate(times):
                if idx < len(times) - 1:
                    model.addConstr(
                        B_vt[dep, times[idx + 1]]
                        == B_vt[dep, t] + eta * P_vt[dep, t] * delta_t - e_dep[dep][t],
                        name=f"B_bal_{dep}_{t}",
                    )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        if model.Status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write("charging.ilp")
        raise ValueError(f"Charging optimization did not solve to optimality: status={model.Status}")

    E_out: Dict[str, Dict[int, float]] = {m: {t: float(E[m, t].X) for t in times} for m in vehicles}
    p_out: Dict[str, Dict[str, Dict[int, float]]] = {
        m: {s: {t: float(p_ch[m, s, t].X) for t in times} for s in stations} for m in vehicles
    }
    y_out: Dict[str, Dict[str, Dict[int, int]]] = {
        m: {s: {t: int(round(y[m, s, t].X)) for t in times} for s in stations} for m in vehicles
    }

    B_out = None
    P_out = None
    inv_residuals = None
    shadow_prices = None
    if include_vt:
        B_out = {dep: {t: float(B_vt[dep, t].X) for t in times} for dep in e_dep}
        P_out = {dep: {t: float(P_vt[dep, t].X) for t in times} for dep in e_dep}
        inv_residuals = compute_inventory_residuals(data, e_dep, B_out, P_out)

    residuals = compute_charging_residuals(data, E_out, p_out, y_out, P_out)

    return E_out, p_out, y_out, residuals, B_out, P_out, inv_residuals, shadow_prices


def _solve_charging_pulp(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]] | None = None,
    d_dep: Dict[str, Dict[int, float]] | None = None,
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[str, Dict[int, float]]],
    Dict[str, Dict[str, Dict[int, int]]],
    Dict[str, float],
    Dict[str, Dict[int, float]] | None,
    Dict[str, Dict[int, float]] | None,
    Dict[str, float] | None,
    Dict[str, Dict[int, float]] | None,
]:
    global LAST_SOLVER_USED
    LAST_SOLVER_USED = "pulp"

    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = _hybrid_station_list(data)
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    # d_dep kept for API compatibility/future vehicle turnover constraints; LP here uses e_dep only.
    include_vt = (e_dep is not None) and any(
        val > 0.0 for dep in (e_dep or {}) for val in e_dep[dep].values()
    )
    if include_vt and storage_params is None:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    E_out = {m: {t: 0.0 for t in times} for m in vehicles}
    p_out = {m: {s: {t: 0.0 for t in times} for s in stations} for m in vehicles}
    y_out = {m: {s: {t: 0 for t in times} for s in stations} for m in vehicles}

    B_out = None
    P_out = None
    inv_residuals = None
    shadow_prices = None

    P_vt_remaining = {s: {t: _effective_station_power_cap(data, s, t) for t in times} for s in stations}
    if include_vt:
        B_out = {dep: {t: 0.0 for t in times} for dep in e_dep}
        P_out = {dep: {t: 0.0 for t in times} for dep in e_dep}
        for dep in e_dep:
            B_out[dep][times[0]] = storage_params[dep]["B_init"]
            eta = storage_params[dep]["eta_ch"]
            for idx, t in enumerate(times):
                desired = max(0.0, e_dep[dep][t] / (eta * delta_t))
                P_out[dep][t] = min(P_vt_remaining[dep][t], desired)
                P_vt_remaining[dep][t] -= P_out[dep][t]
                next_B = B_out[dep][t] + eta * P_out[dep][t] * delta_t - e_dep[dep][t]
                if idx < len(times) - 1:
                    B_out[dep][times[idx + 1]] = max(storage_params[dep]["B_min"], min(storage_params[dep]["B_max"], next_B))
        inv_residuals = compute_inventory_residuals(data, e_dep, B_out, P_out)

    stall_remaining = {s: {t: station_params[s]["cap_stall"] for t in times} for s in stations}
    for m in vehicles:
        E_out[m][times[0]] = charging_params[m]["E_init"]
        eta = charging_params[m]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                projected = E_out[m][t] - e_fly_or_drive[m][t]
                needed = max(0.0, charging_params[m]["E_res"] - projected)
                required_power = needed / (eta * delta_t) if eta > 0 else 0.0
                if required_power > 0.0:
                    candidate_stations = [
                        s for s in stations if avail[m][s][t] == 1 and stall_remaining[s][t] > 0
                    ]
                    if candidate_stations:
                        s_choice = max(candidate_stations, key=lambda s: P_vt_remaining[s][t])
                        charge = min(required_power, charging_params[m]["P_max"], P_vt_remaining[s_choice][t])
                        if charge > 0.0:
                            p_out[m][s_choice][t] = charge
                            y_out[m][s_choice][t] = 1
                            P_vt_remaining[s_choice][t] -= charge
                            stall_remaining[s_choice][t] -= 1
                            projected += eta * charge * delta_t
                E_out[m][times[idx + 1]] = projected

    residuals = compute_charging_residuals(data, E_out, p_out, y_out, P_out)

    return E_out, p_out, y_out, residuals, B_out, P_out, inv_residuals, shadow_prices


def compute_charging_residuals(
    data: Dict[str, Any],
    E: Dict[str, Dict[int, float]],
    p_ch: Dict[str, Dict[str, Dict[int, float]]],
    y: Dict[str, Dict[str, Dict[int, int]]],
    P_vt: Dict[str, Dict[int, float]] | None = None,
) -> Dict[str, float]:
    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = _hybrid_station_list(data)
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]

    residuals = {
        "C12": 0.0,
        "C13": 0.0,
        "C14": 0.0,
        "C15": 0.0,
        "C16": 0.0,
        "C17": 0.0,
        "C18": 0.0,
    }

    for m in vehicles:
        E_min = charging_params[m]["E_min"]
        E_max = charging_params[m]["E_max"]
        E_res = charging_params[m]["E_res"]
        eta = charging_params[m]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = E[m][times[idx + 1]]
                rhs = E[m][t] - e_fly_or_drive[m][t]
                charge_term = sum(eta * p_ch[m][s][t] * delta_t for s in stations)
                rhs += charge_term
                residuals["C12"] = max(residuals["C12"], abs(lhs - rhs))
            residuals["C13"] = max(residuals["C13"], max(E_min - E[m][t], E[m][t] - E_max, 0.0))
            residuals["C17"] = max(residuals["C17"], max(E_res - E[m][t], 0.0))
            residuals["C18"] = max(
                residuals["C18"],
                max(0.0, sum(y[m][s][t] for s in stations) - 1.0),
            )
            for s in stations:
                P_max = charging_params[m]["P_max"]
                residuals["C14"] = max(
                    residuals["C14"],
                    max(0.0, p_ch[m][s][t] - P_max * y[m][s][t], y[m][s][t] - avail[m][s][t]),
                )
    for s in stations:
        cap = station_params[s]["cap_stall"]
        for t in times:
            total_power = sum(p_ch[m][s][t] for m in vehicles)
            if P_vt and s in P_vt:
                total_power += P_vt[s][t]
            total_y = sum(y[m][s][t] for m in vehicles)
            P_site = _effective_station_power_cap(data, s, t)
            residuals["C15"] = max(residuals["C15"], max(0.0, total_power - P_site))
            residuals["C16"] = max(residuals["C16"], max(0.0, total_y - cap))
    return residuals


def compute_inventory_residuals(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]],
    B_vt: Dict[str, Dict[int, float]],
    P_vt: Dict[str, Dict[int, float]],
) -> Dict[str, float]:
    times = data["sets"]["time"]
    delta_t = data["meta"]["delta_t"]
    station_params = data["parameters"]["stations"]
    storage_params = data["parameters"]["vertiport_storage"]

    residuals = {"INV1": 0.0, "INV2": 0.0, "INV3": 0.0}
    for dep in e_dep:
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = B_vt[dep][times[idx + 1]]
                rhs = B_vt[dep][t] + eta * P_vt[dep][t] * delta_t - e_dep[dep][t]
                residuals["INV1"] = max(residuals["INV1"], abs(lhs - rhs))
            residuals["INV2"] = max(
                residuals["INV2"],
                max(storage_params[dep]["B_min"] - B_vt[dep][t], B_vt[dep][t] - storage_params[dep]["B_max"], 0.0),
            )
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, P_vt[dep][t] - station_params[dep]["P_site"][t]),
            )
    return residuals
