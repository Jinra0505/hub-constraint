from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback handled by caller
    HAS_SCIPY = False


class DistributionGridError(RuntimeError):
    pass


def _time_value(raw: Any, t: int, default: float = 0.0) -> float:
    if isinstance(raw, dict):
        if t in raw:
            return float(raw[t])
        key = str(t)
        if key in raw:
            return float(raw[key])
        return float(default)
    if raw is None:
        return float(default)
    return float(raw)


def _build_topology(grid: Dict[str, Any]) -> Dict[str, Any]:
    buses = [str(x) for x in grid.get("grid_buses", [])]
    branches = [str(x) for x in grid.get("grid_branches", [])]
    root = str(grid.get("grid_root_bus"))
    branch_data = {str(k): v for k, v in grid.get("grid_branch", {}).items()}
    if not buses or not branches or root not in buses:
        raise DistributionGridError("Invalid distribution grid sets/root bus")

    children: Dict[str, List[str]] = defaultdict(list)
    parent: Dict[str, str] = {}
    outgoing_branch: Dict[Tuple[str, str], str] = {}
    for br in branches:
        info = branch_data.get(br, {})
        frm = str(info.get("from_bus"))
        to = str(info.get("to_bus"))
        if frm not in buses or to not in buses:
            raise DistributionGridError(f"Branch {br} references unknown buses")
        children[frm].append(to)
        if to in parent:
            raise DistributionGridError(f"Bus {to} has multiple parents; feeder must be radial")
        parent[to] = frm
        outgoing_branch[(frm, to)] = br

    visited = []
    q = deque([root])
    seen = {root}
    while q:
        node = q.popleft()
        visited.append(node)
        for child in children.get(node, []):
            if child in seen:
                raise DistributionGridError("Distribution feeder contains a cycle")
            seen.add(child)
            q.append(child)
    if set(visited) != set(buses):
        missing = sorted(set(buses) - set(visited))
        raise DistributionGridError(f"Grid is disconnected from root bus {root}: {missing}")

    downstream = {bus: [] for bus in buses}
    for bus in reversed(visited):
        desc = [bus]
        for child in children.get(bus, []):
            desc.extend(downstream[child])
        downstream[bus] = desc

    return {
        "buses": buses,
        "branches": branches,
        "root": root,
        "children": children,
        "parent": parent,
        "branch_data": branch_data,
        "outgoing_branch": outgoing_branch,
        "topo_order": visited,
        "downstream": downstream,
    }


def validate_distribution_grid(data: Dict[str, Any]) -> None:
    config = data.get("config", {})
    use_grid = bool(config.get("use_distribution_grid", False))
    params = data.get("parameters", {})
    grid = params.get("distribution_grid")
    if not use_grid and grid is None:
        return
    if grid is None:
        if use_grid:
            raise ValueError("config.use_distribution_grid=true requires parameters.distribution_grid")
        return
    topo = _build_topology(grid)
    ev_stations = [str(x) for x in data.get("sets", {}).get("ev_stations", [])]
    station_to_bus = {str(k): str(v) for k, v in grid.get("station_to_bus", {}).items()}
    missing = sorted(set(ev_stations) - set(station_to_bus))
    if missing:
        raise ValueError(f"distribution_grid.station_to_bus missing ev stations: {missing}")
    bad = sorted({bus for bus in station_to_bus.values() if bus not in topo["buses"]})
    if bad:
        raise ValueError(f"distribution_grid.station_to_bus references unknown buses: {bad}")


def solve_distribution_grid(
    data: Dict[str, Any],
    station_power_request_kw: Dict[str, Dict[int, float]],
    times: List[int],
) -> Dict[str, Any]:
    if not HAS_SCIPY:
        raise DistributionGridError("SciPy is required for the linearized distribution-grid solver")

    grid = data.get("parameters", {}).get("distribution_grid")
    if grid is None:
        raise DistributionGridError("Missing parameters.distribution_grid")
    topo = _build_topology(grid)
    buses = topo["buses"]
    branches = topo["branches"]
    root = topo["root"]
    branch_data = topo["branch_data"]
    children = topo["children"]
    outgoing_branch = topo["outgoing_branch"]
    station_to_bus = {str(k): str(v) for k, v in grid.get("station_to_bus", {}).items()}
    bus_limits = {str(k): v for k, v in grid.get("grid_bus_limits", {}).items()}
    base_load_p = {str(k): v for k, v in grid.get("grid_base_load_p", {}).items()}
    sub_cap = grid.get("grid_substation_cap", {})
    grid_price_base = grid.get("grid_price_base", {})
    root_voltage = float(grid.get("grid_root_voltage", 1.0))

    bus_to_stations: Dict[str, List[str]] = defaultdict(list)
    for station, bus in station_to_bus.items():
        bus_to_stations[bus].append(station)

    result = {
        "distribution_grid_enabled": True,
        "bus_voltage": {bus: {t: root_voltage for t in times} for bus in buses},
        "branch_flow_p": {br: {t: 0.0 for t in times} for br in branches},
        "branch_loading_ratio": {br: {t: 0.0 for t in times} for br in branches},
        "substation_loading": {t: 0.0 for t in times},
        "station_grid_available_power": {s: {t: float(station_power_request_kw.get(s, {}).get(t, 0.0)) for t in times} for s in station_to_bus},
        "station_grid_shadow_price": {s: {t: 0.0 for t in times} for s in station_to_bus},
        "grid_binding_flags": {
            "branch": {br: {t: False for t in times} for br in branches},
            "voltage": {bus: {t: False for t in times} for bus in buses},
            "substation": {t: False for t in times},
            "station": {s: {t: False for t in times} for s in station_to_bus},
        },
        "grid_binding_count": 0,
        "voltage_violation_count": 0,
        "branch_overload_count": 0,
        "substation_binding_count": 0,
        "station_to_bus": station_to_bus,
        "solver": "highs_two_stage",
    }

    for t in times:
        station_list = list(station_to_bus.keys())

        def _solve_stage(stage: str, served_target: float | None = None):
            var_idx: Dict[Tuple[str, str], int] = {}
            c: List[float] = []
            bounds: List[Tuple[float | None, float | None]] = []

            def add_var(kind: str, key: str, lb: float | None, ub: float | None, obj: float = 0.0) -> None:
                var_idx[(kind, key)] = len(c)
                c.append(float(obj))
                bounds.append((lb, ub))

            for s in station_list:
                req = max(0.0, float(station_power_request_kw.get(s, {}).get(t, 0.0)))
                add_var("PSTAT", s, 0.0, req, -1.0 if stage == "served" else 0.0)
            for br in branches:
                lim = float(_time_value(branch_data[br].get("p_max"), t, 0.0))
                add_var("PF", br, -lim, lim, 0.0)
            for bus in buses:
                lim = bus_limits.get(bus, {})
                add_var("V", bus, float(lim.get("v_min", 0.95)), float(lim.get("v_max", 1.05)), 0.0)
            add_var("PGRID", root, 0.0, float(_time_value(sub_cap, t, 1.0e12)), 1.0e-6 if stage == "served" else 0.0)
            if stage == "fair":
                add_var("Z", "fair", 0.0, 1.0, 1.0)

            n = len(c)
            A_eq: List[List[float]] = []
            b_eq: List[float] = []
            A_ub: List[List[float]] = []
            b_ub: List[float] = []
            station_rows: Dict[str, int] = {}
            branch_lim_rows: Dict[str, Tuple[int, int]] = {}
            voltage_rows: Dict[str, Tuple[int, int]] = {}
            substation_row: int | None = None

            row = [0.0] * n
            row[var_idx[("V", root)]] = 1.0
            A_eq.append(row)
            b_eq.append(root_voltage)

            for bus in buses:
                row = [0.0] * n
                if bus == root:
                    row[var_idx[("PGRID", root)]] = 1.0
                else:
                    parent = topo["parent"][bus]
                    br_in = outgoing_branch[(parent, bus)]
                    row[var_idx[("PF", br_in)]] += 1.0
                for child in children.get(bus, []):
                    br_out = outgoing_branch[(bus, child)]
                    row[var_idx[("PF", br_out)]] -= 1.0
                for s in bus_to_stations.get(bus, []):
                    row[var_idx[("PSTAT", s)]] -= 1.0
                A_eq.append(row)
                fixed_load = float(_time_value(base_load_p.get(bus, 0.0), t, 0.0))
                b_eq.append(fixed_load)

            for frm, childs in children.items():
                for to in childs:
                    br = outgoing_branch[(frm, to)]
                    r = float(branch_data[br].get("r", 0.0))
                    row = [0.0] * n
                    row[var_idx[("V", to)]] = 1.0
                    row[var_idx[("V", frm)]] = -1.0
                    row[var_idx[("PF", br)]] = 2.0 * r
                    A_eq.append(row)
                    b_eq.append(0.0)

            if stage == "fair" and served_target is not None:
                row = [0.0] * n
                for s in station_list:
                    row[var_idx[("PSTAT", s)]] = 1.0
                A_eq.append(row)
                b_eq.append(float(served_target))

            for s in station_list:
                row = [0.0] * n
                row[var_idx[("PSTAT", s)]] = 1.0
                A_ub.append(row)
                b_ub.append(float(station_power_request_kw.get(s, {}).get(t, 0.0)))
                station_rows[s] = len(A_ub) - 1

            for br in branches:
                lim = float(_time_value(branch_data[br].get("p_max"), t, 0.0))
                row_pos = [0.0] * n
                row_pos[var_idx[("PF", br)]] = 1.0
                A_ub.append(row_pos)
                b_ub.append(lim)
                pos_idx = len(A_ub) - 1
                row_neg = [0.0] * n
                row_neg[var_idx[("PF", br)]] = -1.0
                A_ub.append(row_neg)
                b_ub.append(lim)
                neg_idx = len(A_ub) - 1
                branch_lim_rows[br] = (pos_idx, neg_idx)

            for bus in buses:
                lim = bus_limits.get(bus, {})
                vmin = float(lim.get("v_min", 0.95))
                vmax = float(lim.get("v_max", 1.05))
                row_hi = [0.0] * n
                row_hi[var_idx[("V", bus)]] = 1.0
                A_ub.append(row_hi)
                b_ub.append(vmax)
                hi_idx = len(A_ub) - 1
                row_lo = [0.0] * n
                row_lo[var_idx[("V", bus)]] = -1.0
                A_ub.append(row_lo)
                b_ub.append(-vmin)
                lo_idx = len(A_ub) - 1
                voltage_rows[bus] = (hi_idx, lo_idx)

            row = [0.0] * n
            row[var_idx[("PGRID", root)]] = 1.0
            A_ub.append(row)
            b_ub.append(float(_time_value(sub_cap, t, 1.0e12)))
            substation_row = len(A_ub) - 1

            if stage == "fair":
                z_idx = var_idx[("Z", "fair")]
                for s in station_list:
                    req = float(station_power_request_kw.get(s, {}).get(t, 0.0))
                    if req <= 1.0e-9:
                        continue
                    row = [0.0] * n
                    row[var_idx[("PSTAT", s)]] = -1.0
                    row[z_idx] = -req
                    A_ub.append(row)
                    b_ub.append(-req)

            res = linprog(
                c=np.array(c, dtype=float),
                A_ub=np.array(A_ub, dtype=float),
                b_ub=np.array(b_ub, dtype=float),
                A_eq=np.array(A_eq, dtype=float),
                b_eq=np.array(b_eq, dtype=float),
                bounds=bounds,
                method="highs",
            )
            if not res.success:
                raise DistributionGridError(f"Distribution-grid LP failed at t={t}, stage={stage}: {res.message}")
            return res, var_idx, station_rows, branch_lim_rows, voltage_rows, substation_row

        res1, idx1, station_rows1, branch_rows1, voltage_rows1, sub_row1 = _solve_stage("served")
        served_target = sum(float(res1.x[idx1[("PSTAT", s)]]) for s in station_list)
        res, var_idx, station_rows, branch_lim_rows, voltage_rows, substation_row = _solve_stage("fair", served_target)

        x = res.x
        marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
        slack = getattr(getattr(res, "ineqlin", None), "residual", None)

        result["substation_loading"][t] = float(x[var_idx[("PGRID", root)]])
        if slack is not None and substation_row is not None and float(slack[substation_row]) <= 1.0e-7:
            result["grid_binding_flags"]["substation"][t] = True
            result["substation_binding_count"] += 1
            result["grid_binding_count"] += 1

        for br in branches:
            flow = float(x[var_idx[("PF", br)]])
            lim = max(1.0e-9, float(_time_value(branch_data[br].get("p_max"), t, 0.0)))
            ratio = abs(flow) / lim
            result["branch_flow_p"][br][t] = flow
            result["branch_loading_ratio"][br][t] = ratio
            if ratio > 1.0 + 1.0e-6:
                result["branch_overload_count"] += 1
            if slack is not None:
                pos_idx, neg_idx = branch_lim_rows[br]
                if float(slack[pos_idx]) <= 1.0e-7 or float(slack[neg_idx]) <= 1.0e-7:
                    result["grid_binding_flags"]["branch"][br][t] = True
                    result["grid_binding_count"] += 1

        for bus in buses:
            voltage = float(x[var_idx[("V", bus)]])
            result["bus_voltage"][bus][t] = voltage
            lim = bus_limits.get(bus, {})
            vmin = float(lim.get("v_min", 0.95))
            vmax = float(lim.get("v_max", 1.05))
            if voltage < vmin - 1.0e-6 or voltage > vmax + 1.0e-6:
                result["voltage_violation_count"] += 1
            if slack is not None:
                hi_idx, lo_idx = voltage_rows[bus]
                if float(slack[hi_idx]) <= 1.0e-7 or float(slack[lo_idx]) <= 1.0e-7:
                    result["grid_binding_flags"]["voltage"][bus][t] = True
                    result["grid_binding_count"] += 1

        for s in station_list:
            served = float(x[var_idx[("PSTAT", s)]])
            result["station_grid_available_power"][s][t] = served
            req = float(station_power_request_kw.get(s, {}).get(t, 0.0))
            curtailed = req - served
            if curtailed > 1.0e-6:
                result["grid_binding_flags"]["station"][s][t] = True
            dual = 0.0
            if marg is not None:
                m = float(marg[station_rows[s]])
                dual = max(0.0, -m)
            if dual <= 1.0e-12 and curtailed > 1.0e-9:
                scarcity_base = max(0.0, float(_time_value(grid_price_base, t, 0.0)))
                dual = scarcity_base * min(1.0, curtailed / max(1.0e-9, req))
            result["station_grid_shadow_price"][s][t] = dual

    return result
