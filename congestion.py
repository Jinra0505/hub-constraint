import math
from typing import Any, Dict, List, Tuple

from .assignment import aggregate_vt_departure_flow_by_class


def _safe_den(x, eps: float = 1e-6) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return eps
    return v if v > 0 else eps


def compute_road_times(
    arc_flows: Dict[str, Dict[int, float]],
    arc_params: Dict[str, Dict[str, float]],
    g_values: Dict[int, float],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    tau = {arc: {} for arc in arc_params}
    for arc, params in arc_params.items():
        tau0 = params["tau0"]
        cap = _safe_den(params.get("cap", 0))
        alpha = params["alpha"]
        beta = params["beta"]
        arc_type = params["type"]
        theta = params.get("theta", 1.0)
        use_bpr = params.get("use_bpr", False)
        for t in times:
            x = arc_flows.get(arc, {}).get(t, 0.0)
            if arc_type == "G":
                tau[arc][t] = tau0 * (1.0 + alpha * (x / cap) ** beta)
            elif arc_type == "CBD":
                tau[arc][t] = tau0 * g_values.get(t, 1.0) * theta
            else:
                if use_bpr:
                    tau[arc][t] = tau0 * (1.0 + alpha * (x / cap) ** beta)
                else:
                    tau[arc][t] = tau0
    return tau


def compute_station_waits(
    utilization: Dict[str, Dict[int, float]],
    station_params: Dict[str, Dict[str, float]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    """EV station waiting-time proxy.

    ``utilization`` should be EV-related vehicle charging demand proxy per period
    (not passenger demand). This is an empirical BPR-style proxy rather than a
    rigorous queueing model.
    """
    waits = {s: {} for s in station_params}
    for station, params in station_params.items():
        cap = _safe_den(params.get("cap_stall", 0))
        w0 = params["w0"]
        for t in times:
            u = utilization.get(station, {}).get(t, 0.0)
            waits[station][t] = w0 * (1.0 + 0.15 * (u / cap) ** 4)
    return waits


def compute_vt_departure_waits(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, float]]]]:
    """Compute class-specific eVTOL departure waits.

    Units:
    - ``flows`` are passenger flows (pax/period).
    - Internal queue loads and capacities are handled in departures/period.
    - Wait outputs are time in the same unit as itinerary travel time (typically hours).
    """
    hybrid = data.get("sets", {}).get("hybrid_stations")
    if not isinstance(hybrid, list) or not hybrid:
        raise ValueError("Missing required key path: sets.hybrid_stations for VT departure waits")
    stations = [str(x) for x in hybrid]
    station_params = data.get("parameters", {}).get("stations", {})
    vt_caps = data.get("parameters", {}).get("vt_departure_capacity", {})
    vt_caps_total = data.get("parameters", {}).get("vt_departure_capacity_total", {})
    vt_caps_fast = data.get("parameters", {}).get("vt_departure_capacity_fast", {})

    pax_fast = float(data.get("parameters", {}).get("vt_pax_per_departure_fast", 2.0) or 2.0)
    pax_slow = float(data.get("parameters", {}).get("vt_pax_per_departure_slow", 4.0) or 4.0)

    a_fast = float(config.get("vt_queue_a_fast", 1.0))
    a_slow = float(config.get("vt_queue_a_slow", 1.5))
    g_fast = float(config.get("vt_queue_gamma_fast", 2.0))
    g_slow = float(config.get("vt_queue_gamma_slow", 2.0))
    w0_fast = float(config.get("vt_queue_w0_fast", 0.1))
    w0_slow = float(config.get("vt_queue_w0_slow", 0.2))
    spill_ratio = float(config.get("vt_fast_overflow_to_slow_ratio", 0.5))
    enforce_fast_le_slow = bool(config.get("vt_enforce_fast_le_slow", False))

    dep_flows = aggregate_vt_departure_flow_by_class(
        itineraries,
        flows,
        times,
        output_unit="departures",
        vt_pax_per_departure_fast=pax_fast,
        vt_pax_per_departure_slow=pax_slow,
    )
    for s in stations:
        dep_flows.setdefault(s, {}).setdefault("fast", {t: 0.0 for t in times})
        dep_flows.setdefault(s, {}).setdefault("slow", {t: 0.0 for t in times})
    waits: Dict[str, Dict[str, Dict[int, float]]] = {s: {"fast": {}, "slow": {}} for s in stations}

    for s in stations:
        for t in times:
            q_fast = float(dep_flows.get(s, {}).get("fast", {}).get(t, 0.0))
            q_slow = float(dep_flows.get(s, {}).get("slow", {}).get(t, 0.0))
            cap_total = vt_caps_total.get(s, {}).get(t)
            if cap_total is None:
                cap_total = vt_caps.get(s, {}).get(t)
            if cap_total is None:
                cap_total = max(1.0, 0.3 * float(station_params.get(s, {}).get("P_site", {}).get(t, 0.0)))
            cap_total = _safe_den(cap_total)
            cap_fast = vt_caps_fast.get(s, {}).get(t)
            if cap_fast is None:
                cap_fast = 0.35 * cap_total
            cap_fast = min(_safe_den(cap_fast), cap_total)
            cap_slow = _safe_den(max(cap_total - cap_fast, 1.0e-6))

            q_fast_overflow = max(0.0, q_fast - cap_fast)
            q_slow_eff = q_slow + max(0.0, spill_ratio) * q_fast_overflow

            w_fast = max(0.0, w0_fast + a_fast * (q_fast / max(cap_fast, 1.0e-6)) ** g_fast)
            w_slow = max(0.0, w0_slow + a_slow * (q_slow_eff / cap_slow) ** g_slow)
            if math.isnan(w_fast) or math.isinf(w_fast):
                w_fast = 0.0
            if math.isnan(w_slow) or math.isinf(w_slow):
                w_slow = 0.0
            # Optional policy-style monotonicity guard.
            if enforce_fast_le_slow and w_fast > w_slow:
                w_fast = max(w0_fast, w_slow)
            waits[s]["fast"][t] = float(w_fast)
            waits[s]["slow"][t] = float(w_slow)

    return waits, dep_flows
