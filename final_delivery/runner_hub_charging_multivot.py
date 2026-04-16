import argparse
import json
from typing import Any, Dict, List, Tuple

from .assignment import (
    aggregate_arc_flows,
    aggregate_ev_station_utilization,
    aggregate_vt_departure_flow_by_class,
    classify_supermode,
    compute_itinerary_costs,
    get_evtol_service_class,
    is_evtol_itinerary,
    logit_assignment,
)
from .charging_hiGHS_and_gurobi_bound_fix import compute_station_loads_from_flows, solve_shared_power_inventory_lp
from .data_loader import load_data


def _init_station_time(stations: List[str], times: List[int], value: float) -> Dict[str, Dict[int, float]]:
    return {s: {t: float(value) for t in times} for s in stations}


def _time_value(x: Any, t: int, default: float = 0.0) -> float:
    if isinstance(x, dict):
        return float(x.get(t, x.get(str(t), default)))
    if x is None:
        return float(default)
    return float(x)


def _mode_share_by_group_time(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for g in groups:
        out[g] = {}
        for t in times:
            totals = {"EV": 0.0, "eVTOL": 0.0, "EV_to_eVTOL": 0.0}
            for it in itineraries:
                totals[classify_supermode(it)] += float(flows.get(it["id"], {}).get(g, {}).get(t, 0.0))
            den = max(1e-9, sum(totals.values()))
            out[g][t] = {k: v / den for k, v in totals.items()}
    return out


def _group_time_supermode_metrics(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    costs: Dict[str, Dict[int, Dict[str, float]]],
    groups: List[str],
    times: List[int],
    vot: Dict[str, Dict[int, float]],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for g in groups:
        out[g] = {}
        for t in times:
            stats = {
                "EV": {"flow": 0.0, "avg_generalized_cost": 0.0},
                "eVTOL": {"flow": 0.0, "avg_generalized_cost": 0.0},
                "EV_to_eVTOL": {"flow": 0.0, "avg_generalized_cost": 0.0},
            }
            for it in itineraries:
                m = classify_supermode(it)
                f = float(flows.get(it["id"], {}).get(g, {}).get(t, 0.0))
                c = costs[it["id"]][t]
                gc = float(vot[g][t]) * float(c["TT"]) + float(c["Money"]) + float(c["ChargeCost"])
                stats[m]["flow"] += f
                stats[m]["avg_generalized_cost"] += f * gc
            for m in stats:
                den = max(1e-9, stats[m]["flow"])
                stats[m]["avg_generalized_cost"] = stats[m]["avg_generalized_cost"] / den
            out[g][t] = stats
    return out


def _compute_road_times(
    arc_flows: Dict[str, Dict[int, float]],
    arc_params: Dict[str, Dict[str, float]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {a: {t: 0.0 for t in times} for a in arc_flows}
    for a, tm in arc_flows.items():
        p = arc_params.get(a, {})
        tau0 = float(p.get("tau0", 1.0))
        cap = max(1.0e-6, float(p.get("cap", 1.0)))
        alpha = float(p.get("alpha", 0.15))
        beta = float(p.get("beta", 4.0))
        for t in times:
            x = max(0.0, float(tm.get(t, 0.0)))
            out[a][t] = tau0 * (1.0 + alpha * (x / cap) ** beta)
    return out


def _compute_station_waits(
    util: Dict[str, Dict[int, float]],
    station_params: Dict[str, Dict[str, Any]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for s, tm in util.items():
        cap = max(1.0, float(station_params.get(s, {}).get("cap_stall", 1.0)))
        w0 = float(station_params.get(s, {}).get("w0", 0.0))
        out[s] = {}
        for t in times:
            x = max(0.0, float(tm.get(t, 0.0)))
            out[s][t] = w0 * (1.0 + (x / cap) ** 2)
    return out


def _compute_transfer_waits(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    groups = [str(g) for g in data["sets"]["groups"]]
    cap_map = data["parameters"].get("transfer_capacity", {})
    alpha_map = data["parameters"].get("transfer_alpha", {})
    beta_map = data["parameters"].get("transfer_beta", {})
    demand = {s: {t: 0.0 for t in times} for s in data["sets"]["hybrid_stations"]}
    for it in itineraries:
        if str(it.get("mode", "")).lower().startswith("ev_to_evtol"):
            dep = str(it.get("dep_station"))
            for t in times:
                demand[dep][t] += sum(float(flows.get(it["id"], {}).get(g, {}).get(t, 0.0)) for g in groups)
    wait = {s: {t: 0.0 for t in times} for s in demand}
    for s in demand:
        cap = max(1.0e-6, float(cap_map.get(s, 1.0)))
        alpha = float(alpha_map.get(s, 0.0))
        beta = float(beta_map.get(s, 1.0))
        for t in times:
            r = max(0.0, demand[s][t]) / cap
            wait[s][t] = alpha * (r ** beta)
    return wait


def _compute_vt_departure_waits(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, float]]]]:
    stations = [str(s) for s in data["sets"]["hybrid_stations"]]
    groups = [str(g) for g in data["sets"]["groups"]]
    params = data["parameters"]
    pax_fast = max(1.0e-6, float(params.get("vt_pax_per_departure_fast", 2.0)))
    pax_slow = max(1.0e-6, float(params.get("vt_pax_per_departure_slow", 4.0)))
    lag = int(params.get("vt_turnaround_lag", 1))
    dt = float(data["meta"]["delta_t"])
    init = {s: float(params.get("vt_aircraft_init_by_station", {}).get(s, 0.0)) for s in stations}

    req = {s: {"fast": {t: 0.0 for t in times}, "slow": {t: 0.0 for t in times}} for s in stations}
    arr_by_station = {s: {t: 0.0 for t in times} for s in stations}

    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep = str(it.get("dep_station"))
        arr = str(it.get("arr_station"))
        cls = get_evtol_service_class(it)
        per = pax_fast if cls == "fast" else pax_slow
        for t in times:
            pax = sum(float(flows.get(it["id"], {}).get(g, {}).get(t, 0.0)) for g in groups)
            dep_req = pax / per
            req[dep][cls][t] += dep_req
            flt = _time_value(it.get("flight_time", {}), t, 0.0)
            arr_t = t + int(round(flt / max(1.0e-9, dt))) + lag
            if arr_t in arr_by_station.get(arr, {}):
                arr_by_station[arr][arr_t] += dep_req

    waits = {s: {"fast": {t: 0.0 for t in times}, "slow": {t: 0.0 for t in times}} for s in stations}
    diag = {s: {"req_dep": {t: 0.0 for t in times}, "cap_dep": {t: 0.0 for t in times}, "served_ratio": {t: 1.0 for t in times}} for s in stations}
    reposition_enabled = bool(data.get("config", {}).get("vt_reposition_enabled", True))
    reposition_lag = int(data.get("config", {}).get("vt_reposition_lag", 1))
    reposition_reserve = float(data.get("config", {}).get("vt_reposition_reserve", 0.6))
    reposition_max_send = float(data.get("config", {}).get("vt_reposition_max_send_per_period", 1.4))
    vt_wait_scale = float(data.get("config", {}).get("vt_departure_wait_scale", 0.04))
    vt_wait_util_cap_raw = data.get("config", {}).get("vt_departure_wait_util_cap")
    vt_wait_util_cap = float(vt_wait_util_cap_raw) if vt_wait_util_cap_raw is not None else None

    carry = {s: init[s] for s in stations}
    for t in times:
        for s in stations:
            carry[s] += float(arr_by_station[s].get(t, 0.0))
        served_total_map = {}
        shortage_map = {}
        for s in stations:
            req_total = req[s]["fast"][t] + req[s]["slow"][t]
            cap_total = float(params.get("vt_departure_capacity_total", {}).get(s, {}).get(t, 0.0))
            cap_fast = float(params.get("vt_departure_capacity_fast", {}).get(s, {}).get(t, 0.0))
            cap_pax = float(params.get("vertiport_cap_pax", {}).get(s, {}).get(t, 1.0e12))
            avail_departures = max(0.0, carry[s])
            served_fast = min(req[s]["fast"][t], cap_fast, avail_departures)
            served_slow = min(req[s]["slow"][t], max(0.0, cap_total - served_fast), max(0.0, avail_departures - served_fast))
            served_pax = served_fast * pax_fast + served_slow * pax_slow
            if served_pax > cap_pax + 1.0e-9:
                scale = cap_pax / max(1.0e-9, served_pax)
                served_fast *= scale
                served_slow *= scale
            # Aircraft carry is consumed by the actually served departures after any throughput scaling.
            carry[s] = max(0.0, carry[s] - (served_fast + served_slow))
            served_total = served_fast + served_slow
            served_total_map[s] = served_total
            shortage_map[s] = max(0.0, req_total - served_total)
            ratio = 1.0 if req_total <= 1.0e-9 else max(0.0, min(1.0, served_total / req_total))
            diag[s]["req_dep"][t] = req_total
            diag[s]["cap_dep"][t] = cap_total
            diag[s]["served_ratio"][t] = ratio
            util = req_total / max(1.0e-6, min(cap_total + 1.0e-9, served_total + max(0.0, carry[s]))) if req_total > 0 else 0.0
            util_eff = util
            if vt_wait_util_cap is not None:
                util_eff = min(util_eff, max(0.0, vt_wait_util_cap))
            w = vt_wait_scale * (util_eff / max(1.0e-6, 1.0 - min(0.95, util_eff))) if util_eff > 0 else 0.0
            waits[s]["fast"][t] = w
            waits[s]["slow"][t] = w
        if reposition_enabled:
            tot_short = sum(shortage_map.values())
            if tot_short > 1.0e-9:
                for src in stations:
                    send = min(max(0.0, carry[src] - reposition_reserve), reposition_max_send)
                    if send <= 1.0e-9:
                        continue
                    carry[src] -= send
                    arr_t = t + reposition_lag
                    if arr_t not in times:
                        continue
                    for dst in stations:
                        if shortage_map[dst] <= 1.0e-9:
                            continue
                        arr_by_station[dst][arr_t] += send * shortage_map[dst] / tot_short
    return waits, diag


def _aircraft_inventory_diagnostics(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Any]:
    stations = [str(s) for s in data["sets"]["hybrid_stations"]]
    groups = [str(g) for g in data["sets"]["groups"]]
    delta_t = float(data["meta"]["delta_t"])
    lag = int(data["parameters"].get("vt_turnaround_lag", 1))
    pax_fast = max(1.0e-6, float(data["parameters"].get("vt_pax_per_departure_fast", 2.0)))
    pax_slow = max(1.0e-6, float(data["parameters"].get("vt_pax_per_departure_slow", 4.0)))
    init = {s: float(data["parameters"]["vt_aircraft_init_by_station"].get(s, 0.0)) for s in stations}

    inv = {s: {t: 0.0 for t in times} for s in stations}
    dep = {s: {t: 0.0 for t in times} for s in stations}
    ret = {s: {t: 0.0 for t in times} for s in stations}
    bind = 0.0

    itinerary_dep_req: Dict[Tuple[str, int], float] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        cls = get_evtol_service_class(it)
        per = pax_fast if cls == "fast" else pax_slow
        for t in times:
            pax = sum(float(flows.get(it["id"], {}).get(g, {}).get(t, 0.0)) for g in groups)
            itinerary_dep_req[(it["id"], t)] = pax / per

    reposition_enabled = bool(data.get("config", {}).get("vt_reposition_enabled", True))
    reposition_lag = int(data.get("config", {}).get("vt_reposition_lag", 1))
    reposition_reserve = float(data.get("config", {}).get("vt_reposition_reserve", 0.5))
    reposition_max_send = float(data.get("config", {}).get("vt_reposition_max_send_per_period", 1.8))
    reposition_in = {s: {t: 0.0 for t in times} for s in stations}
    reposition_out = {s: {t: 0.0 for t in times} for s in stations}
    carry = {s: init[s] for s in stations}
    for t in times:
        for s in stations:
            carry[s] += ret[s][t] + reposition_in[s][t]
            inv[s][t] = carry[s]

        req_station_total = {
            s: sum(itinerary_dep_req.get((it["id"], t), 0.0) for it in itineraries if is_evtol_itinerary(it) and str(it.get("dep_station")) == s)
            for s in stations
        }
        served_ratio_by_station: Dict[str, float] = {}
        for s in stations:
            req_total = req_station_total[s]
            served_total = min(req_total, carry[s])
            dep[s][t] = served_total
            if req_total > carry[s] + 1.0e-9:
                bind += 1.0
            served_ratio_by_station[s] = 0.0 if req_total <= 1.0e-9 else served_total / req_total
            carry[s] = max(0.0, carry[s] - served_total)

        for it in itineraries:
            if not is_evtol_itinerary(it):
                continue
            s = str(it.get("dep_station"))
            req_i = itinerary_dep_req.get((it["id"], t), 0.0)
            dep_i = req_i * served_ratio_by_station.get(s, 0.0)
            arr = str(it.get("arr_station"))
            if arr not in ret:
                continue
            flt = _time_value(it.get("flight_time", {}), t, 0.0)
            arr_t = t + int(round(flt / max(1.0e-9, delta_t))) + lag
            if arr_t in ret[arr]:
                ret[arr][arr_t] += dep_i

        if reposition_enabled:
            shortage = {s: max(0.0, req_station_total[s] - dep[s][t]) for s in stations}
            tot_short = sum(shortage.values())
            if tot_short > 1.0e-9:
                for src in stations:
                    send = min(max(0.0, carry[src] - reposition_reserve), reposition_max_send)
                    if send <= 1.0e-9:
                        continue
                    carry[src] -= send
                    reposition_out[src][t] += send
                    arr_t = t + reposition_lag
                    if arr_t not in times:
                        continue
                    for dst in stations:
                        if shortage[dst] <= 1.0e-9:
                            continue
                        qty = send * shortage[dst] / tot_short
                        reposition_in[dst][arr_t] += qty
    return {
        "aircraft_inventory_by_station_time": inv,
        "aircraft_departures_by_station_time": dep,
        "aircraft_returns_by_station_time": ret,
        "aircraft_reposition_in_by_station_time": reposition_in,
        "aircraft_reposition_out_by_station_time": reposition_out,
        "aircraft_binding_count": bind,
    }


def _compute_vt_ev_service_probabilities(
    data: Dict[str, Any],
    times: List[int],
    station_loads: Dict[str, Dict[str, Dict[int, float]]],
    shed_ev: Dict[str, Dict[int, float]],
    shed_vt: Dict[str, Dict[int, float]],
    vt_wait_diag: Dict[str, Dict[str, Dict[int, float]]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], Dict[str, Dict[int, Dict[str, float]]]]:
    stations = [str(s) for s in data["sets"]["hybrid_stations"]]
    cfg = data.get("config", {})
    # NOTE:
    # vt_prob is a bounded operational-readiness score (0..1), not a literal realized
    # service-success probability. It blends energy adequacy and operational adequacy.
    vt_floor = float(cfg.get("vt_service_prob_floor", 0.01))
    ev_floor = float(cfg.get("ev_service_prob_floor", 0.01))
    vt_energy_weight = float(cfg.get("vt_energy_weight", 0.45))
    vt_energy_weight = max(0.0, min(1.0, vt_energy_weight))
    ev_prob = _init_station_time(stations, times, 1.0)
    vt_prob = _init_station_time(stations, times, 1.0)
    components = {s: {t: {"energy": 1.0, "departure_capacity": 1.0, "aircraft": 1.0} for t in times} for s in stations}
    for s in stations:
        for t in times:
            ev_req_kwh = float(station_loads["E_ev_req"].get(s, {}).get(t, 0.0))
            vt_req_kwh = float(station_loads["E_vt_req"].get(s, {}).get(t, 0.0))
            ev_shed_kwh = float(shed_ev.get(s, {}).get(t, 0.0)) * float(data["meta"]["delta_t"])
            vt_shed_kwh = float(shed_vt.get(s, {}).get(t, 0.0))
            p_energy = 1.0 if vt_req_kwh <= 1.0e-9 else max(0.0, min(1.0, (vt_req_kwh - vt_shed_kwh) / vt_req_kwh))
            req_dep = float(vt_wait_diag.get(s, {}).get("req_dep", {}).get(t, 0.0))
            cap_dep = float(vt_wait_diag.get(s, {}).get("cap_dep", {}).get(t, 0.0))
            p_depart = 1.0 if req_dep <= 1.0e-9 else max(0.0, min(1.0, cap_dep / req_dep))
            p_air = max(0.0, min(1.0, float(vt_wait_diag.get(s, {}).get("served_ratio", {}).get(t, 1.0))))
            p_ops = min(1.0, 0.5 * (p_depart + p_air))
            vt_integrated = vt_energy_weight * p_energy + (1.0 - vt_energy_weight) * p_ops
            vt_prob[s][t] = max(vt_floor, min(1.0, vt_integrated))
            ev_prob[s][t] = 1.0 if ev_req_kwh <= 1.0e-9 else max(ev_floor, min(1.0, (ev_req_kwh - ev_shed_kwh) / ev_req_kwh))
            components[s][t] = {"energy": p_energy, "departure_capacity": p_depart, "aircraft": p_air, "ops_blend": p_ops}
    return vt_prob, ev_prob, components


def run_hub_charging_multivot(data: Dict[str, Any]) -> Dict[str, Any]:
    times = [int(t) for t in data["sets"]["time"]]
    groups = [str(g) for g in data["sets"]["groups"]]
    stations = [str(s) for s in data["sets"]["ev_stations"]]
    itineraries = data["itineraries"]
    vot = data["parameters"]["vot"]
    lambdas = data["parameters"]["lambda"]
    cfg = data.get("config", {})

    electricity_price = {s: {t: float(data["parameters"]["electricity_price"][s][t]) for t in times} for s in stations}
    vt_service_prob = _init_station_time(stations, times, 1.0)
    ev_service_prob = _init_station_time(stations, times, 1.0)

    flows = {it["id"]: {g: {t: 0.0 for t in times} for g in groups} for it in itineraries}
    hub_diag: Dict[str, Dict[int, Dict[str, Any]]] = {s: {} for s in stations}
    iteration_history: List[Dict[str, Any]] = []

    max_iter = int(cfg.get("max_iter", 120))
    tol_flow = float(cfg.get("tol_flow", cfg.get("tol", 1e-3)))
    flow_relax = float(cfg.get("flow_step_relax", 0.65))
    flow_floor = float(cfg.get("flow_step_floor", 0.15))
    price_relax = float(cfg.get("price_step_relax", 0.45))
    readiness_relax = float(cfg.get("readiness_step_relax", price_relax))
    max_shadow = float(cfg.get("max_local_shadow_price", 2.5))
    stop_reason = "max_iter"
    prev_dx = None

    vt_diag_components: Dict[str, Dict[int, Dict[str, float]]] = {s: {t: {} for t in times} for s in stations}

    for itn in range(1, max_iter + 1):
        alpha = max(flow_floor, flow_relax / (1.0 + 0.08 * (itn - 1)))
        arc_flows = aggregate_arc_flows(itineraries, flows, times)
        travel_times = _compute_road_times(arc_flows, data["parameters"]["arcs"], times)
        ev_wait = _compute_station_waits(aggregate_ev_station_utilization(itineraries, flows, times), data["parameters"]["stations"], times)
        transfer_wait = _compute_transfer_waits(data, itineraries, flows, times)
        vt_waits, vt_wait_diag = _compute_vt_departure_waits(data, itineraries, flows, times)

        mm_penalty = {s: {t: 0.0 for t in times} for s in stations}
        cpen = cfg.get("multimodal_continuity_penalty", {})
        base = float(cpen.get("base_transfer_fragility_penalty", 0.0))
        cev = float(cpen.get("coeff_ev_unreliability", 0.0))
        cvt = float(cpen.get("coeff_vt_unreliability", 0.0))
        cjoint = float(cpen.get("coeff_joint_unreliability", 0.0))
        cover = float(cpen.get("coeff_transfer_overrun", 0.0))
        th = float(cpen.get("transfer_buffer_threshold", 0.0))
        for s in stations:
            for t in times:
                overrun = max(0.0, transfer_wait[s][t] - th)
                mm_penalty[s][t] = base + cev * (1.0 - ev_service_prob[s][t]) + cvt * (1.0 - vt_service_prob[s][t]) + cjoint * (1.0 - ev_service_prob[s][t]) * (1.0 - vt_service_prob[s][t]) + cover * overrun

        costs = compute_itinerary_costs(
            itineraries,
            travel_times,
            ev_wait,
            electricity_price,
            times,
            vt_waits,
            data["parameters"].get("transfer_base_time"),
            transfer_congestion_waits=transfer_wait,
            multimodal_continuity_penalty=mm_penalty,
        )
        flows_new, assign_details = logit_assignment(
            itineraries,
            costs,
            data["parameters"]["q"],
            vot,
            lambdas,
            times,
            vt_service_prob=vt_service_prob,
            ev_service_prob=ev_service_prob,
            vt_reliability_gamma=float(cfg.get("vt_reliability_gamma", 0.28)),
            ev_reliability_gamma=float(cfg.get("ev_reliability_gamma", 0.12)),
            multimodal_reliability_gamma=float(cfg.get("multimodal_reliability_gamma", 0.3)),
            vt_service_prob_skip_below=float(cfg.get("vt_service_prob_skip_below", 0.0)),
            ev_service_prob_skip_below=float(cfg.get("ev_service_prob_skip_below", 0.0)),
        )
        station_loads = compute_station_loads_from_flows(data, itineraries, flows_new, times)
        B_out, P_out, shed_ev, shed_vt, shadow_prices, _, lp_diag = solve_shared_power_inventory_lp(
            data,
            station_loads["E_vt_req"],
            station_loads["E_ev_req"],
        )

        dx = 0.0
        for it in itineraries:
            for g in groups:
                for t in times:
                    old = float(flows[it["id"]][g][t])
                    new = float(flows_new[it["id"]][g][t])
                    dx = max(dx, abs(new - old))
                    flows[it["id"]][g][t] = (1.0 - alpha) * old + alpha * new
        if prev_dx is not None and dx > prev_dx * 1.03:
            alpha = max(flow_floor, 0.7 * alpha)
        prev_dx = dx

        vt_service_prob_new, ev_service_prob_new, vt_diag_components = _compute_vt_ev_service_probabilities(
            data, times, station_loads, shed_ev, shed_vt, vt_wait_diag
        )

        for s in stations:
            for t in times:
                base_p = float(data["parameters"]["electricity_price"][s][t])
                mu_dual = max(0.0, float(shadow_prices.get(s, {}).get(t, 0.0) or 0.0))
                mu_used = min(max_shadow, mu_dual)
                electricity_price[s][t] = (1.0 - price_relax) * electricity_price[s][t] + price_relax * (base_p + mu_used)
                vt_service_prob[s][t] = (1.0 - readiness_relax) * vt_service_prob[s][t] + readiness_relax * vt_service_prob_new[s][t]
                ev_service_prob[s][t] = (1.0 - readiness_relax) * ev_service_prob[s][t] + readiness_relax * ev_service_prob_new[s][t]

                dt = float(data["meta"]["delta_t"])
                eta = float(data["parameters"].get("vertiport_storage", {}).get(s, {}).get("eta_ch", 1.0))
                served_ev_kw = max(0.0, float(station_loads["P_ev_req_kw"].get(s, {}).get(t, 0.0)) - float(shed_ev.get(s, {}).get(t, 0.0)))
                vt_charge_power_from_grid_kw = max(0.0, float(P_out.get(s, {}).get(t, 0.0)))
                b_t = float(B_out.get(s, {}).get(t, 0.0))
                b_next = float(B_out.get(s, {}).get(t + 1, b_t))
                delta_b = b_next - b_t
                charge_kw = max(0.0, delta_b / max(1.0e-9, eta * dt))
                discharge_kw = max(0.0, -delta_b / max(1.0e-9, dt))
                grid_draw_kw = served_ev_kw + vt_charge_power_from_grid_kw
                hub_diag[s][t] = {
                    "grid_connection_cap_kw": float(data["parameters"]["stations"][s]["P_site"][t]),
                    "actual_grid_draw_kw": grid_draw_kw,
                    "storage_charge_kw": charge_kw,
                    "storage_discharge_kw": discharge_kw,
                    "vt_charge_power_from_grid_kw": vt_charge_power_from_grid_kw,
                    "storage_state_kwh": float(B_out.get(s, {}).get(t, 0.0)),
                    "storage_state_next_kwh": float(B_out.get(s, {}).get(t + 1, B_out.get(s, {}).get(t, 0.0))),
                    "local_shadow_price_dual_raw": mu_dual,
                    "local_shadow_price_used_for_price": mu_used,
                    "storage_port_semantics": "overlap_capped_charge_with_departure",
                    "vt_operational_readiness": vt_service_prob[s][t],
                    "ev_service_probability": ev_service_prob[s][t],
                    "vt_service_components": vt_diag_components[s][t],
                    "transfer_congestion_wait": transfer_wait[s][t],
                    "multimodal_continuity_penalty": mm_penalty[s][t],
                }
        iteration_history.append({"iteration": itn, "alpha": alpha, "max_flow_delta": dx})
        if dx <= tol_flow:
            stop_reason = "tol_flow"
            break

    mode_share = _mode_share_by_group_time(itineraries, flows, groups, times)
    group_metrics = _group_time_supermode_metrics(itineraries, flows, costs, groups, times, vot)
    aircraft_diag = _aircraft_inventory_diagnostics(data, itineraries, flows, times)

    return {
        "flows": flows,
        "costs": costs,
        "mode_share_by_group_time": mode_share,
        "group_time_supermode_metrics": group_metrics,
        "effective_electricity_price": electricity_price,
        "vt_operational_readiness": vt_service_prob,
        "vt_operational_readiness_meta": {
            "interpretation": "bounded operational readiness score in [0,1], not a literal realized service-success probability",
            "construction": "convex blend of VT energy adequacy and VT operational adequacy (departure-capacity + aircraft served-ratio), with floor clipping",
            "ops_blend_component": "0.5 * (departure_capacity_component + aircraft_component)",
        },
        "ev_service_prob": ev_service_prob,
        "hub_diagnostics": hub_diag,
        "assignment_diagnostics": assign_details,
        "iteration_history": iteration_history,
        "convergence": {
            "final_iteration": iteration_history[-1]["iteration"] if iteration_history else 0,
            "final_max_flow_delta": iteration_history[-1]["max_flow_delta"] if iteration_history else None,
            "stopping_reason": stop_reason,
            "tol_flow": tol_flow,
        },
        "shared_power_price_signal_check": {
            "solver_used": str(data.get("diagnostics_runtime", {}).get("lp_failure", {}).get("fallback_solver", "highs"))
            if not bool(data.get("diagnostics_runtime", {}).get("lp_ok", True))
            else "highs",
            "lp_dual_available": bool(data.get("diagnostics_runtime", {}).get("lp_ok", False)),
            "shared_power_fallback_used": not bool(data.get("diagnostics_runtime", {}).get("lp_ok", True)),
            "binding_cap_total_count": int(lp_diag.get("binding_cap_total_count", 0)) if isinstance(lp_diag, dict) else 0,
            "binding_cap_with_positive_dual_count": int(lp_diag.get("binding_cap_with_positive_dual_count", 0)) if isinstance(lp_diag, dict) else 0,
        },
        **aircraft_diag,
        "summary": {
            "time_periods": len(times),
            "hybrid_hubs": list(data["sets"]["hybrid_stations"]),
            "avg_vt_service_probability": sum(vt_service_prob[s][t] for s in stations for t in times) / max(1, len(stations) * len(times)),
            "avg_ev_service_probability": sum(ev_service_prob[s][t] for s in stations for t in times) / max(1, len(stations) * len(times)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--report_out", required=True)
    args = parser.parse_args()
    data = load_data(args.data, args.schema)
    result = run_hub_charging_multivot(data)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
