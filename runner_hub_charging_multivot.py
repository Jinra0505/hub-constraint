import argparse
import copy
import json
import math
from typing import Any, Dict, List, Tuple

from .assignment import (
    aggregate_arc_flows,
    aggregate_ev_station_utilization,
    classify_supermode,
    compute_itinerary_costs,
    get_evtol_service_class,
    is_evtol_itinerary,
    is_multimodal_evtol,
    logit_assignment,
)
from .charging_hiGHS_and_gurobi_bound_fix import compute_station_loads_from_flows, solve_shared_power_inventory_lp
from .congestion import compute_road_times, compute_station_waits, compute_vt_departure_waits
from .data_loader import load_data


def _value_by_time(value: Any, t: int, default: float = 0.0) -> float:
    if isinstance(value, dict):
        if t in value:
            return float(value[t])
        ts = str(t)
        if ts in value:
            return float(value[ts])
        return float(default)
    if value is None:
        return float(default)
    return float(value)


def _msa_update(prev: Dict[str, Dict[int, float]], nxt: Dict[str, Dict[int, float]], alpha: float) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for s in set(prev.keys()) | set(nxt.keys()):
        out[s] = {}
        for t in set(prev.get(s, {}).keys()) | set(nxt.get(s, {}).keys()):
            out[s][t] = (1.0 - alpha) * float(prev.get(s, {}).get(t, 0.0)) + alpha * float(nxt.get(s, {}).get(t, 0.0))
    return out


def _initialize_by_station(stations: List[str], times: List[int], value: float) -> Dict[str, Dict[int, float]]:
    return {s: {t: float(value) for t in times} for s in stations}


def _apply_hub_power_scenario(data: Dict[str, Any]) -> None:
    cfg = data.setdefault("config", {})
    scenario = cfg.get("hub_power_scenario")
    factor = cfg.get("hub_power_scale")
    if scenario is None and factor is None:
        return

    scenario_map = cfg.get("hub_power_scenarios", {"baseline": 1.0, "moderate": 0.8, "severe": 0.6})
    scale = float(factor) if factor is not None else float(scenario_map.get(str(scenario), 1.0))
    if scale <= 0:
        raise ValueError("hub power scale must be positive")

    station_caps = data.get("parameters", {}).get("stations", {})
    for s, p in station_caps.items():
        p_site = p.get("P_site", {})
        if isinstance(p_site, dict):
            p["P_site"] = {t: float(v) * scale for t, v in p_site.items()}
        else:
            p["P_site"] = float(p_site) * scale


def _build_fallback_travel_times(
    itineraries: List[Dict[str, Any]],
    times: List[int],
    config: Dict[str, Any],
) -> Dict[str, Dict[int, float]]:
    """Road-time fallback when arc_params are unavailable.

    Precedence per segment:
    1) segment explicit time field (travel_time / time / tau / tau0),
    2) itinerary-level totals split over segment-count for same t,
    3) segment tau0-like value,
    4) positive config default fallback_road_segment_time.
    """
    default_seg_time = max(1.0e-6, float(config.get("fallback_road_segment_time", 0.25) or 0.25))
    travel_times: Dict[str, Dict[int, float]] = {}
    seg_fields = ("travel_time", "time", "tau", "tau0")
    for it in itineraries:
        all_segments = (it.get("road_arcs", []) or []) + (it.get("access_arcs", []) or []) + (it.get("egress_arcs", []) or [])
        per_t_count = {t: 0 for t in times}
        for seg in all_segments:
            t = seg.get("t")
            if t in per_t_count:
                per_t_count[int(t)] += 1

        for seg in all_segments:
            arc = seg.get("arc")
            t_raw = seg.get("t")
            if arc is None or t_raw is None:
                continue
            t = int(t_raw)
            travel_times.setdefault(str(arc), {tt: default_seg_time for tt in times})

            seg_time = None
            for fld in seg_fields:
                if fld in seg:
                    seg_time = _value_by_time(seg.get(fld), t, None)  # type: ignore[arg-type]
                    if seg_time is not None:
                        break

            if seg_time is None:
                it_total = 0.0
                for fld in ("road_time", "access_time", "egress_time", "road_tt", "access_tt", "egress_tt"):
                    if fld in it:
                        it_total += max(0.0, _value_by_time(it.get(fld), t, 0.0))
                if it_total > 0.0 and per_t_count.get(t, 0) > 0:
                    seg_time = it_total / max(1, per_t_count[t])

            if seg_time is None and "tau0" in seg:
                seg_time = _value_by_time(seg.get("tau0"), t, None)  # type: ignore[arg-type]

            if seg_time is None:
                seg_time = default_seg_time
            travel_times[str(arc)][t] = max(0.0, float(seg_time))
    return travel_times


def _build_itineraries_by_od(itineraries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for it in itineraries:
        key = f"{it['od'][0]}-{it['od'][1]}"
        out.setdefault(key, []).append(it)
    return out


def _utility_for_alt(utilities: Dict[str, Dict[str, Dict[int, float]]], it_id: str, group: str, t: int) -> float:
    return float(utilities.get(it_id, {}).get(group, {}).get(t, -float("inf")))


def _reroute_excess_with_conditional_logit(
    *,
    flows: Dict[str, Dict[str, Dict[int, float]]],
    itineraries_by_od: Dict[str, List[Dict[str, Any]]],
    utilities: Dict[str, Dict[str, Dict[int, float]]],
    od_key: str,
    group: str,
    t: int,
    excess_pax: float,
    blocked_dep_station: str | None,
    unserved_demand: Dict[str, Dict[str, Dict[int, float]]],
    reroute_logit_temperature: float = 1.0,
) -> Dict[str, float]:
    if excess_pax <= 0.0:
        return {"to_evtol": 0.0, "to_multimodal": 0.0, "to_ev": 0.0, "unserved": 0.0, "feasible_vt_alts": 0.0}
    feasible: List[Tuple[Dict[str, Any], float]] = []
    feasible_vt = 0.0
    for it in itineraries_by_od.get(od_key, []):
        dep = it.get("dep_station")
        if blocked_dep_station is not None and is_evtol_itinerary(it) and dep == blocked_dep_station:
            continue
        util = _utility_for_alt(utilities, it["id"], group, t)
        if (math.isinf(util) and util < 0.0) or math.isnan(util):
            continue
        feasible.append((it, util))
        if is_evtol_itinerary(it):
            feasible_vt += 1.0
    if not feasible:
        unserved_demand.setdefault(od_key, {}).setdefault(group, {})[t] = unserved_demand.setdefault(od_key, {}).setdefault(group, {}).get(t, 0.0) + excess_pax
        return {"to_evtol": 0.0, "to_multimodal": 0.0, "to_ev": 0.0, "unserved": excess_pax, "feasible_vt_alts": feasible_vt}
    temp = max(1.0e-6, float(reroute_logit_temperature))
    max_u = max(u / temp for _, u in feasible)
    ws = [math.exp((u / temp) - max_u) for _, u in feasible]
    den = max(1.0e-12, sum(ws))
    to_evtol = to_multimodal = to_ev = 0.0
    for (it, _), w in zip(feasible, ws):
        add = excess_pax * (w / den)
        flows[it["id"]].setdefault(group, {})[t] = flows[it["id"]].get(group, {}).get(t, 0.0) + add
        if is_multimodal_evtol(it):
            to_multimodal += add
        elif is_evtol_itinerary(it):
            to_evtol += add
        else:
            to_ev += add
    return {"to_evtol": to_evtol, "to_multimodal": to_multimodal, "to_ev": to_ev, "unserved": 0.0, "feasible_vt_alts": feasible_vt}


def _enforce_aircraft_inventory(
    flows: Dict[str, Dict[str, Dict[int, float]]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
    delta_t: float,
    utilities: Dict[str, Dict[str, Dict[int, float]]],
    unserved_demand: Dict[str, Dict[str, Dict[int, float]]],
    vt_pax_per_departure_fast: float,
    vt_pax_per_departure_slow: float,
    vt_turnaround_lag: int,
    vt_aircraft_init_by_station: Dict[str, float],
    reroute_logit_temperature: float = 1.0,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Any]]:
    """Ported minimal aircraft-stock + return-lag constraint from legacy runner."""
    stations = sorted({st for it in itineraries if is_evtol_itinerary(it) for st in [it.get("dep_station"), it.get("arr_station")] if st is not None})
    vt_by_dep: Dict[str, List[Dict[str, Any]]] = {s: [] for s in stations}
    for it in itineraries:
        dep = it.get("dep_station")
        if is_evtol_itinerary(it) and dep in vt_by_dep:
            vt_by_dep[dep].append(it)
    itineraries_by_od = _build_itineraries_by_od(itineraries)

    max_t = max(times) if times else 0
    ext_t = list(range(min(times), max_t + int(vt_turnaround_lag) + 6)) if times else []
    ret_sched = {s: {tt: 0.0 for tt in ext_t} for s in stations}
    inv = {s: {t: 0.0 for t in times} for s in stations}
    dep_out = {s: {t: 0.0 for t in times} for s in stations}
    ret_out = {s: {t: 0.0 for t in times} for s in stations}
    avail = {s: float(vt_aircraft_init_by_station.get(s, 0.0)) for s in stations}

    stats = {"binding_count": 0.0, "to_evtol": 0.0, "to_multimodal": 0.0, "to_ev": 0.0, "unserved": 0.0}
    for t in times:
        reductions: Dict[Tuple[str, str, str], float] = {}
        for s in stations:
            arrivals = float(ret_sched.get(s, {}).get(t, 0.0))
            ret_out[s][t] = arrivals
            avail_start = max(0.0, avail[s] + arrivals)
            inv[s][t] = avail_start
            req_dep = 0.0
            for it in vt_by_dep.get(s, []):
                cls = get_evtol_service_class(it)
                pax_per_dep = max(1.0e-6, float(vt_pax_per_departure_fast if cls == "fast" else vt_pax_per_departure_slow))
                for _, tm in flows.get(it["id"], {}).items():
                    req_dep += float(tm.get(t, 0.0)) / pax_per_dep
            served_ratio = 1.0
            if req_dep > avail_start + 1.0e-12:
                served_ratio = avail_start / max(req_dep, 1.0e-12)
                stats["binding_count"] += 1.0

            served_dep = 0.0
            for it in vt_by_dep.get(s, []):
                od_key = f"{it['od'][0]}-{it['od'][1]}"
                cls = get_evtol_service_class(it)
                pax_per_dep = max(1.0e-6, float(vt_pax_per_departure_fast if cls == "fast" else vt_pax_per_departure_slow))
                for g, tm in flows[it["id"]].items():
                    original = float(tm.get(t, 0.0))
                    served = original * served_ratio
                    delta = original - served
                    if delta > 0.0:
                        reductions[(od_key, g, s)] = reductions.get((od_key, g, s), 0.0) + delta
                    flows[it["id"]].setdefault(g, {})[t] = served
                    flights = served / pax_per_dep
                    served_dep += flights
                    arr_station = it.get("arr_station")
                    lag = int(math.ceil(max(0.0, _value_by_time(it.get("flight_time"), t, 0.0)) / max(delta_t, 1.0e-9))) + int(vt_turnaround_lag)
                    arr_t = t + lag
                    if arr_station in ret_sched:
                        ret_sched[arr_station][arr_t] = ret_sched[arr_station].get(arr_t, 0.0) + flights
            dep_out[s][t] = served_dep
            avail[s] = max(0.0, avail_start - served_dep)

        for (od_key, g, blocked_dep), delta in reductions.items():
            rr = _reroute_excess_with_conditional_logit(
                flows=flows,
                itineraries_by_od=itineraries_by_od,
                utilities=utilities,
                od_key=od_key,
                group=g,
                t=t,
                excess_pax=delta,
                blocked_dep_station=blocked_dep,
                unserved_demand=unserved_demand,
                reroute_logit_temperature=reroute_logit_temperature,
            )
            stats["to_evtol"] += rr["to_evtol"]
            stats["to_multimodal"] += rr["to_multimodal"]
            stats["to_ev"] += rr["to_ev"]
            stats["unserved"] += rr["unserved"]
    return flows, {
        "aircraft_inventory_by_station_time": inv,
        "aircraft_departures_by_station_time": dep_out,
        "aircraft_returns_by_station_time": ret_out,
        "aircraft_binding_count": stats["binding_count"],
        "aircraft_rerouted_to_evtol": stats["to_evtol"],
        "aircraft_rerouted_to_multimodal": stats["to_multimodal"],
        "aircraft_rerouted_to_ev": stats["to_ev"],
        "aircraft_unserved": stats["unserved"],
    }


def _compute_group_time_supermode_metrics(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    cost_details: Dict[str, Dict[int, Dict[str, float]]],
    generalized_costs: Dict[str, Dict[str, Dict[int, float]]] | None,
    groups: List[str],
    times: List[int],
    vot: Dict[str, Dict[int, float]],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    lookup = {it["id"]: it for it in itineraries}
    for g in groups:
        metrics[g] = {}
        for t in times:
            agg: Dict[str, Dict[str, float]] = {
                "EV": {"flow": 0.0, "gen_cost": 0.0, "perceived_cost": 0.0, "travel_time": 0.0, "money_plus_charge": 0.0, "transfer_burden": 0.0, "vt_wait": 0.0},
                "eVTOL": {"flow": 0.0, "gen_cost": 0.0, "perceived_cost": 0.0, "travel_time": 0.0, "money_plus_charge": 0.0, "transfer_burden": 0.0, "vt_wait": 0.0},
                "EV_to_eVTOL": {"flow": 0.0, "gen_cost": 0.0, "perceived_cost": 0.0, "travel_time": 0.0, "money_plus_charge": 0.0, "transfer_burden": 0.0, "vt_wait": 0.0},
            }
            for it_id, g_map in flows.items():
                f = float(g_map.get(g, {}).get(t, 0.0))
                if f <= 0:
                    continue
                it = lookup[it_id]
                sm = classify_supermode(it)
                comp = cost_details[it_id][t]
                cb = comp.get("cost_breakdown", {})
                agg[sm]["flow"] += f
                tt = float(comp.get("TT", 0.0))
                money_charge = float(comp.get("Money", 0.0) + comp.get("ChargeCost", 0.0))
                agg[sm]["gen_cost"] += f * (float(vot[g][t]) * tt + money_charge)
                if generalized_costs is not None:
                    agg[sm]["perceived_cost"] += f * float(generalized_costs.get(it_id, {}).get(g, {}).get(t, float(vot[g][t]) * tt + money_charge))
                else:
                    agg[sm]["perceived_cost"] += f * (float(vot[g][t]) * tt + money_charge)
                agg[sm]["travel_time"] += f * tt
                agg[sm]["money_plus_charge"] += f * money_charge
                agg[sm]["transfer_burden"] += f * float(cb.get("multimodal_extra_penalty", 0.0) + cb.get("transfer_time_applied", 0.0))
                agg[sm]["vt_wait"] += f * float(cb.get("vt_departure_wait_applied", 0.0))
            metrics[g][t] = {}
            for sm, vals in agg.items():
                den = max(1e-9, vals["flow"])
                metrics[g][t][sm] = {
                    "flow": vals["flow"],
                    "avg_generalized_cost": vals["gen_cost"] / den,
                    "avg_perceived_cost": vals["perceived_cost"] / den,
                    "avg_travel_time": vals["travel_time"] / den,
                    "avg_money_plus_charge_cost": vals["money_plus_charge"] / den,
                    "avg_transfer_related_burden": vals["transfer_burden"] / den,
                    "avg_evtol_departure_wait_exposure": vals["vt_wait"] / den,
                }
    return metrics


def run_hub_charging_multivot(data: Dict[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(data)
    cfg = data.setdefault("config", {})
    cfg.setdefault("use_distribution_grid", False)
    cfg.setdefault("max_iter", 40)
    cfg.setdefault("tol", 1e-3)
    cfg.setdefault("flow_msa_alpha", None)
    cfg.setdefault("multimodal_continuity_penalty", {})
    cfg.setdefault("ev_service_prob_floor", 1e-4)
    cfg.setdefault("vt_service_prob_floor", 1e-4)
    cfg.setdefault("fallback_road_segment_time", 0.25)
    cfg.setdefault("tol_price", cfg.get("tol", 1e-3))
    cfg.setdefault("tol_flow", cfg.get("tol", 1e-3))
    cfg.setdefault("tol_service_prob", cfg.get("tol", 1e-3))
    cfg.setdefault("reroute_logit_temperature", 1.0)
    _apply_hub_power_scenario(data)

    times = [int(t) for t in data["sets"]["time"]]
    groups = [str(g) for g in data["sets"]["groups"]]
    itineraries = data["itineraries"]
    demand = data["parameters"]["q"]
    vot = data["parameters"]["vot"]
    lambdas = data["parameters"]["lambda"]
    ev_stations = [str(s) for s in data["sets"]["ev_stations"]]
    hybrid_stations = [str(s) for s in data["sets"]["hybrid_stations"]]

    travel_times_fallback_used = False
    travel_times: Dict[str, Dict[int, float]] = {}
    if data.get("parameters", {}).get("arc_params"):
        travel_times = compute_road_times(
            {arc: {t: 0.0 for t in times} for arc in data["parameters"]["arc_params"]},
            data["parameters"]["arc_params"],
            {t: 1.0 for t in times},
            times,
        )
    else:
        travel_times = _build_fallback_travel_times(itineraries, times, cfg)
        travel_times_fallback_used = True

    ev_waits = _initialize_by_station(ev_stations, times, 0.0)
    vt_waits = {s: {"fast": {t: 0.0 for t in times}, "slow": {t: 0.0 for t in times}} for s in hybrid_stations}
    electricity_price = copy.deepcopy(data["parameters"]["electricity_price"])
    vt_service_prob = _initialize_by_station(hybrid_stations, times, 1.0)
    ev_service_prob = _initialize_by_station(ev_stations, times, 1.0)

    flows = {it["id"]: {g: {t: 0.0 for t in times} for g in groups} for it in itineraries}
    diagnostics: Dict[str, Any] = {
        "iteration_history": [],
        "hub_time": {},
        "mode_share_by_group_time": {},
    }

    aircraft_diag_last: Dict[str, Any] = {}
    for itn in range(1, int(cfg["max_iter"]) + 1):
        costs = compute_itinerary_costs(
            itineraries,
            travel_times,
            ev_waits,
            electricity_price,
            times,
            vt_departure_waits=vt_waits,
            transfer_time_default=float(cfg.get("transfer_time_default", 0.0) or 0.0),
            multimodal_penalty_cfg=cfg.get("multimodal_continuity_penalty", {}),
            vt_service_prob=vt_service_prob,
            ev_service_prob=ev_service_prob,
        )
        flows_target, details = logit_assignment(
            itineraries,
            costs,
            demand,
            vot,
            lambdas,
            times,
            vt_service_prob=vt_service_prob,
            ev_service_prob=ev_service_prob,
            vt_service_prob_floor=float(cfg.get("vt_service_prob_floor", 1e-4)),
            ev_service_prob_floor=float(cfg.get("ev_service_prob_floor", 1e-4)),
            vt_reliability_gamma=float(cfg.get("vt_reliability_gamma", 0.0) or 0.0),
            ev_reliability_gamma=float(cfg.get("ev_reliability_gamma", 0.0) or 0.0),
        )
        vt_pax_fast = float(data.get("parameters", {}).get("vt_pax_per_departure_fast", 2.0) or 2.0)
        vt_pax_slow = float(data.get("parameters", {}).get("vt_pax_per_departure_slow", 4.0) or 4.0)
        vt_turn_lag = int(data.get("parameters", {}).get("vt_turnaround_lag", 1) or 1)
        vt_init = {str(k): float(v) for k, v in data.get("parameters", {}).get("vt_aircraft_init_by_station", {}).items()}
        flows_target, aircraft_diag = _enforce_aircraft_inventory(
            flows_target,
            itineraries,
            times,
            float(data["meta"]["delta_t"]),
            details.get("utilities", {}),
            details.get("unserved_demand", {}),
            vt_pax_fast,
            vt_pax_slow,
            vt_turn_lag,
            vt_init,
            reroute_logit_temperature=float(cfg.get("reroute_logit_temperature", 1.0) or 1.0),
        )
        aircraft_diag_last = aircraft_diag

        max_flow_delta = 0.0
        alpha = float(cfg.get("flow_msa_alpha") or (1.0 / itn))
        for it in itineraries:
            it_id = it["id"]
            for g in groups:
                for t in times:
                    prev_flow = float(flows[it_id][g][t])
                    max_flow_delta = max(max_flow_delta, abs(float(flows_target[it_id][g][t]) - prev_flow))
                    flows[it_id][g][t] = (1.0 - alpha) * prev_flow + alpha * float(flows_target[it_id][g][t])

        arc_flows = aggregate_arc_flows(itineraries, flows, times)
        if data.get("parameters", {}).get("arc_params"):
            travel_times_new = compute_road_times(arc_flows, data["parameters"]["arc_params"], {t: 1.0 for t in times}, times)
            travel_times = _msa_update(travel_times, travel_times_new, alpha)

        ev_util = aggregate_ev_station_utilization(itineraries, flows, times)
        ev_waits = compute_station_waits(ev_util, data["parameters"]["stations"], times)
        vt_waits, _ = compute_vt_departure_waits(data, itineraries, flows, times, cfg)

        station_loads = compute_station_loads_from_flows(data, itineraries, flows, times)
        _, _, shed_ev_out, shed_vt_out, shadow_prices, _, lp_diag = solve_shared_power_inventory_lp(
            data,
            station_loads["E_vt_req"],
            station_loads["E_ev_req"],
        )

        max_price_delta = 0.0
        max_vt_prob_delta = 0.0
        max_ev_prob_delta = 0.0
        for s in ev_stations:
            for t in times:
                base_price = float(data["parameters"]["electricity_price"][s][t])
                local_mu = float(shadow_prices.get(s, {}).get(t, 0.0) or 0.0)
                eff_price_new = max(0.0, base_price + local_mu)
                max_price_delta = max(max_price_delta, abs(electricity_price[s][t] - eff_price_new))
                electricity_price[s][t] = (1.0 - alpha) * electricity_price[s][t] + alpha * eff_price_new

                ev_req = float(station_loads["E_ev_req"].get(s, {}).get(t, 0.0))
                ev_shed = float(shed_ev_out.get(s, {}).get(t, 0.0)) * float(data["meta"]["delta_t"])
                ev_prob = 1.0 if ev_req <= 1e-9 else max(0.0, min(1.0, (ev_req - ev_shed) / ev_req))
                max_ev_prob_delta = max(max_ev_prob_delta, abs(ev_service_prob[s][t] - ev_prob))
                ev_service_prob[s][t] = (1.0 - alpha) * ev_service_prob[s][t] + alpha * ev_prob

                vt_req = float(station_loads["E_vt_req"].get(s, {}).get(t, 0.0))
                vt_shed = float(shed_vt_out.get(s, {}).get(t, 0.0))
                vt_prob = 1.0 if vt_req <= 1e-9 else max(0.0, min(1.0, (vt_req - vt_shed) / vt_req))
                if s in vt_service_prob:
                    max_vt_prob_delta = max(max_vt_prob_delta, abs(vt_service_prob[s][t] - vt_prob))
                    vt_service_prob[s][t] = (1.0 - alpha) * vt_service_prob[s][t] + alpha * vt_prob

                diagnostics["hub_time"].setdefault(s, {})[t] = {
                    "pure_ev_charging_kwh": float(station_loads["E_ev_pure_req"].get(s, {}).get(t, 0.0)),
                    "access_ev_charging_kwh": float(station_loads["E_ev_access_req"].get(s, {}).get(t, 0.0)),
                    "evtol_charging_kwh": float(station_loads["E_vt_req"].get(s, {}).get(t, 0.0)),
                    "total_requested_power_kw": float(station_loads["P_total_req"].get(s, {}).get(t, 0.0)),
                    "effective_power_cap_kw": float(data["parameters"]["stations"][s]["P_site"][t]),
                    "local_shadow_price": local_mu,
                    "shed_ev_kwh": ev_shed,
                    "shed_vt_kwh": vt_shed,
                    "effective_electricity_price": float(electricity_price[s][t]),
                    "vt_service_probability": float(vt_service_prob.get(s, {}).get(t, 1.0)),
                    "ev_service_probability": float(ev_service_prob.get(s, {}).get(t, 1.0)),
                }

        diagnostics["iteration_history"].append(
            {
                "iteration": itn,
                "alpha": alpha,
                "max_price_delta": max_price_delta,
                "max_flow_delta": max_flow_delta,
                "max_vt_service_prob_delta": max_vt_prob_delta,
                "max_ev_service_prob_delta": max_ev_prob_delta,
                "max_joint_delta": max(max_price_delta, max_flow_delta, max_vt_prob_delta, max_ev_prob_delta),
                "lp_solver": lp_diag.get("solver") if isinstance(lp_diag, dict) else "unknown",
                "unserved_demand_total": float(details.get("unserved_demand_total", 0.0)),
                "aircraft_binding_count": float(aircraft_diag_last.get("aircraft_binding_count", 0.0)),
            }
        )
        if (
            max_price_delta <= float(cfg.get("tol_price", cfg.get("tol", 1e-3)))
            and max_flow_delta <= float(cfg.get("tol_flow", cfg.get("tol", 1e-3)))
            and max(max_vt_prob_delta, max_ev_prob_delta) <= float(cfg.get("tol_service_prob", cfg.get("tol", 1e-3)))
        ):
            break

    group_time_super = _compute_group_time_supermode_metrics(
        itineraries,
        flows,
        costs,
        details.get("generalized_costs"),
        groups,
        times,
        vot,
    )
    mode_share: Dict[str, Dict[int, Dict[str, float]]] = {}
    for g in groups:
        mode_share[g] = {}
        for t in times:
            totals = {"EV": 0.0, "eVTOL": 0.0, "EV_to_eVTOL": 0.0}
            for it in itineraries:
                totals[classify_supermode(it)] += float(flows[it["id"]][g][t])
            den = max(1e-9, sum(totals.values()))
            mode_share[g][t] = {k: v / den for k, v in totals.items()}

    # story summary: compare service degradation under charging stress
    total_access = sum(v["access_ev_charging_kwh"] for s in diagnostics["hub_time"].values() for v in s.values())
    total_vt = sum(v["evtol_charging_kwh"] for s in diagnostics["hub_time"].values() for v in s.values())
    vt_prob_avg = sum(v["vt_service_probability"] for s in diagnostics["hub_time"].values() for v in s.values()) / max(1, sum(len(s) for s in diagnostics["hub_time"].values()))
    ev_prob_avg = sum(v["ev_service_probability"] for s in diagnostics["hub_time"].values() for v in s.values()) / max(1, sum(len(s) for s in diagnostics["hub_time"].values()))

    mode_totals_by_group = {}
    for g in groups:
        sums = {"EV": 0.0, "eVTOL": 0.0, "EV_to_eVTOL": 0.0}
        for t in times:
            for k in sums:
                sums[k] += mode_share[g][t][k]
        mode_totals_by_group[g] = {k: v / max(1, len(times)) for k, v in sums.items()}

    shift_group = max(groups, key=lambda gg: mode_totals_by_group[gg]["EV_to_eVTOL"]) if groups else None

    return {
        "flows": flows,
        "costs": costs,
        "mode_share_by_group_time": mode_share,
        "group_time_supermode_metrics": group_time_super,
        "effective_electricity_price": electricity_price,
        "vt_service_prob": vt_service_prob,
        "ev_service_prob": ev_service_prob,
        "hub_diagnostics": diagnostics["hub_time"],
        "iteration_history": diagnostics["iteration_history"],
        "travel_times_fallback_used": travel_times_fallback_used,
        "aircraft_inventory_by_station_time": aircraft_diag_last.get("aircraft_inventory_by_station_time", {}),
        "aircraft_departures_by_station_time": aircraft_diag_last.get("aircraft_departures_by_station_time", {}),
        "aircraft_returns_by_station_time": aircraft_diag_last.get("aircraft_returns_by_station_time", {}),
        "aircraft_binding_count": aircraft_diag_last.get("aircraft_binding_count", 0.0),
        "aircraft_curtailment_summary": {
            "rerouted_to_evtol": aircraft_diag_last.get("aircraft_rerouted_to_evtol", 0.0),
            "rerouted_to_multimodal": aircraft_diag_last.get("aircraft_rerouted_to_multimodal", 0.0),
            "rerouted_to_ev": aircraft_diag_last.get("aircraft_rerouted_to_ev", 0.0),
            "unserved": aircraft_diag_last.get("aircraft_unserved", 0.0),
        },
        "ev_charging_consistency_check": {
            s: {
                t: float(station_loads["E_ev_pure_req"].get(s, {}).get(t, 0.0))
                + float(station_loads["E_ev_access_req"].get(s, {}).get(t, 0.0))
                - float(station_loads["E_ev_req"].get(s, {}).get(t, 0.0))
                for t in times
            }
            for s in ev_stations
        },
        "summary": {
            "scheme": "A_shared_hub_access_ev_and_evtol",
            "use_distribution_grid": bool(cfg.get("use_distribution_grid", False)),
            "ev_to_evtol_dual_exposure_enabled": True,
            "total_access_ev_charging_kwh": total_access,
            "total_evtol_charging_kwh": total_vt,
            "avg_vt_service_probability": vt_prob_avg,
            "avg_ev_service_probability": ev_prob_avg,
            "inference": "EV_to_eVTOL is expected to be more sensitive than pure eVTOL under tighter hub caps because it consumes both access-EV and eVTOL hub resources.",
            "group_with_highest_avg_ev_to_evtol_share": shift_group,
            "group_average_mode_share": mode_totals_by_group,
        },
        "unused_legacy_paths": [
            "mfd.py",
            "runner_new_background_boundary_flow.py MFD and boundary-flow logic",
            "CBD accumulation states n_t/g_t",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hub charging + multi-VOT runner without MFD dynamics")
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
