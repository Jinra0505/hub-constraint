import argparse
import copy
import json
from typing import Any, Dict, List

from .assignment import (
    aggregate_arc_flows,
    aggregate_ev_station_utilization,
    classify_supermode,
    compute_itinerary_costs,
    logit_assignment,
)
from .charging_hiGHS_and_gurobi_bound_fix import compute_station_loads_from_flows, solve_shared_power_inventory_lp
from .congestion import compute_road_times, compute_station_waits, compute_vt_departure_waits
from .data_loader import load_data


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


def _compute_group_time_supermode_metrics(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    cost_details: Dict[str, Dict[int, Dict[str, float]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    lookup = {it["id"]: it for it in itineraries}
    for g in groups:
        metrics[g] = {}
        for t in times:
            agg: Dict[str, Dict[str, float]] = {
                "EV": {"flow": 0.0, "gen_cost": 0.0, "travel_time": 0.0, "transfer_burden": 0.0, "vt_wait": 0.0},
                "eVTOL": {"flow": 0.0, "gen_cost": 0.0, "travel_time": 0.0, "transfer_burden": 0.0, "vt_wait": 0.0},
                "EV_to_eVTOL": {"flow": 0.0, "gen_cost": 0.0, "travel_time": 0.0, "transfer_burden": 0.0, "vt_wait": 0.0},
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
                agg[sm]["gen_cost"] += f * float(comp.get("TT", 0.0) + comp.get("Money", 0.0) + comp.get("ChargeCost", 0.0))
                agg[sm]["travel_time"] += f * float(comp.get("TT", 0.0))
                agg[sm]["transfer_burden"] += f * float(cb.get("multimodal_extra_penalty", 0.0) + cb.get("transfer_time_applied", 0.0))
                agg[sm]["vt_wait"] += f * float(cb.get("vt_departure_wait_applied", 0.0))
            metrics[g][t] = {}
            for sm, vals in agg.items():
                den = max(1e-9, vals["flow"])
                metrics[g][t][sm] = {
                    "flow": vals["flow"],
                    "avg_generalized_cost": vals["gen_cost"] / den,
                    "avg_travel_time": vals["travel_time"] / den,
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
    _apply_hub_power_scenario(data)

    times = [int(t) for t in data["sets"]["time"]]
    groups = [str(g) for g in data["sets"]["groups"]]
    itineraries = data["itineraries"]
    demand = data["parameters"]["q"]
    vot = data["parameters"]["vot"]
    lambdas = data["parameters"]["lambda"]
    ev_stations = [str(s) for s in data["sets"]["ev_stations"]]
    hybrid_stations = [str(s) for s in data["sets"]["hybrid_stations"]]

    travel_times: Dict[str, Dict[int, float]] = {}
    if data.get("parameters", {}).get("arc_params"):
        travel_times = compute_road_times(
            {arc: {t: 0.0 for t in times} for arc in data["parameters"]["arc_params"]},
            data["parameters"]["arc_params"],
            {t: 1.0 for t in times},
            times,
        )
    else:
        for it in itineraries:
            for seg in (it.get("road_arcs", []) or []) + (it.get("access_arcs", []) or []) + (it.get("egress_arcs", []) or []):
                arc = seg.get("arc")
                t = seg.get("t")
                if arc is None or t is None:
                    continue
                travel_times.setdefault(str(arc), {tt: 0.0 for tt in times})

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

        alpha = float(cfg.get("flow_msa_alpha") or (1.0 / itn))
        for it in itineraries:
            it_id = it["id"]
            for g in groups:
                for t in times:
                    flows[it_id][g][t] = (1.0 - alpha) * flows[it_id][g][t] + alpha * float(flows_target[it_id][g][t])

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
                ev_service_prob[s][t] = (1.0 - alpha) * ev_service_prob[s][t] + alpha * ev_prob

                vt_req = float(station_loads["E_vt_req"].get(s, {}).get(t, 0.0))
                vt_shed = float(shed_vt_out.get(s, {}).get(t, 0.0))
                vt_prob = 1.0 if vt_req <= 1e-9 else max(0.0, min(1.0, (vt_req - vt_shed) / vt_req))
                if s in vt_service_prob:
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
                "lp_solver": lp_diag.get("solver") if isinstance(lp_diag, dict) else "unknown",
                "unserved_demand_total": float(details.get("unserved_demand_total", 0.0)),
            }
        )
        if max_price_delta <= float(cfg["tol"]):
            break

    group_time_super = _compute_group_time_supermode_metrics(itineraries, flows, costs, groups, times)
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
