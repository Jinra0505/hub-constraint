import copy
import math
import json
import os
import sys
from pathlib import Path

if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    __package__ = "project.src"
from typing import Any, Dict, List, Tuple

from .assignment import (
    aggregate_arc_flows,
    aggregate_ev_energy_demand,
    aggregate_evtol_dep_demand,
    aggregate_evtol_demand,
    aggregate_station_utilization,
    aggregate_ev_station_utilization,
    aggregate_vt_departure_flow_by_class,
    get_evtol_service_class,
    is_evtol_itinerary,
    is_multimodal_evtol,
    classify_mode_label,
    classify_supermode,
    build_incidence,
    compute_evtol_energy_demand,
    compute_itinerary_costs,
    logit_assignment,
)
from . import charging_hiGHS_and_gurobi_bound_fix as charging
from .charging_hiGHS_and_gurobi_bound_fix import compute_station_loads_from_flows, solve_charging
from .congestion import compute_road_times, compute_station_waits, compute_vt_departure_waits
from .data_loader import load_data
from .dist_grid import solve_distribution_grid
from .itinerary_generator import generate_itineraries
from .mfd import boundary_flows, compute_g, update_accumulation






def _clean_number(x: float, eps: float = 1.0e-12) -> float:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return x
        if abs(x) < eps:
            return 0.0
    return float(x)


def _clean_nested_numbers(obj: Any, eps: float = 1.0e-12) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_nested_numbers(v, eps=eps) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nested_numbers(v, eps=eps) for v in obj]
    if isinstance(obj, float):
        return _clean_number(obj, eps=eps)
    return obj

def _safe_den(x: Any, eps: float = 1.0e-6) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return eps
    return val if val > 0.0 else eps




def _compute_cap_values(base_price: float, config: Dict[str, Any]) -> Tuple[float, float | None, float | None, str]:
    cap_mult = config.get("shadow_price_cap_mult")
    cap_abs = config.get("shadow_price_cap_abs")
    cap_mult_value = float(cap_mult) * float(base_price) if cap_mult is not None else None
    cap_abs_value = float(cap_abs) if cap_abs is not None else None
    if cap_mult_value is not None and cap_abs_value is not None:
        return min(cap_abs_value, cap_mult_value), cap_mult_value, cap_abs_value, "min(abs,mult)"
    if cap_abs_value is not None:
        return cap_abs_value, None, cap_abs_value, "abs"
    if cap_mult_value is not None:
        return cap_mult_value, cap_mult_value, None, "mult"
    return float("inf"), None, None, "none"



def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config aliases and set defaults with backward compatibility."""
    cfg = config
    cfg["tol"] = float(cfg.get("tol", 1.0))
    # historical alias mapping
    if "shadow_price_scale" not in cfg and "surcharge_kappa" in cfg:
        cfg["shadow_price_scale"] = cfg.get("surcharge_kappa")
    if "surcharge_kappa" not in cfg and "shadow_price_scale" in cfg:
        cfg["surcharge_kappa"] = cfg.get("shadow_price_scale")
    cfg.setdefault("surcharge_beta", 1.0)
    cfg.setdefault("surcharge_method", "shadow_lp")
    cfg.setdefault("surcharge_decay_raw_threshold", 1.0e-3)
    cfg.setdefault("surcharge_decay_factor", 0.6)
    cfg.setdefault("flow_msa_alpha", None)
    cfg.setdefault("reroute_logit_temperature", 1.25)
    cfg.setdefault("min_multimodal_reroute_share", 0.0)
    cfg.setdefault("shadow_price_scale", 1.0)
    cfg.setdefault("shadow_price_cap_mult", 10.0)
    cfg.setdefault("shadow_price_cap_abs", None)
    cfg.setdefault("vt_reliability_floor", 0.05)
    cfg.setdefault("min_vt_reroute_share", 0.0)
    cfg.setdefault("vt_reliability_alpha", None)
    cfg.setdefault("vt_reliability_skip_below", 0.0)
    cfg.setdefault("vt_reliability_gamma", 0.0)
    cfg.setdefault("ev_reliability_gamma", 0.0)
    cfg.setdefault("max_iter_auto_extend", 0)
    cfg.setdefault("max_total_iter", 800)
    cfg.setdefault("auto_extend_step", 100)
    cfg.setdefault("ev_reliability_floor", cfg.get("vt_reliability_floor", 0.05))
    cfg.setdefault("strict_audit", True)
    cfg.setdefault("audit_raise", True)
    cfg.setdefault("strict_convergence", False)
    cfg.setdefault("use_distribution_grid", False)
    cfg.setdefault("grid_shadow_price_scale", 1.0)
    cfg.setdefault("patience_require_strict_gate", True)
    cfg.setdefault("shared_power_solver", "highs")
    cfg.setdefault("output_full_json", True)
    cfg.setdefault("power_violation_mode", "net")
    cfg.setdefault("voll_ev_per_kwh", 50.0)
    cfg.setdefault("voll_vt_per_kwh", 200.0)
    cfg.setdefault("fail_on_infeasible_demand", False)
    cfg.setdefault("terminal_soc_policy", "at_least_init")
    cfg.setdefault("terminal_soc_target_kwh", None)
    cfg.setdefault("peak_t_selection_rule", "max_total_demand")
    cfg.setdefault("manual_peak_t", None)
    cfg.setdefault("case_label", "illustrative")
    return cfg




def _ignored_config_fields(config: Dict[str, Any]) -> List[str]:
    ignored: List[str] = []
    if not bool(config.get("use_generator", False)) and "K_paths" in config:
        ignored.append("K_paths (ignored when use_generator=false)")
    return ignored
def _select_peak_t(data: Dict[str, Any], times: List[int], config: Dict[str, Any]) -> Tuple[int, str, float]:
    """Select peak analysis period.

    Returns (peak_t, selection_rule_used, total_demand_at_peak_pax).
    """
    if not times:
        return 0, "none", 0.0
    q = data.get("parameters", {}).get("q", {})
    demand_by_t = {t: 0.0 for t in times}
    for od_map in q.values():
        if not isinstance(od_map, dict):
            continue
        for grp_map in od_map.values():
            if not isinstance(grp_map, dict):
                continue
            for t in times:
                demand_by_t[t] += float(grp_map.get(t, 0.0) or 0.0)

    rule = str(config.get("peak_t_selection_rule", "max_total_demand"))
    if rule == "last_period":
        peak_t = times[-1]
    elif rule == "manual":
        cand = config.get("manual_peak_t")
        peak_t = int(cand) if cand in times else times[-1]
    else:
        rule = "max_total_demand"
        peak_t = max(times, key=lambda t: demand_by_t.get(t, 0.0))
    return int(peak_t), rule, float(demand_by_t.get(peak_t, 0.0))


def _convergence_flags(dx: float, dn: float, dprice: float, residuals: Dict[str, float], tol: float, strict: bool) -> Dict[str, Any]:
    """Evaluate loose/strict convergence gates.

    Notes:
    - ``dprice`` must be the same price metric used by stop-gating logic.
    - In this implementation we use the raw (pre-MSA) surcharge delta for gating,
      exposed as ``dprice_gate_end`` in diagnostics/metrics.
    """
    strict_tol = max(1.0e-6, 0.5 * tol)
    loose_ok = max(dx, dn, dprice) <= tol
    strict_metric_ok = max(dx, dn, dprice) <= strict_tol
    strict_resid_ok = (residuals.get("C2_rel", 1e9) <= max(0.2, 0.4 * tol)) and (residuals.get("C10", 1e9) <= max(25.0, 10.0 * tol)) and (residuals.get("C7", 1e9) <= max(10.0, 8.0 * tol))
    converged_strictly = strict_metric_ok and strict_resid_ok
    return {
        "converged_loose": bool(loose_ok),
        "converged_strictly": bool(converged_strictly),
        "max_residual_end": float(max(residuals.get("C2_rel", 0.0), residuals.get("C10", 0.0), residuals.get("C7", 0.0))),
        "patience_only_stop": False,
        "strict_required": bool(strict),
    }
def self_audit(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    floor = float(config.get("vt_reliability_floor", 0.05))
    warnings: List[str] = []
    severe: List[str] = []
    cap_binding = False

    diagnostics = results.get("diagnostics", {})
    expected_groups = set(results.get("data_parameters", {}).get("expected_groups", []))
    if expected_groups:
        for entry in diagnostics.get("mode_share_by_group_and_supermode", []):
            shares = entry.get("shares", {})
            found = set(shares.keys())
            extra = sorted(found - expected_groups)
            if extra:
                severe.append(f"unexpected group label(s) in mode share: {extra}")
    power_tightness = diagnostics.get("power_tightness", {})

    def _scan_numeric(path: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _scan_numeric(f"{path}.{k}" if path else str(k), v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan_numeric(f"{path}[{i}]", v)
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                severe.append(f"nan/inf detected at diagnostics.{path}")

    _scan_numeric("", diagnostics)

    delta_t = float(results.get("meta", {}).get("delta_t", 1.0))
    eta_map = results.get("data_parameters", {}).get("vertiport_storage", {})

    for station, t_map in power_tightness.items():
        for t, entry in t_map.items():
            mu_entry = entry.get("mu_kw")
            mu_kw = float(mu_entry) if mu_entry is not None else 0.0
            local_mu_kw = float(entry.get("local_mu_kw", mu_kw) or 0.0)
            grid_mu_kw = float(entry.get("grid_mu_kw", 0.0) or 0.0)
            total_mu_kw = float(entry.get("mu_total_kw", local_mu_kw + grid_mu_kw) or 0.0)
            local_dual_available = bool(entry.get("local_dual_available", False))
            solver_entry = entry.get("solver_used")
            surcharge = float(entry.get("raw_surcharge", 0.0))
            uncapped_entry = entry.get("raw_surcharge_uncapped")
            uncapped = float(uncapped_entry) if uncapped_entry is not None else 0.0
            cap = float(entry.get("cap", 0.0))
            if mu_entry is not None and not (mu_kw >= 0.0 and mu_kw < float("inf")):
                severe.append(f"invalid mu_kw at {station},{t}: {mu_kw}")
            if not (surcharge >= 0.0 and surcharge < float("inf")):
                severe.append(f"invalid surcharge at {station},{t}: {surcharge}")
            if solver_entry != "highs" and mu_entry is not None:
                severe.append("mu_kw is not LP dual")
            if abs(surcharge - cap) <= 1.0e-9:
                cap_binding = True
            if uncapped_entry is not None and cap > 0.0 and uncapped > 100.0 * cap:
                warnings.append(f"uncapped surcharge very large at {station},{t}; report VOLL and capped surcharge")

            cap_used = float(entry.get("cap_used", entry.get("cap", 0.0)))
            cap_field = float(entry.get("cap", cap_used))
            if abs(cap_field - cap_used) > 1.0e-9:
                severe.append(f"cap/cap_used mismatch at {station},{t}")
            unc = entry.get("raw_surcharge_uncapped")
            bind_expected = False
            if unc is not None:
                bind_expected = float(unc) > cap_used - 1.0e-9
            else:
                bind_expected = float(entry.get("raw_surcharge", 0.0)) >= cap_used - 1.0e-9
            if bool(entry.get("cap_binding", False)) != bool(bind_expected):
                severe.append(f"cap_binding mismatch at {station},{t}")

            vt_prob = float(entry.get("vt_service_prob", 1.0))
            if vt_prob < floor - 1.0e-8 or vt_prob > 1.0 + 1.0e-8:
                severe.append(f"vt_service_prob out of range at {station},{t}: {vt_prob}")

            p_net_served = float(entry.get("p_net_served_kw", 0.0))
            p_site_raw = float(entry.get("P_site_raw", 0.0))
            if p_net_served - p_site_raw > 1.0e-6:
                severe.append(f"served power exceeds site cap at {station},{t}")

            e_vt_req = float(entry.get("E_vt_req_kwh", 0.0))
            shed_vt_kwh = float(entry.get("shed_vt_kwh", 0.0))
            p_vt_charge = float(entry.get("P_vt_charge_kw", 0.0))
            b_before = entry.get("B_before_kwh")
            b_after = entry.get("B_after_kwh")
            if b_before is None or b_after is None:
                severe.append(f"B_before/B_after missing at {station},{t}")
            else:
                eta = float(eta_map.get(station, {}).get("eta_ch", 1.0))
                lhs = float(b_after)
                rhs = float(b_before) + eta * p_vt_charge * delta_t - (e_vt_req - shed_vt_kwh)
                if abs(lhs - rhs) > 1.0e-6:
                    severe.append(f"inventory balance broken at {station},{t}: err={abs(lhs-rhs)}")

            ratio_req = float(entry.get("ratio_req_exogenous", 0.0))
            solver_used_local = str(entry.get("solver_used", diagnostics.get("shared_power_solver_used", "")))
            fallback_used = bool(entry.get("fallback_used", False))
            ratio_net_actual = float(entry.get("ratio_net_actual", 0.0))
            cap_binding_here = bool(entry.get("cap_binding", False))
            scarcity_expected = local_dual_available or grid_mu_kw > 1.0e-9 or cap_binding_here
            if (
                ratio_req > 2.0
                and ratio_net_actual > 0.98
                and abs(total_mu_kw) <= 1.0e-9
                and solver_used_local == "highs"
                and not fallback_used
                and scarcity_expected
            ):
                warnings.append(f"ratio_req high but mu_kw ~ 0 at {station},{t}")

    n_series = diagnostics.get("n_series", {})
    n_values = n_series.values() if isinstance(n_series, dict) else n_series
    has_negative_n = any(float(v) < -1.0e-9 for v in n_values)
    if has_negative_n:
        severe.append("negative value detected in n_series")

    solver_used = diagnostics.get("shared_power_solver_used")
    requested_solver = str(config.get("shared_power_solver", "highs")).lower()
    if solver_used != "highs":
        warnings.append(f"shared_power solver is {solver_used}, not highs; dual interpretation may differ")
    if requested_solver == "highs" and solver_used != "highs":
        severe.append("Requested HiGHS but non-HiGHS solver used")
    if bool(diagnostics.get("missing_inventory_reporting", False)):
        severe.append("missing inventory reporting")
    if diagnostics.get("lp_failure") is not None:
        severe.append("LP failed")
    if any(entry.get("fallback_used") for t_map in power_tightness.values() for entry in t_map.values()):
        severe.append("LP failed and fallback path used")
    if cap_binding:
        warnings.append("price cap binding detected; report capped and uncapped surcharge in analysis")

    stop_reason = diagnostics.get("stop_reason")
    tol = float(diagnostics.get("tol", config.get("tol", 0.0)) or 0.0)
    dx_end = diagnostics.get("dx_end", results.get("convergence", {}).get("dx"))
    dn_end = diagnostics.get("dn_end", results.get("convergence", {}).get("dn"))
    dprice_end = diagnostics.get("dprice_gate_end", results.get("convergence", {}).get("dprice_gate"))
    converged_loose = bool(diagnostics.get("converged_loose", False))
    converged_strictly = bool(diagnostics.get("converged_strictly", False))
    if stop_reason != "patience":
        warnings.append("did not stop by patience")
    if tol > 0.0 and not converged_loose:
        warnings.append("loose convergence thresholds not satisfied")
    if bool(config.get("strict_convergence", False)) and not converged_strictly:
        if bool(config.get("strict_convergence_raise", False)):
            severe.append("strict convergence thresholds not satisfied")
        else:
            warnings.append("strict convergence thresholds not satisfied")
    elif not converged_strictly:
        warnings.append("strict convergence thresholds not satisfied")

    expected_groups = set(results.get("data_parameters", {}).get("lambda", {}).keys())
    output_groups = set()
    for entry in diagnostics.get("mode_share_by_group", []):
        output_groups.update((entry or {}).get("shares", {}).keys())
    flow_map = results.get("f", {})
    for g_map in flow_map.values():
        output_groups.update(g_map.keys())
    if expected_groups and any(g not in expected_groups for g in output_groups):
        severe.append("unexpected traveler group label in outputs")

    util_def = str(diagnostics.get("station_utilization_definition", ""))
    util_builder = str(diagnostics.get("station_utilization_builder", ""))
    if util_builder != "aggregate_ev_station_utilization":
        severe.append("wrong station utilization builder used for C10 / EV-related utilization")
    if util_def != "EV_related_only":
        warnings.append("station utilization definition is ambiguous; ensure C10 uses EV-only utilization")

    unserved_total = float(diagnostics.get("unserved_demand_total", 0.0) or 0.0)
    if unserved_total > 1.0e-9:
        msg = "unserved demand exists because no feasible alternative remained"
        if bool(config.get("fail_on_infeasible_demand", False)):
            severe.append(msg)
        else:
            warnings.append(msg)

    mode_group_mode = diagnostics.get("mode_share_by_group_and_mode", [])
    mode_group_old = diagnostics.get("mode_share_by_group", [])
    if mode_group_mode and mode_group_old:
        last_new = mode_group_mode[-1].get("shares", {})
        last_old = mode_group_old[-1].get("shares", {})
        for grp, comps in last_new.items():
            mm = float(comps.get("multimodal_EV_to_eVTOL_fast", 0.0)) + float(comps.get("multimodal_EV_to_eVTOL_slow", 0.0))
            old_grp = last_old.get(grp, {})
            if "ev" in old_grp or "vt" in old_grp:
                warnings.append("legacy mode_share_by_group is deprecated; use mode_share_by_group_and_supermode")
                old_vt = float(old_grp.get("vt", 0.0))
                pure_vt = float(comps.get("pure_eVTOL_fast", 0.0)) + float(comps.get("pure_eVTOL_slow", 0.0))
                if mm > 1.0e-9 and old_vt > pure_vt + 1.0e-9:
                    severe.append("multimodal itineraries were merged into pure eVTOL shares")
                    break
            else:
                old_mm = float(old_grp.get("EV_to_eVTOL", 0.0))
                if abs(old_mm - mm) > 1.0e-6:
                    severe.append("multimodal itineraries were merged into pure eVTOL shares")
                    break

    cap_bad = float(diagnostics.get("vertiport_cap_feasible_vt_but_all_ev_count", 0.0) or 0.0)
    cap_trig = float(diagnostics.get("vertiport_cap_triggered_count", 0.0) or 0.0)
    if cap_trig >= 3.0 and cap_bad / max(cap_trig, 1.0) > 0.5:
        warnings.append("vertiport cap rerouting ignored feasible VT-related alternatives")

    air_bad = float(diagnostics.get("aircraft_feasible_vt_but_all_ev_count", 0.0) or 0.0)
    air_trig = float(diagnostics.get("aircraft_binding_count", 0.0) or 0.0)
    if air_trig >= 3.0 and air_bad / max(air_trig, 1.0) > 0.5:
        warnings.append("aircraft rerouting ignored feasible VT-related alternatives")

    if float(diagnostics.get("aircraft_lag_oob_count", 0.0) or 0.0) > 0.0:
        severe.append("aircraft return lag index out of bounds")

    inv = diagnostics.get("aircraft_inventory_by_station", {})
    dep = diagnostics.get("aircraft_departures_by_station", {})
    for s, tmap in inv.items():
        for t, a in tmap.items():
            a = float(a)
            d = float(dep.get(s, {}).get(t, 0.0))
            if a < -1.0e-9:
                severe.append("aircraft inventory negative")
            if d > a + 1.0e-9:
                severe.append("aircraft departures exceed available inventory")

    vt_waits = diagnostics.get("vt_departure_waits", {})
    fast_gt_slow = 0
    fast_cmp_total = 0
    for s, cls_map in vt_waits.items():
        for t, wf in cls_map.get("fast", {}).items():
            ws = cls_map.get("slow", {}).get(t)
            if ws is None:
                continue
            wf = float(wf)
            ws = float(ws)
            if math.isnan(wf) or math.isinf(wf) or math.isnan(ws) or math.isinf(ws):
                severe.append("vt departure waits contain NaN/Inf")
                continue
            fast_cmp_total += 1
            if wf > ws + 1.0e-9:
                fast_gt_slow += 1
    if fast_cmp_total > 0 and fast_gt_slow / fast_cmp_total > 0.6:
        warnings.append("fast departure waits are frequently above slow waits; check queue/capacity parameters")

    c2 = float(results.get("residuals", {}).get("C2", 0.0))
    c2_raw = float(results.get("residuals", {}).get("C2_raw", 0.0))
    if c2_raw <= 1.0e-9 and c2 > 1.0e-3:
        warnings.append("C2 is smoothed mismatch; check C2_rel and dx_end for convergence")

    warnings = sorted(set(warnings))
    severe = sorted(set(severe))
    warnings = sorted(set(warnings + severe))
    audit = {
        "ok": len(severe) == 0,
        "warnings": warnings,
        "severe": severe,
        "cap_binding": cap_binding,
        "has_negative_n": has_negative_n,
    }
    if bool(config.get("audit_raise", False)) and severe:
        raise ValueError("SelfAudit severe issues: " + "; ".join(severe))
    return audit

def _time_series_from_any(raw: Any, times: List[int], default: float = 0.0) -> List[float]:
    if raw is None:
        return [float(default) for _ in times]
    if isinstance(raw, (int, float)):
        return [float(raw) for _ in times]
    if isinstance(raw, list):
        if len(raw) == len(times):
            return [float(v) for v in raw]
        if len(raw) == 1:
            return [float(raw[0]) for _ in times]
        raise ValueError(f"Background series list length {len(raw)} does not match time horizon {len(times)}")
    if isinstance(raw, dict):
        out: List[float] = []
        for t in times:
            if t in raw:
                out.append(float(raw[t]))
            elif str(t) in raw:
                out.append(float(raw[str(t)]))
            else:
                out.append(float(default))
        return out
    raise ValueError("Background boundary series must be scalar/list/dict or null")


def _boundary_flows_with_background(
    arc_flows: Dict[str, Dict[int, float]],
    boundary_in: List[str],
    boundary_out: List[str],
    times: List[int],
    background_in: Any = None,
    background_out: Any = None,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    model_in, model_out = boundary_flows(arc_flows, boundary_in, boundary_out, times)
    bg_in = _time_series_from_any(background_in, times, 0.0)
    bg_out = _time_series_from_any(background_out, times, 0.0)
    total_in = [float(model_in[i]) + float(bg_in[i]) for i in range(len(times))]
    total_out = [float(model_out[i]) + float(bg_out[i]) for i in range(len(times))]
    return total_in, total_out, model_in, model_out, bg_in, bg_out


def _fill_missing(mapping: Dict[str, Dict[int, float]], keys: List[str], times: List[int]) -> Dict[str, Dict[int, float]]:
    for key in keys:
        mapping.setdefault(key, {t: 0.0 for t in times})
        for t in times:
            mapping[key].setdefault(t, 0.0)
    return mapping


def _build_itinerary_index(itineraries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {it["id"]: it for it in itineraries}


def _utility_for_alt(
    utilities: Dict[str, Dict[str, Dict[int, float]]],
    it_id: str,
    group: str,
    t: int,
) -> float:
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
    min_vt_reroute_share: float = 0.0,
    min_multimodal_reroute_share: float = 0.0,
    reroute_logit_temperature: float = 1.0,
) -> Dict[str, float]:
    if excess_pax <= 0.0:
        return {"to_evtol": 0.0, "to_multimodal": 0.0, "to_ev": 0.0, "unserved": 0.0, "feasible_vt_alts": 0.0, "feasible_total_alts": 0.0, "feasible_multimodal_alts": 0.0}

    alts = itineraries_by_od.get(od_key, [])
    feasible: List[Tuple[Dict[str, Any], float]] = []
    feasible_vt_alts = 0.0
    feasible_multimodal_alts = 0.0
    for it in alts:
        dep = it.get("dep_station")
        if blocked_dep_station is not None and is_evtol_itinerary(it) and dep == blocked_dep_station:
            continue
        util = _utility_for_alt(utilities, it["id"], group, t)
        if math.isinf(util) and util < 0:
            continue
        if math.isnan(util):
            continue
        feasible.append((it, util))
        if is_evtol_itinerary(it):
            feasible_vt_alts += 1.0
        if is_multimodal_evtol(it):
            feasible_multimodal_alts += 1.0

    if not feasible:
        unserved_demand.setdefault(od_key, {}).setdefault(group, {})[t] = unserved_demand.setdefault(od_key, {}).setdefault(group, {}).get(t, 0.0) + excess_pax
        return {"to_evtol": 0.0, "to_multimodal": 0.0, "to_ev": 0.0, "unserved": excess_pax, "feasible_vt_alts": feasible_vt_alts, "feasible_total_alts": 0.0, "feasible_multimodal_alts": 0.0}

    temp = max(1.0e-6, float(reroute_logit_temperature))
    max_u = max(u / temp for _, u in feasible)
    weights = [math.exp((u / temp) - max_u) for _, u in feasible]
    den = sum(weights)
    if den <= 0.0:
        unserved_demand.setdefault(od_key, {}).setdefault(group, {})[t] = unserved_demand.setdefault(od_key, {}).setdefault(group, {}).get(t, 0.0) + excess_pax
        return {"to_evtol": 0.0, "to_multimodal": 0.0, "to_ev": 0.0, "unserved": excess_pax, "feasible_vt_alts": feasible_vt_alts, "feasible_total_alts": 0.0, "feasible_multimodal_alts": 0.0}

    probs = [w / den for w in weights]
    vt_idx = [i for i, (it, _) in enumerate(feasible) if is_evtol_itinerary(it)]
    mm_idx = [i for i, (it, _) in enumerate(feasible) if is_multimodal_evtol(it)]
    ev_idx = [i for i, (it, _) in enumerate(feasible) if not is_evtol_itinerary(it)]

    def _enforce_min_share(recipient_idx: List[int], target: float, donor_idx: List[int]) -> None:
        if not recipient_idx or target <= 0.0:
            return
        target = min(max(float(target), 0.0), 1.0)
        current = sum(probs[i] for i in recipient_idx)
        if current >= target - 1.0e-12:
            return
        donor_mass = sum(probs[i] for i in donor_idx)
        if donor_mass <= 1.0e-12:
            return
        transfer = min(target - current, donor_mass)
        recip_mass = sum(probs[i] for i in recipient_idx)
        if recip_mass > 1.0e-12:
            for i in recipient_idx:
                probs[i] += transfer * (probs[i] / recip_mass)
        else:
            add = transfer / max(1, len(recipient_idx))
            for i in recipient_idx:
                probs[i] += add
        for i in donor_idx:
            probs[i] -= transfer * (probs[i] / donor_mass)

    if vt_idx and min_vt_reroute_share > 0.0 and ev_idx:
        _enforce_min_share(vt_idx, float(min_vt_reroute_share), ev_idx)

    if mm_idx and min_multimodal_reroute_share > 0.0:
        donor_idx = [i for i in range(len(probs)) if i not in mm_idx]
        _enforce_min_share(mm_idx, float(min_multimodal_reroute_share), donor_idx)

    probs = [max(0.0, p) for p in probs]
    norm = sum(probs)
    probs = [p / max(norm, 1.0e-12) for p in probs]

    to_evtol = 0.0
    to_multimodal = 0.0
    to_ev = 0.0
    for (it, _), p in zip(feasible, probs):
        add = excess_pax * p
        flows[it["id"]].setdefault(group, {})[t] = flows[it["id"]].get(group, {}).get(t, 0.0) + add
        if is_multimodal_evtol(it):
            to_multimodal += add
        elif is_evtol_itinerary(it):
            to_evtol += add
        else:
            to_ev += add
    return {"to_evtol": to_evtol, "to_multimodal": to_multimodal, "to_ev": to_ev, "unserved": 0.0, "feasible_vt_alts": feasible_vt_alts, "feasible_total_alts": float(len(feasible)), "feasible_multimodal_alts": feasible_multimodal_alts}


def _apply_vertiport_caps(
    flows: Dict[str, Dict[str, Dict[int, float]]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
    cap_pax: Dict[str, Dict[int, float]],
    utilities: Dict[str, Dict[str, Dict[int, float]]],
    unserved_demand: Dict[str, Dict[str, Dict[int, float]]],
    min_vt_reroute_share: float = 0.0,
    min_multimodal_reroute_share: float = 0.0,
    reroute_logit_temperature: float = 1.0,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, float]]:
    itineraries_by_od: Dict[str, List[Dict[str, Any]]] = {}
    evtol_by_dep: Dict[str, List[Dict[str, Any]]] = {}

    for it in itineraries:
        od_key = f"{it['od'][0]}-{it['od'][1]}"
        itineraries_by_od.setdefault(od_key, []).append(it)
        if is_evtol_itinerary(it) and it.get("dep_station") is not None:
            evtol_by_dep.setdefault(it["dep_station"], []).append(it)

    stats = {
        "triggered_count": 0.0,
        "excess_pax": 0.0,
        "rerouted_to_evtol": 0.0,
        "rerouted_to_multimodal": 0.0,
        "rerouted_to_ev": 0.0,
        "unserved": 0.0,
        "feasible_vt_but_all_ev_count": 0.0,
        "feasible_total_alts_count": 0.0,
        "feasible_multimodal_alts_count": 0.0,
    }

    for dep, it_list in evtol_by_dep.items():
        for t in times:
            total = 0.0
            for it in it_list:
                for _, time_map in flows[it["id"]].items():
                    total += float(time_map.get(t, 0.0))
            cap = float(cap_pax.get(dep, {}).get(t, float("inf")))
            if total <= cap + 1.0e-12 or total <= 0.0:
                continue
            stats["triggered_count"] += 1.0
            ratio = cap / max(total, 1.0e-12)
            reductions: Dict[Tuple[str, str], float] = {}
            excess = 0.0
            for it in it_list:
                od_key = f"{it['od'][0]}-{it['od'][1]}"
                for group, time_map in flows[it["id"]].items():
                    original = float(time_map.get(t, 0.0))
                    reduced = original * ratio
                    delta = original - reduced
                    if delta <= 0.0:
                        continue
                    flows[it["id"]].setdefault(group, {})[t] = reduced
                    reductions[(od_key, group)] = reductions.get((od_key, group), 0.0) + delta
                    excess += delta
            stats["excess_pax"] += excess

            for (od_key, group), delta in reductions.items():
                rr = _reroute_excess_with_conditional_logit(
                    flows=flows,
                    itineraries_by_od=itineraries_by_od,
                    utilities=utilities,
                    od_key=od_key,
                    group=group,
                    t=t,
                    excess_pax=delta,
                    blocked_dep_station=dep,
                    unserved_demand=unserved_demand,
                    min_vt_reroute_share=min_vt_reroute_share,
                    min_multimodal_reroute_share=min_multimodal_reroute_share,
                    reroute_logit_temperature=reroute_logit_temperature,
                )
                stats["rerouted_to_evtol"] += rr["to_evtol"]
                stats["rerouted_to_multimodal"] += rr["to_multimodal"]
                stats["rerouted_to_ev"] += rr["to_ev"]
                stats["unserved"] += rr["unserved"]
                stats["feasible_total_alts_count"] += rr.get("feasible_total_alts", 0.0)
                stats["feasible_multimodal_alts_count"] += rr.get("feasible_multimodal_alts", 0.0)
                moved = rr["to_evtol"] + rr["to_multimodal"] + rr["to_ev"]
                ev_ratio = rr["to_ev"] / max(moved, 1.0e-12)
                if rr["feasible_vt_alts"] > 0.0 and ev_ratio >= 0.98 and moved > 1.0e-9:
                    stats["feasible_vt_but_all_ev_count"] += 1.0

    return flows, stats


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
    min_vt_reroute_share: float = 0.0,
    min_multimodal_reroute_share: float = 0.0,
    reroute_logit_temperature: float = 1.0,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Any]]:
    stations = sorted({
        st
        for it in itineraries
        if is_evtol_itinerary(it)
        for st in [it.get("dep_station"), it.get("arr_station")]
        if st is not None
    })
    itineraries_by_od: Dict[str, List[Dict[str, Any]]] = {}
    vt_by_dep: Dict[str, List[Dict[str, Any]]] = {s: [] for s in stations}
    for it in itineraries:
        od_key = f"{it['od'][0]}-{it['od'][1]}"
        itineraries_by_od.setdefault(od_key, []).append(it)
        dep = it.get("dep_station")
        if is_evtol_itinerary(it) and dep in vt_by_dep:
            vt_by_dep[dep].append(it)

    max_time = max(times) if times else 0
    max_lag = int(vt_turnaround_lag) + 3
    ext_times = list(range(min(times), max_time + max_lag + 2)) if times else []
    ret_sched: Dict[str, Dict[int, float]] = {s: {tt: 0.0 for tt in ext_times} for s in stations}
    inventory: Dict[str, Dict[int, float]] = {s: {} for s in stations}
    departures: Dict[str, Dict[int, float]] = {s: {t: 0.0 for t in times} for s in stations}
    returns: Dict[str, Dict[int, float]] = {s: {t: 0.0 for t in times} for s in stations}
    avail_now: Dict[str, float] = {s: float(vt_aircraft_init_by_station.get(s, 0.0)) for s in stations}

    stats = {
        "binding_count": 0.0,
        "excess_rerouted_to_evtol": 0.0,
        "excess_rerouted_to_multimodal": 0.0,
        "excess_rerouted_to_ev": 0.0,
        "unserved": 0.0,
        "feasible_vt_but_all_ev_count": 0.0,
        "lag_oob_count": 0.0,
        "feasible_total_alts_count": 0.0,
        "feasible_multimodal_alts_count": 0.0,
    }

    for t in times:
        reductions: Dict[Tuple[str, str, str], float] = {}
        for s in stations:
            arrivals = float(ret_sched.get(s, {}).get(t, 0.0))
            returns[s][t] = arrivals
            avail_start = max(0.0, avail_now[s] + arrivals)
            inventory[s][t] = avail_start

            req_dep = 0.0
            for it in vt_by_dep.get(s, []):
                cls = get_evtol_service_class(it)
                pax_per_dep = vt_pax_per_departure_fast if cls == "fast" else vt_pax_per_departure_slow
                pax_per_dep = max(1.0e-6, float(pax_per_dep))
                for _, time_map in flows.get(it["id"], {}).items():
                    req_dep += float(time_map.get(t, 0.0)) / pax_per_dep

            served_ratio = 1.0
            if req_dep > avail_start + 1.0e-12:
                stats["binding_count"] += 1.0
                served_ratio = avail_start / max(req_dep, 1.0e-12)

            served_dep = 0.0
            for it in vt_by_dep.get(s, []):
                od_key = f"{it['od'][0]}-{it['od'][1]}"
                cls = get_evtol_service_class(it)
                pax_per_dep = vt_pax_per_departure_fast if cls == "fast" else vt_pax_per_departure_slow
                pax_per_dep = max(1.0e-6, float(pax_per_dep))
                for group, time_map in flows[it["id"]].items():
                    original = float(time_map.get(t, 0.0))
                    served = original * served_ratio
                    delta = original - served
                    if delta > 0.0:
                        reductions[(od_key, group, s)] = reductions.get((od_key, group, s), 0.0) + delta
                    flows[it["id"]].setdefault(group, {})[t] = served
                    flights = served / pax_per_dep
                    served_dep += flights

                    arr_station = it.get("arr_station")
                    ft_raw = it.get("flight_time", 0.0)
                    if isinstance(ft_raw, dict):
                        flight_time = float(ft_raw.get(t, ft_raw.get(str(t), 0.0)))
                    else:
                        flight_time = float(ft_raw)
                    lag = int(math.ceil(max(0.0, flight_time) / max(delta_t, 1.0e-9))) + int(vt_turnaround_lag)
                    arr_t = t + lag
                    if arr_station in ret_sched:
                        ret_sched[arr_station][arr_t] = ret_sched[arr_station].get(arr_t, 0.0) + flights
                    else:
                        stats["lag_oob_count"] += 1.0

            departures[s][t] = served_dep
            avail_now[s] = max(0.0, avail_start - served_dep)

        for (od_key, group, blocked_dep), delta in reductions.items():
            rr = _reroute_excess_with_conditional_logit(
                flows=flows,
                itineraries_by_od=itineraries_by_od,
                utilities=utilities,
                od_key=od_key,
                group=group,
                t=t,
                excess_pax=delta,
                blocked_dep_station=blocked_dep,
                unserved_demand=unserved_demand,
                min_vt_reroute_share=min_vt_reroute_share,
                min_multimodal_reroute_share=min_multimodal_reroute_share,
                reroute_logit_temperature=reroute_logit_temperature,
            )
            stats["excess_rerouted_to_evtol"] += rr["to_evtol"]
            stats["excess_rerouted_to_multimodal"] += rr["to_multimodal"]
            stats["excess_rerouted_to_ev"] += rr["to_ev"]
            stats["unserved"] += rr["unserved"]
            stats["feasible_total_alts_count"] += rr.get("feasible_total_alts", 0.0)
            stats["feasible_multimodal_alts_count"] += rr.get("feasible_multimodal_alts", 0.0)
            moved = rr["to_evtol"] + rr["to_multimodal"] + rr["to_ev"]
            ev_ratio = rr["to_ev"] / max(moved, 1.0e-12)
            if rr["feasible_vt_alts"] > 0.0 and ev_ratio >= 0.98 and moved > 1.0e-9:
                stats["feasible_vt_but_all_ev_count"] += 1.0

    return flows, {
        "aircraft_inventory_by_station": inventory,
        "aircraft_departures_by_station": departures,
        "aircraft_returns_by_station": returns,
        "aircraft_binding_count": stats["binding_count"],
        "aircraft_excess_rerouted_to_evtol": stats["excess_rerouted_to_evtol"],
        "aircraft_excess_rerouted_to_multimodal": stats["excess_rerouted_to_multimodal"],
        "aircraft_excess_rerouted_to_ev": stats["excess_rerouted_to_ev"],
        "aircraft_unserved": stats["unserved"],
        "aircraft_feasible_vt_but_all_ev_count": stats["feasible_vt_but_all_ev_count"],
        "aircraft_lag_oob_count": stats["lag_oob_count"],
        "aircraft_feasible_total_alts_count": stats["feasible_total_alts_count"],
        "aircraft_feasible_multimodal_alts_count": stats["feasible_multimodal_alts_count"],
    }


def compute_residuals(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    arc_flows: Dict[str, Dict[int, float]],
    arc_flows_raw: Dict[str, Dict[int, float]] | None,
    tau: Dict[str, Dict[int, float]],
    n_series: List[float],
    g_series: List[float],
    inc_road: Dict[str, Dict[str, Dict[int, float]]],
    inc_station: Dict[str, Dict[str, Dict[int, float]]],
    utilization: Dict[str, Dict[int, float]],
    utilization_raw: Dict[str, Dict[int, float]] | None,
) -> Dict[str, float]:
    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    demand = data["parameters"]["q"]
    arc_params = data["parameters"]["arcs"]
    boundary_in = data["parameters"]["boundary_in"]
    boundary_out = data["parameters"]["boundary_out"]
    mfd_params = data["parameters"]["mfd"]
    delta_t = data["meta"]["delta_t"]

    residuals = {
        "C1": 0.0,
        "C2": 0.0,
        "C2_raw": 0.0,
        "C7": 0.0,
        "C8": 0.0,
        "C9": 0.0,
        "C11": 0.0,
        "C10": 0.0,
        "C10_raw": 0.0,
        "C2_rel": 0.0,
    }

    for od_key, groups in demand.items():
        for group, time_map in groups.items():
            for t, q_val in time_map.items():
                total = 0.0
                for it in itineraries:
                    if f"{it['od'][0]}-{it['od'][1]}" != od_key:
                        continue
                    total += flows[it["id"]].get(group, {}).get(t, 0.0)
                residuals["C1"] = max(residuals["C1"], abs(total - q_val))
                residuals["C11"] = max(residuals["C11"], abs(total - q_val))

    for arc in arcs:
        for t in times:
            expected = 0.0
            for it in itineraries:
                for group, time_map in flows[it["id"]].items():
                    expected += inc_road[arc][it["id"]][t] * time_map.get(t, 0.0)
            residuals["C2"] = max(residuals["C2"], abs(arc_flows[arc][t] - expected))
            if arc_flows_raw is not None:
                residuals["C2_raw"] = max(residuals["C2_raw"], abs(arc_flows_raw[arc][t] - expected))

    inflow_total, outflow_total, model_inflow_total, model_outflow_total, bg_inflow_total, bg_outflow_total = _boundary_flows_with_background(
        arc_flows,
        boundary_in,
        boundary_out,
        times,
        data.get("parameters", {}).get("boundary_in_background"),
        data.get("parameters", {}).get("boundary_out_background"),
    )
    inflow = [val / delta_t for val in inflow_total]
    outflow = [val / delta_t for val in outflow_total]
    for idx in range(len(times)):
        expected_n_next = max(0.0, n_series[idx] + delta_t * (inflow[idx] - outflow[idx]))
        residuals["C7"] = max(
            residuals["C7"],
            abs(n_series[idx + 1] - expected_n_next),
        )

    expected_g = compute_g(n_series, mfd_params)
    for idx, t in enumerate(times):
        residuals["C8"] = max(residuals["C8"], abs(g_series[idx] - expected_g[idx]))
        for arc, params in arc_params.items():
            if params["type"] == "CBD":
                expected_tau = params["tau0"] * g_series[idx] * params.get("theta", 1.0)
                residuals["C9"] = max(residuals["C9"], abs(tau[arc][t] - expected_tau))

    for station in utilization:
        for t in times:
            # C10 checks EV-related station utilization consistency only (via EV-only inc_station).
            expected_u = 0.0
            for it in itineraries:
                for group, time_map in flows[it["id"]].items():
                    expected_u += inc_station.get(station, {}).get(it["id"], {}).get(t, 0.0) * time_map.get(t, 0.0)
            residuals["C10"] = max(
                residuals["C10"],
                abs(utilization[station][t] - expected_u),
            )
            if utilization_raw is not None:
                residuals["C10_raw"] = max(
                    residuals["C10_raw"],
                    abs(utilization_raw[station][t] - expected_u),
                )

    max_flow = max(max(abs(arc_flows[a][t]), 1.0) for a in arcs for t in times)
    residuals["C2_rel"] = residuals["C2"] / _safe_den(max_flow, 1.0)
    return residuals


def run_equilibrium(data: Dict[str, Any], overrides: Dict[str, Any] | None = None, run_dispatch: bool = False) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if overrides:
        data = copy.deepcopy(data)
        for key, value in overrides.items():
            data[key] = value

    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    stations = data["sets"]["stations"]
    groups = list(data.get("sets", {}).get("groups", []))
    ev_stations = list(data.get("sets", {}).get("ev_stations", stations))
    hybrid_stations = list(data.get("sets", {}).get("hybrid_stations", stations))
    power_stations = hybrid_stations
    delta_t = data["meta"]["delta_t"]
    config = _normalize_config(data["config"])
    itineraries = list(data["itineraries"])
    seen_ids = {it["id"] for it in itineraries}
    arc_params = data["parameters"]["arcs"]
    station_params = data["parameters"]["stations"]
    electricity_price = data["parameters"]["electricity_price"]
    vt_departure_allowed = {
        str(k): bool(v) for k, v in data.get("parameters", {}).get("vt_departure_allowed", {}).items()
    }
    vt_arrival_allowed = {
        str(k): bool(v) for k, v in data.get("parameters", {}).get("vt_arrival_allowed", {}).items()
    }

    arc_flows = {a: {t: 0.0 for t in times} for a in arcs}
    utilization = {s: {t: 0.0 for t in times} for s in ev_stations}
    n_series = update_accumulation(
        data["parameters"]["n0"],
        [0.0 for _ in times],
        [0.0 for _ in times],
        delta_t,
    )
    g_series = compute_g(n_series, data["parameters"]["mfd"])
    energy_surcharge = {s: {t: 0.0 for t in times} for s in ev_stations}
    vt_service_prob = {s: {t: 1.0 for t in times} for s in power_stations}
    ev_service_prob = {s: {t: 1.0 for t in times} for s in ev_stations}
    min_iters = int(config.get("min_iters", 10))
    patience = int(config.get("patience", 5))
    stop_reason = "max_iters"
    dprice = 0.0
    peak_t, peak_t_rule, peak_t_total_demand = _select_peak_t(data, times, config)
    demand_keys = list(data["parameters"]["q"].keys())
    representative_od = config.get("representative_od")
    if not representative_od:
        representative_od = "A-B" if "A-B" in demand_keys else (demand_keys[0] if demand_keys else "")
    access_overlap_itins = 0
    access_inconsistent_itins = 0
    for it in itineraries:
        access_stops = it.get("access_stations", []) or []
        scalar = it.get("access_energy_kwh", 0.0)
        for t in times:
            explicit = sum(float(st.get("energy", 0.0) or 0.0) for st in access_stops if st.get("t") == t)
            scalar_t = float(scalar.get(t, scalar.get(str(t), 0.0)) if isinstance(scalar, dict) else (scalar or 0.0))
            if explicit > 1.0e-12 and scalar_t > 1.0e-12:
                access_overlap_itins += 1
                rel = abs(explicit - scalar_t) / max(1.0e-6, max(explicit, scalar_t))
                if rel > 0.15:
                    access_inconsistent_itins += 1
                break
    diagnostics = {
        "dx_history": [],
        "dn_history": [],
        "dprice_history": [],
        "dprice_raw_history": [],
        "max_surcharge_history": [],
        "price_snapshots": [],
        "mode_share": [],
        "mode_share_by_supermode": [],
        "surcharge_history": [],
        "mode_share_by_group": [],
        "mode_share_by_group_and_mode": [],
        "mode_share_by_group_and_supermode": [],
        "mode_share_by_service_class": [],
        "pure_evtol_service_class_share": [],
        "all_evtol_service_class_share": [],
        "vt_departure_waits": {},
        "vt_departure_flow_by_class": {},
        "power_tightness_summary_history": [],
        "power_tightness": {},
        "station_utilization_definition": "EV_related_only",
        "station_utilization_builder": "aggregate_ev_station_utilization",
        "shared_power_solver_used": "unknown",
        "module_paths": {
            "runner_file": str(Path(__file__).resolve()),
            "charging_file": str(Path(charging.__file__).resolve()),
            "sys_path_head": list(sys.path[:5]),
        },
        "station_sets": {
            "ev_stations": list(ev_stations),
            "hybrid_stations": list(hybrid_stations),
            "vt_departure_allowed": vt_departure_allowed,
            "vt_arrival_allowed": vt_arrival_allowed,
        },
        "peak_t_selection_rule": peak_t_rule,
        "peak_t_total_demand": peak_t_total_demand,
        "representative_od": representative_od,
        "case_label": str(config.get("case_label", "illustrative")),
        "ignored_config_fields": _ignored_config_fields(config),
        "parameter_effective": {
            "min_vt_reroute_share": config.get("min_vt_reroute_share", 0.0),
            "surcharge_kappa": config.get("surcharge_kappa"),
            "surcharge_beta": config.get("surcharge_beta"),
            "K_paths": config.get("K_paths"),
            "eps_improve": config.get("eps_improve"),
            "reroute_logit_temperature": config.get("reroute_logit_temperature"),
            "min_multimodal_reroute_share": config.get("min_multimodal_reroute_share"),
            "patience_require_strict_gate": config.get("patience_require_strict_gate"),
            "shadow_price_scale": config.get("shadow_price_scale"),
            "shadow_price_cap_mult": config.get("shadow_price_cap_mult"),
            "shadow_price_cap_abs": config.get("shadow_price_cap_abs"),
            "tol": config.get("tol"),
            "strict_convergence": config.get("strict_convergence"),
            "strict_audit": config.get("strict_audit"),
            "audit_raise": config.get("audit_raise"),
            "terminal_soc_policy": config.get("terminal_soc_policy"),
            "terminal_soc_target_kwh": config.get("terminal_soc_target_kwh"),
        },
        "scipy_version": charging.SCIPY_VERSION,
        "scipy_ok": bool(charging.HAS_SCIPY),
        "lp_ok": None,
        "vertiport_cap_triggered_count": 0.0,
        "vertiport_cap_excess_pax": 0.0,
        "vertiport_cap_rerouted_to_evtol": 0.0,
        "vertiport_cap_rerouted_to_multimodal": 0.0,
        "vertiport_cap_rerouted_to_ev": 0.0,
        "vertiport_cap_unserved": 0.0,
        "vertiport_cap_feasible_vt_but_all_ev_count": 0.0,
        "aircraft_inventory_by_station": {},
        "aircraft_departures_by_station": {},
        "aircraft_returns_by_station": {},
        "aircraft_binding_count": 0.0,
        "aircraft_excess_rerouted_to_evtol": 0.0,
        "aircraft_excess_rerouted_to_multimodal": 0.0,
        "aircraft_excess_rerouted_to_ev": 0.0,
        "aircraft_unserved": 0.0,
        "aircraft_feasible_vt_but_all_ev_count": 0.0,
        "aircraft_lag_oob_count": 0.0,
        "power_binding_count": 0,
        "terminal_soc_by_station": {},
        "storage_energy_totals_by_station": {},
        "access_energy_overlap_itineraries": access_overlap_itins,
        "access_energy_inconsistent_itineraries": access_inconsistent_itins,
    }

    dx = 0.0
    dn = 0.0
    last_iteration = 0

    d_vt_route: Dict[str, Dict[int, float]] = {}
    d_vt_dep: Dict[str, Dict[int, float]] = {}
    e_dep: Dict[str, Dict[int, float]] = {}
    flows_prev: Dict[str, Dict[str, Dict[int, float]]] | None = None
    shadow_prices: Dict[str, Dict[int, float]] | None = None
    raw_surcharge: Dict[str, Dict[int, float]] = {s: {t: 0.0 for t in times} for s in ev_stations}
    raw_surcharge_uncapped: Dict[str, Dict[int, float | None]] = {s: {t: 0.0 for t in times} for s in ev_stations}
    raw_zero_streak: Dict[str, Dict[int, int]] = {s: {t: 0 for t in times} for s in ev_stations}
    shed_ev: Dict[str, Dict[int, float]] | None = None
    shed_vt: Dict[str, Dict[int, float]] | None = None
    P_vt_lp: Dict[str, Dict[int, float]] | None = None
    B_vt_lp: Dict[str, Dict[int, float]] | None = None
    shared_power_residuals: Dict[str, float] | None = None
    shared_power_lp_diag: Dict[str, Any] = {}
    load_agg: Dict[str, Dict[str, Dict[int, float]]] = {}
    lp_ok = False
    missing_inventory_reporting = False
    grid_results: Dict[str, Any] = {
        "distribution_grid_enabled": bool(config.get("use_distribution_grid", False)),
        "station_to_bus": {},
        "bus_voltage": {},
        "branch_flow_p": {},
        "branch_loading_ratio": {},
        "substation_loading": {},
        "station_grid_available_power": {s: {t: float(station_params[s]["P_site"][t]) for t in times} for s in ev_stations},
        "station_grid_shadow_price": {s: {t: 0.0 for t in times} for s in ev_stations},
        "grid_binding_flags": {},
        "grid_binding_count": 0,
        "voltage_violation_count": 0,
        "branch_overload_count": 0,
        "substation_binding_count": 0,
    }
    effective_prices = {s: {t: float(electricity_price[s][t]) for t in times} for s in ev_stations}
    effective_price_components = {
        s: {
            t: {
                "base_price": float(electricity_price[s][t]),
                "grid_component": 0.0,
                "local_component": 0.0,
                "effective_price": float(electricity_price[s][t]),
            }
            for t in times
        }
        for s in ev_stations
    }

    costs: Dict[str, Dict[int, Dict[str, float]]] = {}
    x_new = arc_flows
    u_new = utilization
    good_count = 0
    prev_metric = float("inf")
    eps_improve = float(config.get("eps_improve", 0.0) or 0.0)
    current_max_iter = int(config["max_iter"])
    auto_extend = int(config.get("auto_extend_step", config.get("max_iter_auto_extend", 0)))
    max_total_iter = int(config.get("max_total_iter", 800))
    iteration = 0
    ev_station_waits = {s: {t: 0.0 for t in times} for s in ev_stations}
    vt_departure_waits = {s: {"fast": {t: 0.0 for t in times}, "slow": {t: 0.0 for t in times}} for s in hybrid_stations}
    vt_departure_flow_by_class = {s: {"fast": {t: 0.0 for t in times}, "slow": {t: 0.0 for t in times}} for s in hybrid_stations}
    while iteration < current_max_iter and iteration < max_total_iter:
        last_iteration = iteration + 1
        g_by_time = {t: g_series[min(len(g_series) - 1, idx)] for idx, t in enumerate(times)}
        tau = compute_road_times(arc_flows, arc_params, g_by_time, times)
        ev_station_waits = compute_station_waits(utilization, station_params, times)
        # Traveler-facing effective price uses one unified composition throughout the model:
        # effective_price = base_price + grid_component + local_component.
        effective_prices = {
            s: {
                t: electricity_price[s][t]
                + float((grid_results.get("station_grid_shadow_price", {}) or {}).get(s, {}).get(t, 0.0))
                + float(energy_surcharge.get(s, {}).get(t, 0.0))
                for t in times
            }
            for s in ev_stations
        }
        effective_price_components = {
            s: {
                t: {
                    "base_price": float(electricity_price[s][t]),
                    "grid_component": float((grid_results.get("station_grid_shadow_price", {}) or {}).get(s, {}).get(t, 0.0)),
                    "local_component": float(energy_surcharge.get(s, {}).get(t, 0.0)),
                    "effective_price": float(effective_prices[s][t]),
                }
                for t in times
            }
            for s in ev_stations
        }
        generated = generate_itineraries(data, tau, ev_station_waits, config)
        if generated:
            for it in generated:
                if it["id"] not in seen_ids:
                    itineraries.append(it)
                    seen_ids.add(it["id"])
        inc_road, inc_station = build_incidence(itineraries, arcs, ev_stations, times)
        costs = compute_itinerary_costs(
            itineraries,
            tau,
            ev_station_waits,
            effective_prices,
            times,
            vt_departure_waits=vt_departure_waits,
            transfer_time_by_station=data.get("parameters", {}).get("transfer_time_by_station"),
            transfer_time_default=float(data.get("parameters", {}).get("transfer_time_default", 0.0)),
        )
        flows, logit_details = logit_assignment(
            itineraries,
            costs,
            data["parameters"]["q"],
            data["parameters"]["VOT"],
            data["parameters"]["lambda"],
            times,
            vt_service_prob=vt_service_prob,
            ev_service_prob=ev_service_prob,
            vt_service_prob_floor=1.0e-4,
            ev_service_prob_floor=float(config.get("ev_reliability_floor", 0.05)),
            vt_reliability_gamma=float(config["vt_reliability_gamma"]),
            ev_reliability_gamma=float(config.get("ev_reliability_gamma", 0.0)),
            vt_service_prob_skip_below=float(config["vt_reliability_skip_below"]),
            fail_on_infeasible_demand=bool(config.get("fail_on_infeasible_demand", False)),
        )
        vt_departure_waits, vt_departure_flow_by_class = compute_vt_departure_waits(
            data,
            itineraries,
            flows,
            times,
            config,
        )
        diagnostics["vt_departure_waits"] = vt_departure_waits
        diagnostics["vt_departure_flow_by_class"] = vt_departure_flow_by_class
        diagnostics["unserved_demand"] = logit_details.get("unserved_demand", {})
        diagnostics["unserved_demand_total"] = logit_details.get("unserved_demand_total", 0.0)
        diagnostics["unserved_cases_count"] = logit_details.get("unserved_cases_count", 0)

        cap_pax = data["parameters"].get("vertiport_cap_pax")
        if cap_pax:
            flows, cap_stats = _apply_vertiport_caps(
                flows,
                itineraries,
                times,
                cap_pax,
                logit_details.get("utilities", {}),
                diagnostics["unserved_demand"],
                min_vt_reroute_share=float(config.get("min_vt_reroute_share", 0.0)),
                min_multimodal_reroute_share=float(config.get("min_multimodal_reroute_share", 0.0)),
                reroute_logit_temperature=float(config.get("reroute_logit_temperature", 1.0)),
            )
            diagnostics["vertiport_cap_triggered_count"] = cap_stats.get("triggered_count", 0.0)
            diagnostics["vertiport_cap_excess_pax"] = cap_stats.get("excess_pax", 0.0)
            diagnostics["vertiport_cap_rerouted_to_evtol"] = cap_stats.get("rerouted_to_evtol", 0.0)
            diagnostics["vertiport_cap_rerouted_to_multimodal"] = cap_stats.get("rerouted_to_multimodal", 0.0)
            diagnostics["vertiport_cap_rerouted_to_ev"] = cap_stats.get("rerouted_to_ev", 0.0)
            diagnostics["vertiport_cap_unserved"] = cap_stats.get("unserved", 0.0)
            diagnostics["vertiport_cap_feasible_vt_but_all_ev_count"] = cap_stats.get("feasible_vt_but_all_ev_count", 0.0)
            diagnostics["vertiport_cap_feasible_total_alts_count"] = cap_stats.get("feasible_total_alts_count", 0.0)
            diagnostics["vertiport_cap_feasible_multimodal_alts_count"] = cap_stats.get("feasible_multimodal_alts_count", 0.0)

        vt_fast = float(data.get("parameters", {}).get("vt_pax_per_departure_fast", 2.0))
        vt_slow = float(data.get("parameters", {}).get("vt_pax_per_departure_slow", 4.0))
        vt_turn_lag = int(data.get("parameters", {}).get("vt_turnaround_lag", 1))
        vt_init = data.get("parameters", {}).get("vt_aircraft_init_by_station", {})
        flows, aircraft_diag = _enforce_aircraft_inventory(
            flows,
            itineraries,
            times,
            delta_t,
            logit_details.get("utilities", {}),
            diagnostics["unserved_demand"],
            vt_fast,
            vt_slow,
            vt_turn_lag,
            vt_init,
            min_vt_reroute_share=float(config.get("min_vt_reroute_share", 0.0)),
            min_multimodal_reroute_share=float(config.get("min_multimodal_reroute_share", 0.0)),
            reroute_logit_temperature=float(config.get("reroute_logit_temperature", 1.0)),
        )
        diagnostics.update(aircraft_diag)
        diagnostics["unserved_demand_total"] = sum(
            float(v)
            for od in diagnostics.get("unserved_demand", {}).values()
            for grp in od.values()
            for v in grp.values()
        )
        diagnostics["unserved_cases_count"] = sum(
            1
            for od in diagnostics.get("unserved_demand", {}).values()
            for grp in od.values()
            for _t in grp.keys()
        )

        alpha_flow_cfg = config.get("flow_msa_alpha")
        phi_flow = float(alpha_flow_cfg) if alpha_flow_cfg is not None else (2.0 / (iteration + 2.0))
        if alpha_flow_cfg is not None and bool(config.get("strict_convergence", False)):
            phi_flow = min(phi_flow, 2.0 / (iteration + 2.0))
        phi_flow = min(1.0, max(1.0e-3, phi_flow))
        if flows_prev is not None:
            for it in itineraries:
                it_id = it["id"]
                for g in groups:
                    prev_map = flows_prev.get(it_id, {}).get(g, {})
                    cur_map = flows.get(it_id, {}).get(g, {})
                    for t in times:
                        prev_v = float(prev_map.get(t, 0.0))
                        cur_v = float(cur_map.get(t, 0.0))
                        flows[it_id].setdefault(g, {})[t] = (1.0 - phi_flow) * prev_v + phi_flow * cur_v
        flows_prev = {
            it["id"]: {g: {t: float(flows.get(it["id"], {}).get(g, {}).get(t, 0.0)) for t in times} for g in groups}
            for it in itineraries
        }

        d_vt_route = aggregate_evtol_demand(flows, itineraries, times)
        d_vt_dep = aggregate_evtol_dep_demand(itineraries, flows, times)
        e_dep = compute_evtol_energy_demand(d_vt_route, itineraries, times)

        x_new = aggregate_arc_flows(itineraries, flows, times)
        u_new = aggregate_ev_station_utilization(itineraries, flows, times)
        x_new = _fill_missing(x_new, arcs, times)
        u_new = _fill_missing(u_new, ev_stations, times)

        dx_abs = max(abs(x_new[a][t] - arc_flows[a][t]) for a in arcs for t in times)
        max_flow = max(
            max(abs(x_new[a][t]), abs(arc_flows[a][t])) for a in arcs for t in times
        )
        dx = dx_abs / max(1.0, max_flow)

        phi = 1.0 / (iteration + 1.0)
        for a in arcs:
            for t in times:
                arc_flows[a][t] = (1.0 - phi) * arc_flows[a][t] + phi * x_new[a][t]
        for s in ev_stations:
            for t in times:
                utilization[s][t] = (1.0 - phi) * utilization[s][t] + phi * u_new[s][t]
        inflow_total, outflow_total, model_inflow_total, model_outflow_total, bg_inflow_total, bg_outflow_total = _boundary_flows_with_background(
            arc_flows,
            data["parameters"]["boundary_in"],
            data["parameters"]["boundary_out"],
            times,
            data["parameters"].get("boundary_in_background"),
            data["parameters"].get("boundary_out_background"),
        )
        inflow = [val / delta_t for val in inflow_total]
        outflow = [val / delta_t for val in outflow_total]
        n_new = update_accumulation(data["parameters"]["n0"], inflow, outflow, delta_t)
        dn = max(abs(n_new[idx] - n_series[idx]) for idx in range(len(n_series)))
        n_series = n_new
        g_series = compute_g(n_series, data["parameters"]["mfd"])

        load_agg = compute_station_loads_from_flows(data, itineraries, flows, times)
        e_vt_req = load_agg["E_vt_req"]
        p_ev_req = load_agg["P_ev_req_kw"]
        p_vt_req_energy = load_agg["P_vt_req_kw_energy"]
        p_vt_req_grid = load_agg["P_vt_req_kw_grid"]
        ev_energy_kwh = load_agg["E_ev_req"]
        station_power_request = {
            s: {t: float(p_ev_req.get(s, {}).get(t, 0.0)) + float(p_vt_req_grid.get(s, {}).get(t, 0.0)) for t in times}
            for s in ev_stations
        }
        if bool(config.get("use_distribution_grid", False)):
            grid_results = solve_distribution_grid(data, station_power_request, times)
        else:
            grid_results = {
                "distribution_grid_enabled": False,
                "station_to_bus": {},
                "bus_voltage": {},
                "branch_flow_p": {},
                "branch_loading_ratio": {},
                "substation_loading": {},
                "station_grid_available_power": {s: {t: float(station_params[s]["P_site"][t]) for t in times} for s in ev_stations},
                "station_grid_shadow_price": {s: {t: 0.0 for t in times} for s in ev_stations},
                "grid_binding_flags": {},
                "grid_binding_count": 0,
                "voltage_violation_count": 0,
                "branch_overload_count": 0,
                "substation_binding_count": 0,
            }
        data.setdefault("diagnostics_runtime", {})["station_power_cap_effective"] = {
            s: {
                t: min(float(station_params[s]["P_site"][t]), float(grid_results.get("station_grid_available_power", {}).get(s, {}).get(t, station_params[s]["P_site"][t])))
                for t in times
            }
            for s in ev_stations
        }
        surcharge_method = config.get("surcharge_method", "shadow_lp")
        shadow_scale = float(config.get("shadow_price_scale", 1.0))
        surcharge_beta = max(1.0e-6, float(config.get("surcharge_beta", 1.0)))
        cap_mult = float(config.get("shadow_price_cap_mult", 10.0))
        lp_ok = False
        fallback_used = False
        solver_requested = str(config.get("shared_power_solver", "highs")).lower()
        if solver_requested != "highs":
            raise ValueError("config.shared_power_solver must be 'highs' (LP-only mode)")

        shadow_prices: Dict[str, Dict[int, float | None]] = {s: {t: None for t in times} for s in ev_stations}
        shed_ev = {s: {t: 0.0 for t in times} for s in power_stations}
        shed_vt = {s: {t: 0.0 for t in times} for s in power_stations}
        shared_power_residuals = {"INV3": 0.0}
        B_vt_lp = {s: {t: 0.0 for t in times} for s in power_stations}
        P_vt_lp = {s: {t: 0.0 for t in times} for s in power_stations}
        heuristic_surcharge = {s: {t: 0.0 for t in times} for s in ev_stations}

        if surcharge_method != "shadow_lp":
            raise ValueError("Only surcharge_method='shadow_lp' is supported")
        (
            B_vt_lp,
            P_vt_lp,
            shed_ev,
            shed_vt,
            shadow_prices,
            shared_power_residuals,
            shared_power_lp_diag,
        ) = charging.solve_shared_power_inventory_lp(data, e_dep, ev_energy_kwh)
        diagnostics["shared_power_solver_used"] = charging.LAST_SHARED_POWER_SOLVER_USED
        runtime_diag = data.get("diagnostics_runtime", {})
        diagnostics["lp_ok"] = bool(runtime_diag.get("lp_ok", diagnostics["shared_power_solver_used"] == "highs"))
        if runtime_diag.get("lp_failure") is not None:
            diagnostics["lp_failure"] = runtime_diag.get("lp_failure")
        lp_ok = diagnostics["shared_power_solver_used"] == "highs" and bool(diagnostics.get("lp_ok", False))
        fallback_used = diagnostics["shared_power_solver_used"] != "highs" or not lp_ok

        vt_service_prob_target = {s: {t: 1.0 for t in times} for s in power_stations}
        shed_ratio_map = {s: {t: 0.0 for t in times} for s in ev_stations}
        cap_binding_map = {s: {t: False for t in times} for s in ev_stations}

        for s in ev_stations:
            for t in times:
                base_price = max(1.0e-6, float(electricity_price[s][t]))
                cap, cap_mult_value, cap_abs_value, cap_mode = _compute_cap_values(base_price, config)
                local_mu_kw = float((shadow_prices.get(s, {}) or {}).get(t, 0.0) or 0.0)
                grid_mu_kw = float((grid_results.get("station_grid_shadow_price", {}) or {}).get(s, {}).get(t, 0.0) or 0.0)
                mu_kw = local_mu_kw + float(config.get("grid_shadow_price_scale", 1.0)) * grid_mu_kw
                if diagnostics.get("shared_power_solver_used") == "highs":
                    raw = shadow_scale * (float(mu_kw or 0.0) ** surcharge_beta) / max(1e-9, delta_t)
                    raw_surcharge_uncapped[s][t] = raw
                    raw_surcharge[s][t] = min(raw, cap)
                else:
                    raw_surcharge_uncapped[s][t] = None
                    heuristic_surcharge[s][t] = max(0.0, (float(p_ev_req.get(s, {}).get(t, 0.0)) + float(p_vt_req_grid.get(s, {}).get(t, 0.0))) / max(1e-6, float(data["diagnostics_runtime"]["station_power_cap_effective"].get(s, {}).get(t, station_params[s]["P_site"][t]))) - 1.0) * cap
                    raw_surcharge[s][t] = min(heuristic_surcharge[s][t], cap)
                cap_binding_map[s][t] = (raw_surcharge_uncapped[s][t] is not None and float(raw_surcharge_uncapped[s][t]) > cap - 1.0e-9) or (raw_surcharge_uncapped[s][t] is None and raw_surcharge[s][t] >= cap - 1.0e-9)

                req_e_vt = float(e_vt_req.get(s, {}).get(t, 0.0))
                shed_vt_kwh = float((shed_vt or {}).get(s, {}).get(t, 0.0))
                if req_e_vt <= 1.0e-9:
                    target_prob = 1.0
                    shed_ratio = 0.0
                else:
                    shed_ratio = min(1.0, max(0.0, shed_vt_kwh / max(1.0e-9, req_e_vt)))
                    target_prob = max(float(config["vt_reliability_floor"]), 1.0 - shed_ratio)
                if s in vt_service_prob_target:
                    vt_service_prob_target[s][t] = target_prob
                shed_ratio_map[s][t] = shed_ratio

        prev_surcharge = {s: {t: energy_surcharge[s][t] for t in times} for s in ev_stations}
        prev_raw = {s: {t: float(raw_surcharge[s][t]) for t in times} for s in ev_stations}
        alpha_override = config.get("surcharge_msa_alpha")
        alpha = float(alpha_override) if alpha_override is not None else 1.0 / (iteration + 1.0)
        alpha_vt = config.get("vt_reliability_alpha")
        alpha_vt = float(alpha_vt) if alpha_vt is not None else alpha

        for s in ev_stations:
            for t in times:
                candidate = (1.0 - alpha) * energy_surcharge[s][t] + alpha * raw_surcharge[s][t]
                raw_thr = float(config.get("surcharge_decay_raw_threshold", 1.0e-3))
                decay = float(config.get("surcharge_decay_factor", 0.6))
                if abs(float(raw_surcharge[s][t])) <= raw_thr:
                    raw_zero_streak[s][t] += 1
                    if raw_zero_streak[s][t] >= 3:
                        energy_surcharge[s][t] = 0.0
                    else:
                        candidate = min(candidate, decay * energy_surcharge[s][t])
                        energy_surcharge[s][t] = max(0.0, candidate)
                else:
                    raw_zero_streak[s][t] = 0
                    energy_surcharge[s][t] = max(0.0, candidate)
                if s in vt_service_prob_target:
                    vt_next = (1.0 - alpha_vt) * vt_service_prob[s][t] + alpha_vt * vt_service_prob_target[s][t]
                    vt_floor = float(config.get("vt_reliability_floor", 0.05))
                    vt_service_prob[s][t] = min(1.0, max(vt_floor, vt_next))
                base_price = max(1.0e-6, float(electricity_price[s][t]))
                p_site_raw = float(data["diagnostics_runtime"]["station_power_cap_effective"].get(s, {}).get(t, station_params[s]["P_site"][t]))
                p_site_den = _safe_den(p_site_raw)
                p_ev_req_kw = float(p_ev_req[s][t])
                p_vt_req_energy_kw = float(p_vt_req_energy.get(s, {}).get(t, 0.0))
                p_vt_req_grid_kw = float(p_vt_req_grid.get(s, {}).get(t, 0.0))
                p_ev_served_kw = max(0.0, p_ev_req_kw - (shed_ev or {}).get(s, {}).get(t, 0.0))
                if s not in power_stations:
                    p_ev_served_kw = min(p_ev_served_kw, p_site_raw)
                p_vt_served_kw = (P_vt_lp or {}).get(s, {}).get(t, 0.0)
                t_idx = times.index(t)
                t_next = times[t_idx + 1] if t_idx + 1 < len(times) else None
                b_before = (B_vt_lp or {}).get(s, {}).get(t)
                terminal_key = times[-1] + 1
                if t_next is not None:
                    b_after = (B_vt_lp or {}).get(s, {}).get(t_next)
                elif (B_vt_lp or {}).get(s, {}).get(terminal_key) is not None:
                    b_after = (B_vt_lp or {}).get(s, {}).get(terminal_key)
                else:
                    eta = 1.0
                    if s in data["parameters"].get("vertiport_storage", {}):
                        eta = float(data["parameters"]["vertiport_storage"][s].get("eta_ch", 1.0))
                    if b_before is not None:
                        b_after = float(b_before) + eta * p_vt_served_kw * delta_t - (
                            float(e_vt_req[s][t]) - float((shed_vt or {}).get(s, {}).get(t, 0.0))
                        )
                    else:
                        b_after = None
                if b_before is None:
                    if s in power_stations:
                        missing_inventory_reporting = True
                    b_before = 0.0
                if b_after is None:
                    if s in power_stations:
                        missing_inventory_reporting = True
                    b_after = 0.0
                b_before = float(b_before)
                b_after = float(b_after)
                p_net_served_kw = p_ev_served_kw + p_vt_served_kw
                current_shed_ratio = 0.0 if float(e_vt_req.get(s, {}).get(t, 0.0)) <= 1.0e-12 else float((shed_vt or {}).get(s, {}).get(t, 0.0)) / max(1.0e-12, float(e_vt_req.get(s, {}).get(t, 0.0)))
                current_shed_ratio = min(1.0, max(0.0, current_shed_ratio))
                vt_prob_from_shed = min(1.0, max(0.0, 1.0 - current_shed_ratio))
                ev_req_den = max(1.0e-9, float(p_ev_req_kw))
                ev_prob_from_shed = 1.0 - float((shed_ev or {}).get(s, {}).get(t, 0.0)) / ev_req_den
                grid_cap_here = float((grid_results.get("station_grid_available_power", {}) or {}).get(s, {}).get(t, p_site_raw))
                total_req_here = float(p_ev_req_kw + p_vt_req_grid_kw)
                grid_prob = 1.0 if total_req_here <= 1.0e-9 else min(1.0, max(0.0, grid_cap_here / max(1.0e-9, total_req_here)))
                ev_floor = float(config.get("ev_reliability_floor", 0.05))
                ev_prob_from_shed = min(1.0, max(ev_floor, min(ev_prob_from_shed, grid_prob)))
                ev_service_prob[s][t] = ev_prob_from_shed
                solver_used = diagnostics.get("shared_power_solver_used")
                local_mu_value = float((shadow_prices.get(s, {}) or {}).get(t, 0.0) or 0.0)
                grid_mu_value = float((grid_results.get("station_grid_shadow_price", {}) or {}).get(s, {}).get(t, 0.0) or 0.0)
                total_mu_value = local_mu_value + float(config.get("grid_shadow_price_scale", 1.0)) * grid_mu_value
                mu_entry = _clean_number(local_mu_value) if local_mu_value is not None and solver_used == "highs" else None
                uncapped_entry = _clean_number(raw_surcharge_uncapped[s][t]) if (raw_surcharge_uncapped[s][t] is not None and solver_used == "highs") else None
                ratio_req_exogenous = (p_ev_req_kw + p_vt_req_grid_kw) / p_site_den
                ratio_req_energy = (p_ev_req_kw + p_vt_req_energy_kw) / p_site_den
                base_price = max(1.0e-6, float(electricity_price[s][t]))
                cap, cap_mult_value, cap_abs_value, cap_mode = _compute_cap_values(base_price, config)
                diagnostics["power_tightness"].setdefault(s, {})[t] = {
                    "P_site_raw": p_site_raw,
                    "P_site_den": p_site_den,
                    "ratio_req_exogenous": _clean_number(ratio_req_exogenous),
                    "ratio_req_energy": _clean_number(ratio_req_energy),
                    "ratio_net_actual": _clean_number(p_net_served_kw / p_site_den),
                    "P_ev_req_kw": _clean_number(p_ev_req_kw),
                    "P_vt_req_kw_energy": _clean_number(p_vt_req_energy_kw),
                    "P_vt_req_kw_grid": _clean_number(p_vt_req_grid_kw),
                    "mu_kw": mu_entry,
                    "local_mu_kw": _clean_number(local_mu_value),
                    "grid_mu_kw": _clean_number(grid_mu_value),
                    "mu_total_kw": _clean_number(total_mu_value),
                    "local_dual_available": bool(s in power_stations),
                    "base_price": float(electricity_price[s][t]),
                    "grid_price_component": float((grid_results.get("station_grid_shadow_price", {}) or {}).get(s, {}).get(t, 0.0)),
                    "local_price_component": float(energy_surcharge[s][t]),
                    "effective_price": float(effective_prices[s][t]),
                    "raw_surcharge_uncapped_kwh": uncapped_entry,
                    "raw_surcharge_uncapped": uncapped_entry,
                    "raw_surcharge_capped_kwh": _clean_number(raw_surcharge[s][t]),
                    "raw_surcharge_capped": _clean_number(raw_surcharge[s][t]),
                    "raw_surcharge": _clean_number(raw_surcharge[s][t]),
                    "heuristic_surcharge": _clean_number(heuristic_surcharge[s][t]) if solver_used != "highs" else None,
                    "surcharge_smoothed": _clean_number(energy_surcharge[s][t]),
                    "cap": _clean_number(cap),
                    "cap_used": cap,
                    "cap_mult_value": cap_mult_value,
                    "cap_abs_value": cap_abs_value,
                    "cap_mode": cap_mode,
                    "cap_binding": cap_binding_map[s][t],
                    "E_vt_req_kwh": e_vt_req.get(s, {}).get(t, 0.0),
                    "P_vt_charge_kw": p_vt_served_kw,
                    "B_before_kwh": b_before,
                    "B_after_kwh": b_after,
                    "shed_ev_kw": _clean_number((shed_ev or {}).get(s, {}).get(t, 0.0)),
                    "shed_vt_kwh": _clean_number((shed_vt or {}).get(s, {}).get(t, 0.0)),
                    "p_ev_served_kw": p_ev_served_kw,
                    "p_vt_served_kw": p_vt_served_kw,
                    "p_net_served_kw": p_net_served_kw,
                    "vt_service_prob_target": vt_service_prob_target.get(s, {}).get(t, 1.0),
                    "vt_service_prob": vt_service_prob.get(s, {}).get(t, 1.0),
                    "ev_service_prob": ev_service_prob.get(s, {}).get(t, 1.0),
                    "shed_ratio": current_shed_ratio,
                    "solver_used": solver_used,
                    "fallback_used": fallback_used,
                }

        dprice = max(
            abs(energy_surcharge[s][t] - prev_surcharge[s][t]) for s in ev_stations for t in times
        )
        dprice_raw = max(
            abs(raw_surcharge[s][t] - prev_raw[s][t]) for s in ev_stations for t in times
        )
        max_surcharge = max(energy_surcharge[s][t] for s in ev_stations for t in times) if ev_stations else 0.0
        diagnostics["dx_history"].append(dx)
        diagnostics["dn_history"].append(dn)
        diagnostics["dprice_history"].append(dprice)
        diagnostics["dprice_raw_history"].append(dprice_raw)
        diagnostics["max_surcharge_history"].append(max_surcharge)

        peak_entries = [diagnostics["power_tightness"].get(s, {}).get(peak_t, {}) for s in power_stations]
        if peak_entries:
            bind_ratio = sum(1.0 for e in peak_entries if e.get("cap_binding")) / max(1, len(peak_entries))
            avg_vt_prob = sum(float(e.get("vt_service_prob", 1.0)) for e in peak_entries) / max(1, len(peak_entries))
            avg_ev_prob = sum(float(e.get("ev_service_prob", 1.0)) for e in peak_entries) / max(1, len(peak_entries))
            avg_ratio_req = sum(float(e.get("ratio_req_exogenous", 0.0)) for e in peak_entries) / max(1, len(peak_entries))
            diagnostics["power_tightness_summary_history"].append({
                "iteration": iteration + 1,
                "peak_t": peak_t,
                "cap_binding_ratio": bind_ratio,
                "avg_vt_service_prob": avg_vt_prob,
                "avg_ev_service_prob": avg_ev_prob,
                "avg_ratio_req_exogenous": avg_ratio_req,
            })

        snapshot = {
            "iteration": iteration + 1,
            "t": peak_t,
            "stations": {},
        }
        for s in power_stations[:3]:
            base_price = float(electricity_price[s][peak_t])
            grid_component = float((grid_results.get("station_grid_shadow_price", {}) or {}).get(s, {}).get(peak_t, 0.0))
            local_component = float(energy_surcharge[s][peak_t])
            snapshot["stations"][s] = {
                "base_price": base_price,
                "grid_component": grid_component,
                "local_component": local_component,
                "effective_price": float(effective_prices[s][peak_t]),
            }
        diagnostics["price_snapshots"].append(snapshot)

        pure_ev_total = 0.0
        pure_vt_total = 0.0
        multimodal_total = 0.0
        if representative_od:
            for it in itineraries:
                if f"{it['od'][0]}-{it['od'][1]}" != representative_od:
                    continue
                for _, time_map in flows[it["id"]].items():
                    val = time_map.get(peak_t, 0.0)
                    if is_multimodal_evtol(it):
                        multimodal_total += val
                    elif is_evtol_itinerary(it):
                        pure_vt_total += val
                    else:
                        pure_ev_total += val
        total_flow = pure_ev_total + pure_vt_total + multimodal_total
        ev_share = pure_ev_total / total_flow if total_flow > 0.0 else 0.0
        vt_share = pure_vt_total / total_flow if total_flow > 0.0 else 0.0
        mm_share = multimodal_total / total_flow if total_flow > 0.0 else 0.0
        diagnostics["mode_share"].append(
            {
                "iteration": iteration + 1,
                "od": representative_od,
                "t": peak_t,
                "ev_share": ev_share,
                "vt_share": vt_share,
                "multimodal_share": mm_share,
            }
        )
        diagnostics["mode_share_by_supermode"].append(
            {
                "iteration": iteration + 1,
                "od": representative_od,
                "t": peak_t,
                "shares": {"EV": ev_share, "eVTOL": vt_share, "EV_to_eVTOL": mm_share},
            }
        )
        mode_share_by_group = {}
        mode_share_by_group_and_mode = {}
        mode_share_by_group_and_supermode = {}
        if representative_od:
            expected_groups = set(data.get("sets", {}).get("groups", []))
            rep_its = [it for it in itineraries if f"{it['od'][0]}-{it['od'][1]}" == representative_od]
            groups_in_flows = sorted(
                {
                    grp
                    for it in rep_its
                    for grp in flows.get(it["id"], {}).keys()
                    if (not expected_groups) or (grp in expected_groups)
                }
            )
            if not groups_in_flows:
                flow_keys = {it_id: list(group_map.keys()) for it_id, group_map in flows.items()}
                raise ValueError(
                    f"No group keys found in flows for OD {representative_od}. "
                    f"Representative itineraries={[it['id'] for it in rep_its]}, flow keys={flow_keys}"
                )
            mode_keys = ["pure_EV", "pure_eVTOL_fast", "pure_eVTOL_slow", "multimodal_EV_to_eVTOL_fast", "multimodal_EV_to_eVTOL_slow"]
            super_keys = ["EV", "eVTOL", "EV_to_eVTOL"]
            for grp in groups_in_flows:
                mode_vals = {k: 0.0 for k in mode_keys}
                super_vals = {k: 0.0 for k in super_keys}
                for it in rep_its:
                    val = flows[it["id"]].get(grp, {}).get(peak_t, 0.0)
                    mlabel = classify_mode_label(it)
                    smode = classify_supermode(it)
                    mode_vals[mlabel] = mode_vals.get(mlabel, 0.0) + val
                    super_vals[smode] = super_vals.get(smode, 0.0) + val
                den_all = max(1e-12, sum(mode_vals.values()))
                den_super = max(1e-12, sum(super_vals.values()))
                mode_share_by_group_and_mode[grp] = {k: v / den_all for k, v in mode_vals.items()}
                mode_share_by_group_and_supermode[grp] = {k: v / den_super for k, v in super_vals.items()}
                mode_share_by_group[grp] = {
                    "EV": mode_share_by_group_and_supermode[grp]["EV"],
                    "eVTOL": mode_share_by_group_and_supermode[grp]["eVTOL"],
                    "EV_to_eVTOL": mode_share_by_group_and_supermode[grp]["EV_to_eVTOL"],
                }
        diagnostics["mode_share_by_group"].append(
            {"iteration": iteration + 1, "od": representative_od, "t": peak_t, "shares": mode_share_by_group, "legacy": True}
        )
        diagnostics["mode_share_by_group_and_mode"].append(
            {"iteration": iteration + 1, "od": representative_od, "t": peak_t, "shares": mode_share_by_group_and_mode}
        )
        diagnostics["mode_share_by_group_and_supermode"].append(
            {"iteration": iteration + 1, "od": representative_od, "t": peak_t, "shares": mode_share_by_group_and_supermode}
        )
        service_class_share = {}
        pure_evtol_service_class_share = {}
        all_evtol_service_class_share = {}
        if representative_od:
            rep_its = [it for it in itineraries if f"{it['od'][0]}-{it['od'][1]}" == representative_od]
            pure_fast_total = 0.0
            pure_slow_total = 0.0
            all_fast_total = 0.0
            all_slow_total = 0.0
            for it in rep_its:
                if not is_evtol_itinerary(it):
                    continue
                cls = get_evtol_service_class(it)
                for _, time_map in flows.get(it["id"], {}).items():
                    val = time_map.get(peak_t, 0.0)
                    if cls == "fast":
                        all_fast_total += val
                        if not is_multimodal_evtol(it):
                            pure_fast_total += val
                    else:
                        all_slow_total += val
                        if not is_multimodal_evtol(it):
                            pure_slow_total += val
            den_all = max(1.0e-12, all_fast_total + all_slow_total)
            den_pure = max(1.0e-12, pure_fast_total + pure_slow_total)
            all_evtol_service_class_share = {"fast": all_fast_total / den_all, "slow": all_slow_total / den_all}
            pure_evtol_service_class_share = {"fast": pure_fast_total / den_pure, "slow": pure_slow_total / den_pure}
            service_class_share = dict(all_evtol_service_class_share)
        diagnostics["mode_share_by_service_class"].append(
            {"iteration": iteration + 1, "od": representative_od, "t": peak_t, "shares": service_class_share}
        )
        diagnostics["pure_evtol_service_class_share"].append(
            {"iteration": iteration + 1, "od": representative_od, "t": peak_t, "shares": pure_evtol_service_class_share}
        )
        diagnostics["all_evtol_service_class_share"].append(
            {"iteration": iteration + 1, "od": representative_od, "t": peak_t, "shares": all_evtol_service_class_share}
        )
        diagnostics["surcharge_history"].append(
            {
                "iteration": iteration + 1,
                "surcharge": {s: {t: energy_surcharge[s][t] for t in times} for s in power_stations},
            }
        )
        if len(diagnostics["surcharge_history"]) > 20:
            diagnostics["surcharge_history"] = diagnostics["surcharge_history"][-20:]

        should_print = (iteration == 0) or ((iteration + 1) % 5 == 0)
        if should_print:
            print(
                "iter="
                f"{iteration + 1} dx={dx:.6f} dn={dn:.6f} dprice={dprice:.6f} "
                f"max_surcharge={max_surcharge:.6f} alpha={alpha:.3f}"
            )
            if "s1" in snapshot["stations"]:
                vals = snapshot["stations"]["s1"]
                print(
                    f"peak_t={peak_t} s1_eff_price={vals['effective_price']:.3f} "
                    f"(base={vals['base_price']:.3f},grid={vals.get('grid_component', 0.0):.3f},local={vals.get('local_component', 0.0):.3f})"
                )
            if representative_od:
                print(
                    f"peak_t={peak_t} od={representative_od} ev_share={ev_share:.3f} vt_share={vt_share:.3f}"
                )

        if iteration + 1 >= min_iters:
            metric = max(dx, dn, dprice_raw)
            improved = (prev_metric - metric) > eps_improve
            strict_gate_required = bool(config.get("patience_require_strict_gate", True)) and bool(config.get("strict_convergence", False))
            strict_gate_ok = False
            if strict_gate_required:
                residuals_iter = compute_residuals(
                    data,
                    itineraries,
                    flows,
                    arc_flows,
                    x_new,
                    tau,
                    n_series,
                    g_series,
                    inc_road,
                    inc_station,
                    utilization,
                    u_new,
                )
                strict_gate_ok = _convergence_flags(
                    dx,
                    dn,
                    dprice_raw,
                    residuals_iter,
                    float(config.get("tol", 0.0)),
                    True,
                )["converged_strictly"]
            if strict_gate_required:
                qualifies = strict_gate_ok
            else:
                qualifies = metric < config["tol"]
            if qualifies:
                good_count += 1
            else:
                good_count = 0
            prev_metric = metric
            if good_count >= patience:
                if not should_print:
                    print(
                        "iter="
                        f"{iteration + 1} dx={dx:.6f} dn={dn:.6f} dprice={dprice:.6f} "
                        f"max_surcharge={max_surcharge:.6f} alpha={alpha:.3f}"
                    )
                    if "s1" in snapshot["stations"]:
                        vals = snapshot["stations"]["s1"]
                        print(
                            f"peak_t={peak_t} s1_eff_price={vals['effective_price']:.3f} "
                            f"(base={vals['base_price']:.3f},grid={vals.get('grid_component', 0.0):.3f},local={vals.get('local_component', 0.0):.3f})"
                        )
                    if representative_od:
                        print(
                            f"peak_t={peak_t} od={representative_od} ev_share={ev_share:.3f} vt_share={vt_share:.3f}"
                        )
                stop_reason = "patience"
                break

        if iteration + 1 >= current_max_iter and good_count < patience and auto_extend > 0 and current_max_iter < max_total_iter:
            current_max_iter = min(max_total_iter, min(1000, current_max_iter + auto_extend))

        iteration += 1

    if stop_reason != "patience":
        stop_reason = "max_total_iter" if iteration >= max_total_iter else "max_iters"

    residuals = compute_residuals(
        data,
        itineraries,
        flows,
        arc_flows,
        x_new,
        tau,
        n_series,
        g_series,
        inc_road,
        inc_station,
        utilization,
        u_new,
    )
    dprice_for_conv = diagnostics.get("dprice_raw_history", [dprice])[-1] if diagnostics.get("dprice_raw_history") else dprice
    conv_flags = _convergence_flags(dx, dn, dprice_for_conv, residuals, float(config.get("tol", 0.0)), bool(config.get("strict_convergence", False)))
    conv_flags["patience_only_stop"] = bool(stop_reason == "patience" and not conv_flags["converged_strictly"])

    if run_dispatch:
        E, p_ch, y, charging_residuals, B_vt, P_vt, inv_residuals, _ = solve_charging(data, e_dep, d_vt_dep)
    else:
        E = {m: {t: 0.0 for t in times} for m in data["sets"]["vehicles"]}
        p_ch = {m: {s: {t: 0.0 for t in times} for s in ev_stations} for m in data["sets"]["vehicles"]}
        y = {m: {s: {t: 0 for t in times} for s in ev_stations} for m in data["sets"]["vehicles"]}
        charging_residuals = {"SKIPPED": 0.0}
        B_vt = B_vt_lp or {}
        if lp_ok and P_vt_lp:
            P_vt = P_vt_lp
        else:
            P_vt = {dep: {t: load_agg.get("P_vt_req", {}).get(dep, {}).get(t, 0.0) for t in times} for dep in load_agg.get("P_vt_req", {})}
        inv_residuals = shared_power_residuals or {}

    cap_violation = 0.0
    cap_pax = data["parameters"].get("vertiport_cap_pax")
    if cap_pax:
        for dep, time_map in d_vt_dep.items():
            for t in times:
                cap_violation = max(cap_violation, max(0.0, time_map[t] - cap_pax[dep][t]))

    power_violation_site_raw = 0.0
    power_violation_effective_cap = 0.0
    power_mode = str(config.get("power_violation_mode", "net"))
    final_effective_caps = data.get("diagnostics_runtime", {}).get("station_power_cap_effective", {})
    for s in power_stations:
        for t in times:
            if run_dispatch:
                total_power = sum(p_ch[m][s][t] for m in data["sets"]["vehicles"])
                if P_vt and s in P_vt:
                    total_power += P_vt[s][t]
            else:
                if power_mode == "request":
                    total_power = load_agg.get("P_total_req", {}).get(s, {}).get(t, 0.0)
                else:
                    p_ev_served = max(0.0, p_ev_req[s][t] - (shed_ev or {}).get(s, {}).get(t, 0.0))
                    p_vt_served = (P_vt_lp or {}).get(s, {}).get(t, p_vt_req_grid[s][t])
                    total_power = p_ev_served + p_vt_served
            power_violation_site_raw = max(power_violation_site_raw, max(0.0, total_power - station_params[s]["P_site"][t]))
            eff_cap_here = float(final_effective_caps.get(s, {}).get(t, station_params[s]["P_site"][t]))
            power_violation_effective_cap = max(power_violation_effective_cap, max(0.0, total_power - eff_cap_here))

    inflow_total, outflow_total, model_inflow_total, model_outflow_total, bg_inflow_total, bg_outflow_total = _boundary_flows_with_background(
        arc_flows,
        data["parameters"]["boundary_in"],
        data["parameters"]["boundary_out"],
        times,
        data["parameters"].get("boundary_in_background"),
        data["parameters"].get("boundary_out_background"),
    )
    g_used_by_time = {t: g_series[min(idx, len(g_series) - 1)] for idx, t in enumerate(times)}
    times_ext = list(times) + [times[-1] + 1] if times else []
    B_vt_report: Dict[str, Dict[int, float | None]] = {}
    storage_model_status_by_station: Dict[str, str] = {}
    for s in power_stations:
        station_has_storage_path = bool((B_vt_lp or {}).get(s))
        storage_model_status_by_station[s] = "active" if station_has_storage_path else "not_modeled_or_inactive"
        B_vt_report[s] = {}
        for idx, t_ext in enumerate(times_ext):
            if B_vt_lp is not None and s in B_vt_lp and t_ext in B_vt_lp[s]:
                B_vt_report[s][t_ext] = float(B_vt_lp[s][t_ext])
            elif not station_has_storage_path:
                B_vt_report[s][t_ext] = None
            elif idx == len(times_ext) - 1 and len(times_ext) >= 2:
                prev_t = times_ext[-2]
                B_vt_report[s][t_ext] = float((B_vt_lp or {}).get(s, {}).get(prev_t, 0.0))
            else:
                B_vt_report[s][t_ext] = float((B_vt_lp or {}).get(s, {}).get(t_ext, 0.0))

    cbd_tau = {}
    for arc, params in arc_params.items():
        if params.get("type") != "CBD":
            continue
        theta = float(params.get("theta", 1.0))
        cbd_tau[arc] = {}
        for t in times:
            g_t = g_used_by_time[t]
            cbd_tau[arc][t] = float(params["tau0"]) * g_t * theta

    positive_mu_points = 0
    stressed_power_points = 0
    stressed_power_no_price_points = 0
    for s in power_stations:
        for t in times:
            mu_here = float(((shadow_prices or {}).get(s, {}).get(t, 0.0) or 0.0))
            if mu_here > 1.0e-9:
                positive_mu_points += 1
            shed_ev_here = float(((shed_ev or {}).get(s, {}).get(t, 0.0) or 0.0))
            shed_vt_here = float(((shed_vt or {}).get(s, {}).get(t, 0.0) or 0.0))
            p_site_here = float(station_params.get(s, {}).get("P_site", {}).get(t, 0.0) or 0.0)
            p_vt_here = float(((P_vt_lp or {}).get(s, {}).get(t, 0.0) or 0.0))
            p_ev_req_here = float((p_ev_req.get(s, {}).get(t, 0.0) or 0.0))
            p_ev_served_here = max(0.0, p_ev_req_here - shed_ev_here)
            p_net_here = p_vt_here + p_ev_served_here
            stressed_here = (
                shed_ev_here > 1.0e-9
                or shed_vt_here > 1.0e-9
                or (p_site_here > 1.0e-9 and p_net_here >= p_site_here - 1.0e-9)
            )
            if stressed_here:
                stressed_power_points += 1
                if mu_here <= 1.0e-9:
                    stressed_power_no_price_points += 1
    diagnostics["power_binding_count"] = int(positive_mu_points)
    diagnostics["power_stress_point_count"] = int(stressed_power_points)
    diagnostics["power_stress_without_price_count"] = int(stressed_power_no_price_points)
    terminal_soc_by_station: Dict[str, Dict[str, float | bool]] = {}
    storage_energy_totals_by_station: Dict[str, Dict[str, float]] = {}
    storage_params = data.get("parameters", {}).get("vertiport_storage", {})
    for s in power_stations:
        b_cfg = storage_params.get(s, {})
        b_init = float(b_cfg.get("B_init", b_cfg.get("B_min", 0.0)))
        b_target = float(charging._terminal_soc_target(b_cfg, data.get("config", {}), s))
        b_end_raw = (B_vt_lp or {}).get(s, {}).get(times[-1] + 1, (B_vt_lp or {}).get(s, {}).get(times[-1])) if times else None
        b_end = float(b_end_raw) if b_end_raw is not None else None
        gap = (b_end - b_target) if b_end is not None else None
        modeled = b_end is not None
        terminal_soc_by_station[s] = {
            "terminal_soc_policy": str(config.get("terminal_soc_policy", "at_least_init")),
            "storage_model_status": storage_model_status_by_station[s],
            "storage_modeled": modeled,
            "terminal_soc_target_kwh": b_target,
            "terminal_soc_actual_end_kwh": b_end,
            "terminal_soc_gap_kwh": gap,
            "terminal_soc_binding": (abs(gap) <= 1.0e-6) if gap is not None else None,
            "terminal_soc_above_target": (b_end >= b_target - 1.0e-6) if b_end is not None else None,
            "initial_soc_kwh": b_init,
            "terminal_vs_init_gap_kwh": (b_end - b_init) if b_end is not None else None,
        }
        charge_kwh = 0.0
        discharge_kwh = 0.0
        for t in times:
            p_ch = float((P_vt_lp or {}).get(s, {}).get(t, 0.0))
            e_served = float((e_dep or {}).get(s, {}).get(t, 0.0)) - float((shed_vt or {}).get(s, {}).get(t, 0.0))
            charge_kwh += max(0.0, p_ch * float(delta_t))
            discharge_kwh += max(0.0, e_served)
        storage_energy_totals_by_station[s] = {
            "storage_total_charge_kwh": charge_kwh,
            "storage_total_discharge_kwh": discharge_kwh,
            "storage_net_discharge_kwh": max(0.0, discharge_kwh - charge_kwh),
        }
    diagnostics["terminal_soc_by_station"] = terminal_soc_by_station
    diagnostics["storage_energy_totals_by_station"] = storage_energy_totals_by_station
    diagnostics["storage_model_status_by_station"] = storage_model_status_by_station
    diagnostics["storage_soc_trajectory_kwh"] = B_vt_report
    diagnostics["storage_charge_power_kw"] = {s: {t: float((P_vt_lp or {}).get(s, {}).get(t, 0.0)) for t in times} for s in power_stations}

    if positive_mu_points > 0:
        diagnostics["bottleneck_summary"] = "mixed_or_power_constrained"
    elif stressed_power_points > 0:
        diagnostics["bottleneck_summary"] = "mixed_with_stressed_power_usage_but_no_shadow_price_pass_through"
    elif float(diagnostics.get("aircraft_binding_count", 0.0)) > 0:
        diagnostics["bottleneck_summary"] = "aircraft_inventory_or_turnaround_dominant_not_shared_power"
    else:
        diagnostics["bottleneck_summary"] = "mixed_without_active_shared_power_price_signal"

    grid_station_caps = grid_results.get("station_grid_available_power", {}) if isinstance(grid_results, dict) else {}
    station_power_cap_effective = data.get("diagnostics_runtime", {}).get("station_power_cap_effective", {})
    num_stations_grid_limited = sum(
        1
        for s in ev_stations
        if any(
            float(grid_station_caps.get(s, {}).get(t, station_params[s]["P_site"][t])) + 1.0e-9 < float(station_params[s]["P_site"][t])
            and float(load_agg.get("P_total_req", {}).get(s, {}).get(t, 0.0)) > float(grid_station_caps.get(s, {}).get(t, station_params[s]["P_site"][t])) + 1.0e-9
            for t in times
        )
    )
    branch_loading_map = grid_results.get("branch_loading_ratio", {}) if isinstance(grid_results, dict) else {}
    bus_voltage_map = grid_results.get("bus_voltage", {}) if isinstance(grid_results, dict) else {}
    grid_shadow_map = grid_results.get("station_grid_shadow_price", {}) if isinstance(grid_results, dict) else {}
    max_branch_loading_ratio = max((float(v) for br in branch_loading_map.values() for v in br.values()), default=0.0)
    min_bus_voltage = min((float(v) for bus in bus_voltage_map.values() for v in bus.values()), default=1.0)
    max_grid_shadow_price = max((float(v) for st in grid_shadow_map.values() for v in st.values()), default=0.0)

    # Recompute final exported price objects from the converged end-of-loop state so
    # output/report prices exactly match the final grid + local surcharge state.
    effective_prices = {
        s: {
            t: float(electricity_price[s][t])
            + float(grid_shadow_map.get(s, {}).get(t, 0.0))
            + float(energy_surcharge.get(s, {}).get(t, 0.0))
            for t in times
        }
        for s in ev_stations
    }
    effective_price_components = {
        s: {
            t: {
                "base_price": float(electricity_price[s][t]),
                "grid_component": float(grid_shadow_map.get(s, {}).get(t, 0.0)),
                "local_component": float(energy_surcharge.get(s, {}).get(t, 0.0)),
                "effective_price": float(effective_prices[s][t]),
            }
            for t in times
        }
        for s in ev_stations
    }

    results = {
        "x": arc_flows,
        "tau": tau,
        "n": n_series,
        "g": g_series,
        "u": utilization,
        "w": ev_station_waits,
        "f": flows,
        "costs": costs,
        "generalized_costs": logit_details.get("generalized_costs", {}),
        "generalized_costs_raw": logit_details.get("generalized_costs_raw", {}),
        "utilities": logit_details.get("utilities", {}),
        "d_vt_route": d_vt_route,
        "d_vt_dep": d_vt_dep,
        "e_dep": e_dep,
        "inventory": {"B_vt": B_vt, "P_vt": P_vt, "residuals": inv_residuals},
        "shadow_price_power": shadow_prices,
        "distribution_grid": grid_results,
        "effective_price": effective_prices,
        "effective_price_components": effective_price_components,
        "surcharge_power": energy_surcharge,
        "surcharge_power_raw": raw_surcharge,
        "surcharge_power_uncapped": raw_surcharge_uncapped,
        "shared_power": {
            "B_vt": B_vt_lp,
            "P_vt": P_vt_lp,
            "shed_ev": shed_ev,
            "shed_vt": shed_vt,
            "residuals": shared_power_residuals,
        },
        "validation": {
            "cap_violation": cap_violation,
            "power_violation": power_violation_effective_cap,
            "power_violation_site_raw": power_violation_site_raw,
            "power_violation_effective_cap": power_violation_effective_cap,
        },
        "residuals": residuals,
        "convergence": {
            "dx": dx,
            "dn": dn,
            "dprice": dprice,
            "dprice_raw": dprice_for_conv,
            "dprice_gate": dprice_for_conv,
            "iterations": last_iteration,
        },
        "diagnostics": {
            **diagnostics,
            "stop_reason": stop_reason,
            "boundary_inflow": {t: inflow_total[idx] for idx, t in enumerate(times)},
            "boundary_outflow": {t: outflow_total[idx] for idx, t in enumerate(times)},
            "boundary_inflow_model": {t: model_inflow_total[idx] for idx, t in enumerate(times)},
            "boundary_outflow_model": {t: model_outflow_total[idx] for idx, t in enumerate(times)},
            "boundary_inflow_background": {t: bg_inflow_total[idx] for idx, t in enumerate(times)},
            "boundary_outflow_background": {t: bg_outflow_total[idx] for idx, t in enumerate(times)},
            "n_series": {idx: n_series[idx] for idx in range(len(n_series))},
            "g_series": {idx: g_series[idx] for idx in range(len(g_series))},
            "g_state_series": [float(v) for v in g_series],
            "g_used_by_time": g_used_by_time,
            "cbd_tau": cbd_tau,
            "cbd_tau_used": cbd_tau,
            "station_loads": load_agg,
            "distribution_grid": grid_results,
            "effective_price": effective_prices,
            "effective_price_components": effective_price_components,
            "vt_service_prob": vt_service_prob,
            "ev_service_prob": ev_service_prob,
            "vt_inventory_B": B_vt_report,
            "vt_charge_power_P": P_vt_lp,
            "shared_power_lp": shared_power_lp_diag,
            "distribution_grid_enabled": grid_results.get("distribution_grid_enabled", False),
            "station_to_bus": grid_results.get("station_to_bus", {}),
            "bus_voltage": grid_results.get("bus_voltage", {}),
            "branch_flow_p": grid_results.get("branch_flow_p", {}),
            "branch_loading_ratio": grid_results.get("branch_loading_ratio", {}),
            "substation_loading": grid_results.get("substation_loading", {}),
            "station_grid_available_power": grid_results.get("station_grid_available_power", {}),
            "station_grid_shadow_price": grid_results.get("station_grid_shadow_price", {}),
            "station_power_cap_effective": station_power_cap_effective,
            "power_violation_site_raw": power_violation_site_raw,
            "power_violation_effective_cap": power_violation_effective_cap,
            "grid_binding_count": grid_results.get("grid_binding_count", 0),
            "num_stations_grid_limited": num_stations_grid_limited,
            "max_branch_loading_ratio": max_branch_loading_ratio,
            "min_bus_voltage": min_bus_voltage,
            "max_grid_shadow_price": max_grid_shadow_price,
            "voltage_violation_count": grid_results.get("voltage_violation_count", 0),
            "branch_overload_count": grid_results.get("branch_overload_count", 0),
            "substation_binding_count": grid_results.get("substation_binding_count", 0),
            "missing_inventory_reporting": missing_inventory_reporting,
            "iter_count_total": last_iteration,
            "max_iter_effective": current_max_iter,
            "tol": config.get("tol"),
            "dx_end": dx,
            "dn_end": dn,
            "dprice_smoothed_end": dprice,
            "dprice_raw_end": dprice_for_conv,
            "dprice_gate_end": dprice_for_conv,
            "dprice_end": dprice_for_conv,
            "converged_loose": conv_flags["converged_loose"],
            "converged_strictly": conv_flags["converged_strictly"],
            "audit_ok": False,
            "max_residual_end": conv_flags["max_residual_end"],
            "patience_only_stop": conv_flags["patience_only_stop"],
            "stopped_by_patience_only": conv_flags["patience_only_stop"],
            "audit_has_warning": False,
        },
        "data_parameters": {**data.get("parameters", {}), "expected_groups": list(data.get("sets", {}).get("groups", []))},
        "meta": data.get("meta", {}),
        "charging": {
            "E": E,
            "p_ch": p_ch,
            "y": y,
            "residuals": charging_residuals,
        },
        "solver_used": charging.LAST_SOLVER_USED if run_dispatch else "aggregate",
    }
    audit = self_audit(results, config)
    results["SelfAudit"] = audit
    results["diagnostics"]["audit_ok"] = bool(audit.get("ok", False))
    results["diagnostics"]["audit_has_warning"] = bool(audit.get("warnings"))
    must_raise_tokens = [
        "Requested HiGHS but non-HiGHS solver used",
        "inventory balance broken",
        "vt_service_prob out of range",
        "unexpected group label",
    ]
    if any(any(tok in s for tok in must_raise_tokens) for s in audit.get("severe", [])):
        raise ValueError("Critical self-audit failure: " + "; ".join(audit.get("severe", [])))
    if bool(config.get("strict_audit", True)):
        diagnostics["audit_warnings"] = audit.get("warnings", [])
        if audit.get("severe") and not bool(config.get("audit_raise", False)):
            raise ValueError("Strict audit failed: " + "; ".join(audit.get("severe", [])))
    return results, residuals


def save_outputs(results: Dict[str, Any], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def _resolve_path(path: str, fallback: str) -> str:
    import os

    module_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = module_dir
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(module_dir, path),
    ]
    candidates.extend([
        os.path.join(repo_root, path),
        os.path.join(repo_root, "project", "data", os.path.basename(path)),
        os.path.join(repo_root, "project", os.path.basename(path)),
        fallback,
        os.path.join(repo_root, fallback),
        os.path.join(module_dir, os.path.basename(path)),
        os.path.join(module_dir, os.path.basename(fallback)),
    ])
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find required file '{path}'. Tried: {candidates}. "
        "Check working directory, CLI path, and project/data_schema.yaml location."
    )


def main() -> None:
    import argparse
    import datetime
    import platform
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="simple_case.yaml", help="Path to data YAML")
    parser.add_argument("--schema", default="data_schema.yaml", help="Path to schema YAML")
    parser.add_argument("--dispatch", action="store_true", help="Run vehicle-level dispatch module")
    parser.add_argument("--report-out", "--report_out", dest="report_out", default="report.json", help="Path to report JSON output")
    parser.add_argument("--full_report", action="store_true", help="Write full report payload")
    parser.add_argument("--full-json", dest="full_json", action=argparse.BooleanOptionalAction, default=True, help="Emit full diagnostics JSON")
    parser.add_argument("data_pos", nargs="?", default=None, help="Optional positional data path")
    args = parser.parse_args()

    try:
        data_arg = args.data
        if args.data_pos:
            data_arg = args.data_pos
        data_path = _resolve_path(data_arg, "project/data/simple_case.yaml")
        schema_path = _resolve_path(args.schema, "project/data_schema.yaml")
        data = load_data(data_path, schema_path)

        results, residuals = run_equilibrium(data, run_dispatch=args.dispatch)
        save_outputs(results, "project/output.json")
        dprice_hist = results["diagnostics"].get("dprice_history", [])
        max_surcharge = max(v for m in results["surcharge_power"].values() for v in m.values())
        cbd_tau = results["diagnostics"].get("cbd_tau", {})
        cbd_tau_one = {}
        if cbd_tau:
            first_arc = next(iter(cbd_tau.keys()))
            cbd_tau_one = {first_arc: cbd_tau[first_arc]}
        peak_snapshot = results["diagnostics"].get("price_snapshots", [])
        peak_prices_last = peak_snapshot[-1] if peak_snapshot else {}
        mode_share_last = results["diagnostics"].get("mode_share_by_group", [])
        mode_share_last = mode_share_last[-1] if mode_share_last else {}
        output_full_json = bool(data.get("config", {}).get("output_full_json", True))
        if args.full_report:
            output_full_json = True
        output_full_json = bool(output_full_json and args.full_json)

        git_commit = None
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            git_commit = None

        equilibrium = {
            "C1": residuals.get("C1", 0.0),
            "C2": residuals.get("C2", 0.0),
            "C2_rel": residuals.get("C2_rel", 0.0),
            "C2_raw": residuals.get("C2_raw", 0.0),
            "C7": residuals.get("C7", 0.0),
            "C8": residuals.get("C8", 0.0),
            "C9": residuals.get("C9", 0.0),
            "C10": residuals.get("C10", 0.0),
            "C10_raw": residuals.get("C10_raw", 0.0),
            "C11": residuals.get("C11", 0.0),
        }

        diagnostics_full = {**results.get("diagnostics", {})}
        diagnostics_full.update({
            "iter_count": results["convergence"].get("iterations"),
            "dx_end": results["convergence"].get("dx"),
            "dn_end": results["convergence"].get("dn"),
            "dprice_start": dprice_hist[0] if dprice_hist else None,
            "dprice_smoothed_end": dprice_hist[-1] if dprice_hist else None,
            "dprice_raw_end": results.get("diagnostics", {}).get("dprice_raw_end"),
            "dprice_end": results.get("diagnostics", {}).get("dprice_gate_end"),
            "dprice_definition": "dprice_end uses gate/raw delta (same metric used in stop criterion); dprice_smoothed_end is post-MSA delta",
            "max_surcharge": max_surcharge,
            "peak_t": peak_prices_last.get("t"),
            "peak_prices": peak_prices_last.get("stations", {}),
            "objective_by_station_time": results.get("diagnostics", {}).get("shared_power_lp", {}).get("objective_components", {}),
            "objective_totals": results.get("diagnostics", {}).get("shared_power_lp", {}).get("objective_totals", {}),
            "dual_trace": results.get("diagnostics", {}).get("shared_power_lp", {}).get("dual_trace", {}),
        })

        diagnostics_summary = {
            "stop_reason": diagnostics_full.get("stop_reason"),
            "iter_count": diagnostics_full.get("iter_count"),
            "dx_end": diagnostics_full.get("dx_end"),
            "dn_end": diagnostics_full.get("dn_end"),
            "dprice_end": diagnostics_full.get("dprice_end"),
            "dprice_raw_end": diagnostics_full.get("dprice_raw_end"),
            "dprice_smoothed_end": diagnostics_full.get("dprice_smoothed_end"),
            "dprice_definition": diagnostics_full.get("dprice_definition"),
            "max_surcharge": diagnostics_full.get("max_surcharge"),
            "peak_t": diagnostics_full.get("peak_t"),
            "peak_prices": diagnostics_full.get("peak_prices"),
            "converged_loose": diagnostics_full.get("converged_loose"),
            "converged_strictly": diagnostics_full.get("converged_strictly"),
            "shared_power_solver_used": results["diagnostics"].get("shared_power_solver_used"),
            "delta_t": data.get("meta", {}).get("delta_t"),
            "cbd_tau": cbd_tau_one,
        }

        report = {
            "meta": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "git_commit_if_available": git_commit,
                "python_version": sys.version,
                "platform": platform.platform(),
                "module_paths": diagnostics_full.get("module_paths", {}),
            },
            "config_used": data.get("config", {}),
            "equilibrium": equilibrium,
            "effective_price": results.get("effective_price", {}),
            "effective_price_components": results.get("effective_price_components", {}),
            "distribution_grid": results.get("distribution_grid", {}),
            "summary": {
                "representative_od": diagnostics_full.get("representative_od"),
                "case_label": diagnostics_full.get("case_label"),
                "convergence": {
                    "stop_reason": diagnostics_summary.get("stop_reason"),
                    "converged_loose": diagnostics_summary.get("converged_loose"),
                    "converged_strictly": diagnostics_summary.get("converged_strictly"),
                    "dx_end": diagnostics_summary.get("dx_end"),
                    "dn_end": diagnostics_summary.get("dn_end"),
                    "dprice_end": diagnostics_summary.get("dprice_end"),
                    "dprice_raw_end": diagnostics_summary.get("dprice_raw_end"),
                    "dprice_smoothed_end": diagnostics_summary.get("dprice_smoothed_end"),
                    "dprice_definition": diagnostics_summary.get("dprice_definition"),
                },
                "bottleneck_summary": diagnostics_full.get("bottleneck_summary"),
                "power_binding_count": diagnostics_full.get("power_binding_count"),
                "max_surcharge": diagnostics_summary.get("max_surcharge"),
                "distribution_grid_enabled": diagnostics_full.get("distribution_grid_enabled"),
                "grid_binding_count": diagnostics_full.get("grid_binding_count"),
                "num_stations_grid_limited": diagnostics_full.get("num_stations_grid_limited"),
                "max_branch_loading_ratio": diagnostics_full.get("max_branch_loading_ratio"),
                "min_bus_voltage": diagnostics_full.get("min_bus_voltage"),
                "max_grid_shadow_price": diagnostics_full.get("max_grid_shadow_price"),
            },
            "surcharge_power": results["surcharge_power"],
            "surcharge_power_uncapped": results.get("surcharge_power_uncapped", {}),
            "diagnostics": diagnostics_summary,
            "SelfAudit": results.get("SelfAudit", {}),
        }

        if output_full_json:
            report["diagnostics_detail"] = diagnostics_full
            report["diagnostics_station_detail"] = {
                "station_sets": diagnostics_full.get("station_sets", {}),
                "power_tightness": diagnostics_full.get("power_tightness", {}),
                "vt_departure_waits": diagnostics_full.get("vt_departure_waits", {}),
                "vt_departure_flow_by_class": diagnostics_full.get("vt_departure_flow_by_class", {}),
                "terminal_soc_by_station": diagnostics_full.get("terminal_soc_by_station", {}),
                "storage_energy_totals_by_station": diagnostics_full.get("storage_energy_totals_by_station", {}),
                "storage_model_status_by_station": diagnostics_full.get("storage_model_status_by_station", {}),
                "effective_price": diagnostics_full.get("effective_price", {}),
                "effective_price_components": diagnostics_full.get("effective_price_components", {}),
            }

        report["diagnostics"]["surcharge_kappa"] = data.get("config", {}).get("surcharge_kappa")
        report["diagnostics"]["shadow_price_scale"] = data.get("config", {}).get("shadow_price_scale")
        report["diagnostics"]["surcharge_beta"] = data.get("config", {}).get("surcharge_beta")
        report["diagnostics"]["surcharge_cap_mult"] = data.get("config", {}).get("shadow_price_cap_mult")
        report["diagnostics"]["VOLL_EV"] = data.get("config", {}).get("voll_ev_per_kwh")
        report["diagnostics"]["VOLL_VT"] = data.get("config", {}).get("voll_vt_per_kwh")
        report["diagnostics"]["vt_reliability_gamma"] = data.get("config", {}).get("vt_reliability_gamma")
        report = _clean_nested_numbers(report)

    except Exception as exc:
        report = {
            "meta": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "git_commit_if_available": None,
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            "config_used": {},
            "equilibrium": {},
            "surcharge_power": {},
            "surcharge_power_uncapped": {},
            "diagnostics": {
                "module_paths": {
                    "runner_file": str(Path(__file__).resolve()),
                    "charging_file": str(Path(charging.__file__).resolve()),
                    "sys_path_head": list(sys.path[:5]),
                },
                "scipy_version": charging.SCIPY_VERSION,
                "scipy_ok": bool(charging.HAS_SCIPY),
            },
            "SelfAudit": {"ok": False, "warnings": [], "severe": [repr(exc)], "cap_binding": False, "has_negative_n": False},
        }
        payload = json.dumps(_clean_nested_numbers(report), ensure_ascii=False, indent=2)
        with open(args.report_out, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print(payload)
        raise

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        handle.write(payload)
    print(
        f"SUMMARY iter_count={report['diagnostics'].get('iter_count')} "
        f"max_surcharge={report['diagnostics'].get('max_surcharge')} "
        f"peak_t={report['diagnostics'].get('peak_t')} solver={report['diagnostics'].get('shared_power_solver_used')}"
    )
    print(f"Wrote {args.report_out} (bytes={len(payload.encode('utf-8'))})")
    print(payload)


if __name__ == "__main__":
    main()
