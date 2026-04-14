import math
from typing import Any, Dict, List, Tuple

from .utils import logsumexp


EVTOL_MODES = {"eVTOL", "eVTOL_fast", "eVTOL_slow", "EV_to_eVTOL_fast", "EV_to_eVTOL_slow"}
MULTIMODAL_MODES = {"EV_to_eVTOL_fast", "EV_to_eVTOL_slow"}


def _value_by_time(x: Any, t: int, default: float = 0.0) -> float:
    if isinstance(x, dict):
        if t in x:
            return float(x[t])
        ts = str(t)
        if ts in x:
            return float(x[ts])
        return float(default)
    if x is None:
        return float(default)
    return float(x)


def is_multimodal_evtol(it: Dict[str, Any]) -> bool:
    return str(it.get("mode", "")) in MULTIMODAL_MODES


def is_pure_evtol(it: Dict[str, Any]) -> bool:
    return str(it.get("mode", "")) in {"eVTOL", "eVTOL_fast", "eVTOL_slow"}


def is_evtol_itinerary(it: Dict[str, Any]) -> bool:
    mode = str(it.get("mode", ""))
    if mode in EVTOL_MODES:
        return True
    return str(it.get("service_class", "")).lower() in {"fast", "slow"}


def get_evtol_service_class(it: Dict[str, Any]) -> str:
    mode = str(it.get("mode", ""))
    if mode.endswith("_fast"):
        return "fast"
    if mode.endswith("_slow"):
        return "slow"
    cls = str(it.get("service_class", "")).lower()
    if cls in {"fast", "slow"}:
        return cls
    return "slow"




def classify_mode_label(it: Dict[str, Any]) -> str:
    if is_multimodal_evtol(it):
        cls = get_evtol_service_class(it)
        return f"multimodal_EV_to_eVTOL_{cls}"
    mode = str(it.get("mode", ""))
    if mode == "EV":
        return "pure_EV"
    if is_evtol_itinerary(it):
        cls = get_evtol_service_class(it)
        return f"pure_eVTOL_{cls}"
    return "pure_EV"


def classify_supermode(it: Dict[str, Any]) -> str:
    if is_multimodal_evtol(it):
        return "EV_to_eVTOL"
    if is_evtol_itinerary(it):
        return "eVTOL"
    return "EV"

def _road_segments(it: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = []
    segs.extend(it.get("road_arcs", []))
    segs.extend(it.get("access_arcs", []))
    segs.extend(it.get("egress_arcs", []))
    return segs


def _ev_stops(it: Dict[str, Any]) -> List[Dict[str, Any]]:
    stops = []
    if str(it.get("mode")) == "EV":
        stops.extend(it.get("stations", []))
    if is_multimodal_evtol(it):
        stops.extend(it.get("access_stations", []))
        if not it.get("access_stations") and it.get("stations"):
            stops.extend(it.get("stations", []))
    return stops


def _ev_stops_at_time(it: Dict[str, Any], t: int) -> List[Dict[str, Any]]:
    """EV charging stops at time t, including implicit multimodal access fallback.

    Fallback rule (Scheme A):
    - For EV_to_eVTOL, if scalar access_energy_kwh is present at t and explicit
      access_stations energy is not present at t, dep_station is treated as an
      implicit EV stop for wait/utilization/reliability/charging accounting.
    """
    out: List[Dict[str, Any]] = []
    for stop in _ev_stops(it):
        if stop.get("t") == t:
            out.append(stop)
    if is_multimodal_evtol(it):
        explicit_access_energy, scalar_access_energy = _access_energy_for_time(it, t)
        dep_station = it.get("dep_station")
        has_explicit_at_t = any(st.get("t") == t for st in (it.get("access_stations", []) or []))
        if scalar_access_energy > 1.0e-12 and explicit_access_energy <= 1.0e-12 and not has_explicit_at_t and dep_station is not None:
            out.append({"station": dep_station, "t": t, "energy": scalar_access_energy, "implicit_access_fallback": True})
    return out




def _access_energy_for_time(it: Dict[str, Any], t: int) -> Tuple[float, float]:
    """Return (explicit_access_station_energy, scalar_access_energy_kwh) per pax at time t."""
    explicit = 0.0
    for stop in it.get("access_stations", []) or []:
        if stop.get("t") != t:
            continue
        explicit += float(stop.get("energy", 0.0) or 0.0)
    scalar = float(_value_by_time(it.get("access_energy_kwh", 0.0), t, 0.0) or 0.0)
    return explicit, scalar
def _access_station_ids(it: Dict[str, Any], t: int | None = None) -> List[str]:
    access = it.get("access_stations", [])
    out: List[str] = []
    for st in access:
        if t is not None and st.get("t") != t:
            continue
        sid = st.get("station")
        if sid is not None:
            out.append(str(sid))
    if out:
        return out
    for st in access:
        sid = st.get("station")
        if sid is not None:
            out.append(str(sid))
    return out


def _access_mode(it: Dict[str, Any]) -> str:
    raw = it.get("access_mode")
    if raw is not None:
        return str(raw)
    if is_multimodal_evtol(it):
        return "private_ev_hub_charging_access"
    if is_evtol_itinerary(it):
        return "external_noncharging_access"
    return "no_access"


def build_incidence(
    itineraries: List[Dict[str, Any]],
    arcs: List[str],
    stations: List[str],
    times: List[int],
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, float]]]]:
    arc_set = set(arcs)
    station_set = set(stations)
    time_set = set(times)
    inc_road = {arc: {it["id"]: {t: 0.0 for t in times} for it in itineraries} for arc in arcs}
    # NOTE: inc_station is EV-related station utilization only.
    # eVTOL departure waiting is modeled separately via vt_departure_waits
    # and is intentionally excluded from C10 consistency checks.
    inc_station = {s: {it["id"]: {t: 0.0 for t in times} for it in itineraries} for s in stations}
    for it in itineraries:
        it_id = it.get("id", "<unknown>")
        for seg in _road_segments(it):
            arc = seg["arc"]
            t = seg["t"]
            if arc not in arc_set:
                raise KeyError(f"Unknown arc in itinerary {it_id}: arc={arc}")
            if t not in time_set:
                raise KeyError(f"Unknown time in itinerary {it_id}: t={t}")
            frac = seg.get("frac", 1.0)
            inc_road[arc][it["id"]][t] += frac
        # Keep incidence consistent with implicit-access-stop semantics.
        for t in times:
            for stop in _ev_stops_at_time(it, t):
                station = stop["station"]
                if station not in station_set:
                    raise KeyError(f"Unknown station in itinerary {it_id}: station={station}")
                if t not in time_set:
                    raise KeyError(f"Unknown time in itinerary {it_id}: t={t}")
                inc_station[station][it["id"]][t] += 1.0
    return inc_road, inc_station


def compute_itinerary_costs(
    itineraries: List[Dict[str, Any]],
    travel_times: Dict[str, Dict[int, float]],
    ev_station_waits: Dict[str, Dict[int, float]],
    electricity_price: Dict[str, Dict[int, float]],
    times: List[int],
    vt_departure_waits: Dict[str, Dict[str, Dict[int, float]]] | None = None,
    transfer_time_by_station: Dict[str, float] | Dict[str, Dict[int, float]] | None = None,
    transfer_time_default: float = 0.0,
    prefer_transfer_time_by_station: bool = False,
    multimodal_penalty_cfg: Dict[str, Any] | None = None,
    vt_service_prob: Dict[str, Dict[int, float]] | None = None,
    ev_service_prob: Dict[str, Dict[int, float]] | None = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    costs: Dict[str, Dict[int, Dict[str, float]]] = {it["id"]: {} for it in itineraries}
    for it in itineraries:
        it_id = it.get("id", "<unknown>")
        phi_markup = float(it.get("phi_energy_markup", 1.0))
        dep_station = it.get("dep_station")
        access_mode = _access_mode(it)
        for t in times:
            money_t = _value_by_time(it.get("money", 0.0), t, 0.0)
            tt = 0.0
            charge_cost = 0.0
            access_travel_time_applied = 0.0
            ev_access_wait_applied = 0.0
            flight_time_applied = 0.0
            for seg in _road_segments(it):
                if seg["t"] != t:
                    continue
                arc = seg["arc"]
                if arc not in travel_times or t not in travel_times[arc]:
                    raise KeyError(f"Missing travel_times for itinerary {it_id}, arc={arc}, t={t}")
                tt += float(travel_times[arc][t]) * float(seg.get("frac", 1.0))
            for seg in (it.get("access_arcs", []) or []):
                if seg.get("t") != t:
                    continue
                arc = seg.get("arc")
                if arc in travel_times and t in travel_times[arc]:
                    access_travel_time_applied += float(travel_times[arc][t]) * float(seg.get("frac", 1.0))

            # EV component (pure EV or multimodal access)
            ev_stops_t = _ev_stops_at_time(it, t) if access_mode == "private_ev_hub_charging_access" else []
            has_implicit_access_fallback = any(bool(st.get("implicit_access_fallback", False)) for st in ev_stops_t)
            for stop in ev_stops_t:
                station = stop.get("station")
                if station not in ev_station_waits or t not in ev_station_waits[station]:
                    raise KeyError(f"Missing ev_station_waits for itinerary {it_id}, station={station}, t={t}")
                if station not in electricity_price or t not in electricity_price[station]:
                    raise KeyError(f"Missing electricity_price for itinerary {it_id}, station={station}, t={t}")
                wait_val = float(ev_station_waits[station][t])
                tt += wait_val
                if is_multimodal_evtol(it):
                    ev_access_wait_applied += wait_val
                charge_cost += float(stop.get("energy", 0.0)) * float(electricity_price[station][t])

            transfer_time_applied = 0.0
            transfer_time_source = "none"
            access_energy_price_source = "none"
            multimodal_extra_penalty = 0.0
            vt_wait_applied = 0.0
            multimodal_penalty_components = {
                "base_transfer_fragility_penalty": 0.0,
                "ev_unreliability_term": 0.0,
                "vt_unreliability_term": 0.0,
                "joint_unreliability_term": 0.0,
                "transfer_overrun_term": 0.0,
                "ev_prob_used": 1.0,
                "vt_prob_used": 1.0,
            }
            transfer_buffer_threshold = 0.0
            preflight_transfer_burden = 0.0
            transfer_overrun = 0.0

            # Optional scalar access energy (multimodal).
            # If access_stations already provide per-station energy for this time, scalar access_energy_kwh
            # is treated as redundant metadata and not re-charged to avoid double counting.
            explicit_access_energy, scalar_access_energy = _access_energy_for_time(it, t) if access_mode == "private_ev_hub_charging_access" else (0.0, 0.0)
            access_energy_kwh = scalar_access_energy if explicit_access_energy <= 1.0e-12 and not has_implicit_access_fallback else 0.0
            access_energy_consistency = "ok"
            if explicit_access_energy > 1.0e-12 and scalar_access_energy > 1.0e-12:
                rel_gap = abs(explicit_access_energy - scalar_access_energy) / max(1.0e-6, max(explicit_access_energy, scalar_access_energy))
                access_energy_consistency = "duplicate_field_used_explicit" if rel_gap <= 0.15 else "inconsistent_duplicate_field_used_explicit"
            if access_energy_kwh > 0.0:
                access_station_ids = _access_station_ids(it, t)
                if access_station_ids:
                    prices = []
                    for sid in access_station_ids:
                        if sid not in electricity_price or t not in electricity_price[sid]:
                            raise KeyError(f"Missing electricity_price for itinerary {it_id}, access_station={sid}, t={t}")
                        prices.append(float(electricity_price[sid][t]))
                    if prices:
                        charge_cost += access_energy_kwh * (sum(prices) / len(prices))
                        access_energy_price_source = "access_station_mean"
                elif dep_station is not None and dep_station in electricity_price and t in electricity_price[dep_station]:
                    charge_cost += access_energy_kwh * float(electricity_price[dep_station][t])
                    access_energy_price_source = "dep_station_fallback"

            if is_evtol_itinerary(it) and dep_station is not None:
                flight_time = _value_by_time(it.get("flight_time", {}), t, 0.0)
                if flight_time <= 0.0:
                    costs[it_id][t] = {
                        "TT": float("inf"),
                        "Money": float("inf"),
                        "ChargeCost": 0.0,
                        "ContinuityPenalty": 0.0,
                        "cost_breakdown": {
                            "access_mode": access_mode,
                            "access_charging_accounted": bool(access_mode == "private_ev_hub_charging_access"),
                            "transfer_time_applied": 0.0,
                            "transfer_time_source": "none",
                            "access_energy_price_source": access_energy_price_source,
                            "access_energy_consistency": access_energy_consistency,
                            "access_travel_time_applied": access_travel_time_applied,
                            "ev_access_wait_applied": ev_access_wait_applied,
                            "transfer_processing_time_applied": 0.0,
                            "preflight_transfer_burden": 0.0,
                            "transfer_buffer_limit_applied": 0.0,
                            "transfer_overrun": 0.0,
                            "transfer_overrun_term": 0.0,
                            "vt_departure_wait_applied": 0.0,
                            "flight_time_applied": 0.0,
                            "continuity_penalty": 0.0,
                            "continuity_penalty_components": multimodal_penalty_components,
                        },
                    }
                    continue
                if is_multimodal_evtol(it):
                    if prefer_transfer_time_by_station and dep_station is not None and transfer_time_by_station and dep_station in transfer_time_by_station:
                        transfer_time_applied = _value_by_time(transfer_time_by_station.get(dep_station, 0.0), t, 0.0)
                        transfer_time_source = "hub_endogenous"
                    elif "transfer_time" in it:
                        transfer_time_applied = _value_by_time(it.get("transfer_time", 0.0), t, 0.0)
                        transfer_time_source = "itinerary"
                    elif dep_station is not None and transfer_time_by_station and dep_station in transfer_time_by_station:
                        transfer_time_applied = _value_by_time(transfer_time_by_station.get(dep_station, 0.0), t, 0.0)
                        transfer_time_source = "station_default"
                    elif transfer_time_default > 0.0:
                        transfer_time_applied = float(transfer_time_default)
                        transfer_time_source = "global_default"
                    tt += transfer_time_applied
                tt += flight_time
                flight_time_applied = flight_time
                svc_class = get_evtol_service_class(it)
                if vt_departure_waits is not None:
                    if dep_station not in vt_departure_waits or svc_class not in vt_departure_waits[dep_station] or t not in vt_departure_waits[dep_station][svc_class]:
                        raise KeyError(f"Missing vt_departure_waits for itinerary {it_id}, dep_station={dep_station}, class={svc_class}, t={t}")
                    vt_wait_applied = float(vt_departure_waits[dep_station][svc_class][t])
                    tt += vt_wait_applied
                e_per_pax = _value_by_time(it.get("e_per_pax", 0.0), t, 0.0)
                if dep_station not in electricity_price or t not in electricity_price[dep_station]:
                    raise KeyError(f"Missing electricity_price for itinerary {it_id}, dep_station={dep_station}, t={t}")
                money_t += phi_markup * e_per_pax * float(electricity_price[dep_station][t])
                if is_multimodal_evtol(it) and multimodal_penalty_cfg:
                    base_penalty = float(multimodal_penalty_cfg.get("base_transfer_fragility_penalty", 0.0) or 0.0)
                    coeff_ev_unrel = float(multimodal_penalty_cfg.get("coeff_ev_unreliability", 0.0) or 0.0)
                    coeff_vt_unrel = float(multimodal_penalty_cfg.get("coeff_vt_unreliability", 0.0) or 0.0)
                    coeff_joint_unrel = float(multimodal_penalty_cfg.get("coeff_joint_unreliability", 0.0) or 0.0)
                    vt_prob = 1.0
                    if vt_service_prob and dep_station in vt_service_prob:
                        vt_prob = min(1.0, max(0.0, float(vt_service_prob[dep_station].get(t, 1.0))))
                    ev_prob = 1.0
                    if ev_service_prob is not None:
                        ev_candidates = []
                        for stop in _ev_stops_at_time(it, t):
                            station = stop.get("station")
                            if station in ev_service_prob:
                                ev_candidates.append(float(ev_service_prob[station].get(t, 1.0)))
                        if ev_candidates:
                            ev_prob = min(1.0, max(0.0, min(ev_candidates)))
                    ev_unrel = 1.0 - ev_prob
                    vt_unrel = 1.0 - vt_prob
                    ev_term = coeff_ev_unrel * ev_unrel
                    vt_term = coeff_vt_unrel * vt_unrel
                    joint_term = coeff_joint_unrel * ev_unrel * vt_unrel
                    coeff_transfer_overrun = float(multimodal_penalty_cfg.get("coeff_transfer_overrun", 0.0) or 0.0)
                    transfer_buffer_threshold = float(multimodal_penalty_cfg.get("transfer_buffer_threshold", 0.0) or 0.0)
                    preflight_transfer_burden = ev_access_wait_applied + transfer_time_applied + vt_wait_applied
                    transfer_overrun = max(0.0, preflight_transfer_burden - transfer_buffer_threshold)
                    transfer_overrun_term = coeff_transfer_overrun * transfer_overrun
                    multimodal_extra_penalty = (
                        base_penalty
                        + ev_term
                        + vt_term
                        + joint_term
                        + transfer_overrun_term
                    )
                    multimodal_penalty_components = {
                        "base_transfer_fragility_penalty": base_penalty,
                        "ev_unreliability_term": ev_term,
                        "vt_unreliability_term": vt_term,
                        "joint_unreliability_term": joint_term,
                        "transfer_overrun_term": transfer_overrun_term,
                        "ev_prob_used": ev_prob,
                        "vt_prob_used": vt_prob,
                    }

            costs[it_id][t] = {
                "TT": tt,
                "Money": money_t,
                "ChargeCost": charge_cost,
                "ContinuityPenalty": multimodal_extra_penalty,
                "cost_breakdown": {
                    "access_mode": access_mode,
                    "access_charging_accounted": bool(access_mode == "private_ev_hub_charging_access"),
                    "transfer_time_applied": transfer_time_applied,
                    "transfer_time_source": transfer_time_source,
                    "access_energy_price_source": access_energy_price_source,
                    "access_energy_consistency": access_energy_consistency,
                    "access_travel_time_applied": access_travel_time_applied,
                    "ev_access_wait_applied": ev_access_wait_applied,
                    "transfer_processing_time_applied": transfer_time_applied,
                    "preflight_transfer_burden": preflight_transfer_burden,
                    "transfer_buffer_limit_applied": transfer_buffer_threshold,
                    "transfer_overrun": transfer_overrun,
                    "transfer_overrun_term": float(multimodal_penalty_components.get("transfer_overrun_term", 0.0)),
                    "continuity_penalty": multimodal_extra_penalty,
                    "vt_departure_wait_applied": vt_wait_applied,
                    "flight_time_applied": flight_time_applied,
                    "continuity_penalty_components": multimodal_penalty_components,
                },
            }
    return costs


def logit_assignment(
    itineraries: List[Dict[str, Any]],
    costs: Dict[str, Dict[int, Dict[str, float]]],
    demand: Dict[str, Dict[str, Dict[int, float]]],
    vot: Dict[str, Dict[int, float]],
    lambdas: Dict[str, float],
    times: List[int],
    vt_service_prob: Dict[str, Dict[int, float]] | None = None,
    ev_service_prob: Dict[str, Dict[int, float]] | None = None,
    vt_service_prob_floor: float = 1.0e-4,
    ev_service_prob_floor: float = 1.0e-4,
    vt_reliability_gamma: float = 0.0,
    ev_reliability_gamma: float = 0.0,
    vt_service_prob_skip_below: float = 0.0,
    ev_service_prob_skip_below: float = 0.0,
    fail_on_infeasible_demand: bool = False,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Any]]:
    """Multinomial-logit assignment.

    Demand/flows are passenger demand in pax/period. If no feasible alternatives exist,
    demand is recorded to ``unserved_demand`` unless ``fail_on_infeasible_demand`` is True.
    """
    all_groups = sorted({g for od_groups in demand.values() for g in od_groups.keys()})
    flows: Dict[str, Dict[str, Dict[int, float]]] = {
        it["id"]: {group: {t: 0.0 for t in times} for group in all_groups} for it in itineraries
    }
    generalized_costs_raw: Dict[str, Dict[str, Dict[int, float]]] = {
        it["id"]: {group: {t: 0.0 for t in times} for group in all_groups} for it in itineraries
    }
    generalized_costs_perceived: Dict[str, Dict[str, Dict[int, float]]] = {
        it["id"]: {group: {t: 0.0 for t in times} for group in all_groups} for it in itineraries
    }
    utilities: Dict[str, Dict[str, Dict[int, float]]] = {
        it["id"]: {group: {t: 0.0 for t in times} for group in all_groups} for it in itineraries
    }
    utility_breakdown: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {
        it["id"]: {group: {t: {} for t in times} for group in all_groups} for it in itineraries
    }

    unserved_demand: Dict[str, Dict[str, Dict[int, float]]] = {}
    unserved_demand_total = 0.0
    unserved_cases_count = 0

    itineraries_by_od: Dict[str, List[Dict[str, Any]]] = {}
    for it in itineraries:
        od_key = f"{it['od'][0]}-{it['od'][1]}"
        itineraries_by_od.setdefault(od_key, []).append(it)

    for od_key, groups in demand.items():
        for group, time_map in groups.items():
            for t in times:
                alts = itineraries_by_od.get(od_key, [])
                if not alts:
                    if fail_on_infeasible_demand:
                        raise ValueError(f"No itineraries for OD {od_key}")
                    unserved_demand.setdefault(od_key, {}).setdefault(group, {})[t] = float(time_map.get(t, 0.0))
                    unserved_demand_total += float(time_map.get(t, 0.0))
                    unserved_cases_count += 1
                    continue
                available_alts = []
                for it in alts:
                    if is_evtol_itinerary(it) and _value_by_time(it.get("flight_time", {}), t, 0.0) <= 0.0:
                        continue
                    available_alts.append(it)
                if not available_alts:
                    if fail_on_infeasible_demand:
                        raise ValueError(f"No available itineraries for OD {od_key} at t={t}")
                    total_demand = float(time_map.get(t, 0.0))
                    if total_demand > 0.0:
                        unserved_demand.setdefault(od_key, {}).setdefault(group, {})[t] = total_demand
                        unserved_demand_total += total_demand
                        unserved_cases_count += 1
                    continue

                feasible_alts = []
                total_demand = float(time_map.get(t, 0.0))
                for it in available_alts:
                    comp = costs[it["id"]][t]
                    raw_cost = float(vot[group][t]) * comp["TT"] + comp["Money"] + comp["ChargeCost"] + float(comp.get("ContinuityPenalty", 0.0))
                    generalized_costs_raw[it["id"]][group][t] = raw_cost
                    if math.isinf(raw_cost):
                        generalized_costs_perceived[it["id"]][group][t] = float("inf")
                        utilities[it["id"]][group][t] = -float("inf")
                        continue

                    vt_prob = 1.0
                    if is_evtol_itinerary(it):
                        dep_station = it.get("dep_station")
                        if vt_service_prob and dep_station in vt_service_prob:
                            vt_prob = float(vt_service_prob[dep_station].get(t, 1.0))
                        vt_prob = min(1.0, max(vt_service_prob_floor, vt_prob))
                        if vt_service_prob_skip_below > 0.0 and vt_prob < vt_service_prob_skip_below:
                            continue

                    ev_prob = 1.0
                    if ev_service_prob is not None:
                        ev_candidates = []
                        for stop in _ev_stops_at_time(it, t):
                            station = stop.get("station")
                            if station in ev_service_prob:
                                ev_candidates.append(float(ev_service_prob[station].get(t, 1.0)))
                        if ev_candidates:
                            ev_prob = min(ev_candidates)
                    ev_prob = min(1.0, max(ev_service_prob_floor, ev_prob))
                    if (str(it.get("mode", "")) == "EV" or is_multimodal_evtol(it)) and ev_service_prob_skip_below > 0.0 and ev_prob < ev_service_prob_skip_below:
                        continue

                    vt_term = vt_reliability_gamma * math.log(max(vt_prob, vt_service_prob_floor))
                    ev_term = ev_reliability_gamma * math.log(max(ev_prob, ev_service_prob_floor))
                    lam = max(float(lambdas[group]), 1.0e-9)
                    perceived_cost = raw_cost - (vt_term + ev_term) / lam
                    util = -lam * raw_cost + vt_term + ev_term

                    generalized_costs_perceived[it["id"]][group][t] = perceived_cost
                    utilities[it["id"]][group][t] = util
                    cb = comp.get("cost_breakdown", {}) if isinstance(comp, dict) else {}
                    utility_breakdown[it["id"]][group][t] = {
                        "raw_cost": raw_cost,
                        "vt_prob": vt_prob,
                        "ev_prob": ev_prob,
                        "vt_reliability_term": vt_term,
                        "ev_reliability_term": ev_term,
                        "perceived_cost": perceived_cost,
                        "transfer_time_applied": float(cb.get("transfer_time_applied", 0.0) or 0.0),
                        "transfer_time_source": str(cb.get("transfer_time_source", "none")),
                        "access_energy_price_source": str(cb.get("access_energy_price_source", "none")),
                    }
                    feasible_alts.append((it, util))

                if total_demand <= 0.0:
                    continue
                if not feasible_alts:
                    unserved_demand.setdefault(od_key, {}).setdefault(group, {})[t] = total_demand
                    unserved_demand_total += total_demand
                    unserved_cases_count += 1
                    continue

                log_denom = logsumexp(util for _, util in feasible_alts)
                for it, util in feasible_alts:
                    flows[it["id"]][group][t] = total_demand * math.exp(util - log_denom)

    details = {
        "generalized_costs": generalized_costs_perceived,
        "generalized_costs_raw": generalized_costs_raw,
        "utilities": utilities,
        "utility_breakdown": utility_breakdown,
        "unserved_demand": unserved_demand,
        "unserved_demand_total": unserved_demand_total,
        "unserved_cases_count": unserved_cases_count,
    }
    return flows, details


def aggregate_arc_flows(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    arc_flows: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for seg in _road_segments(it):
            arc_flows.setdefault(seg["arc"], {t: 0.0 for t in times})
    for it in itineraries:
        for _, time_map in flows[it["id"]].items():
            for seg in _road_segments(it):
                arc = seg["arc"]
                t = seg["t"]
                arc_flows[arc][t] += float(seg.get("frac", 1.0)) * float(time_map.get(t, 0.0))
    return arc_flows


def aggregate_ev_station_utilization(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
    multimodal_access_utilization_factor: float = 1.0,
) -> Dict[str, Dict[int, float]]:
    utilization: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for t in times:
            for stop in _ev_stops_at_time(it, t):
                utilization.setdefault(stop["station"], {tt: 0.0 for tt in times})
    for it in itineraries:
        for _, time_map in flows.get(it.get("id"), {}).items():
            for t in times:
                for stop in _ev_stops_at_time(it, t):
                    station = stop["station"]
                    factor = float(multimodal_access_utilization_factor) if is_multimodal_evtol(it) else 1.0
                    utilization[station][t] += factor * float(time_map.get(t, 0.0))
    return utilization


def aggregate_station_utilization(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    # DEPRECATED / legacy helper.
    # This counts total station usage including eVTOL departure usage and must NOT be used
    # for C10 / EV-related station utilization consistency.
    utilization = aggregate_ev_station_utilization(itineraries, flows, times)
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        utilization.setdefault(dep_station, {t: 0.0 for t in times})
        for _, time_map in flows.get(it.get("id"), {}).items():
            for t in times:
                if _value_by_time(it.get("flight_time", {}), t, 0.0) > 0.0:
                    utilization[dep_station][t] += float(time_map.get(t, 0.0))
    return utilization


def aggregate_evtol_dep_demand(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    d_dep: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        d_dep.setdefault(dep_station, {t: 0.0 for t in times})
        for _, time_map in flows.get(it["id"], {}).items():
            for t in times:
                d_dep[dep_station][t] += float(time_map.get(t, 0.0))
    return d_dep


def aggregate_ev_energy_demand(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    components = aggregate_ev_energy_demand_components(itineraries, flows, times)
    return components["total"]


def aggregate_ev_energy_demand_components(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    pure_ev: Dict[str, Dict[int, float]] = {}
    access_ev: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if str(it.get("mode", "")) == "EV":
            for t in times:
                for stop in _ev_stops_at_time(it, t):
                    pure_ev.setdefault(stop["station"], {tt: 0.0 for tt in times})
        elif is_multimodal_evtol(it):
            for t in times:
                for stop in _ev_stops_at_time(it, t):
                    access_ev.setdefault(stop["station"], {tt: 0.0 for tt in times})
    for it in itineraries:
        for _, time_map in flows.get(it.get("id"), {}).items():
            if str(it.get("mode", "")) == "EV":
                for t in times:
                    for stop in _ev_stops_at_time(it, t):
                        station = stop["station"]
                        pure_ev[station][t] += float(stop.get("energy", 0.0)) * float(time_map.get(t, 0.0))
            elif is_multimodal_evtol(it):
                for t in times:
                    for stop in _ev_stops_at_time(it, t):
                        station = stop["station"]
                        access_ev[station][t] += float(stop.get("energy", 0.0)) * float(time_map.get(t, 0.0))
                for t in times:
                    explicit_access_energy, scalar_access_energy = _access_energy_for_time(it, t)
                    has_implicit = any(bool(st.get("implicit_access_fallback", False)) for st in _ev_stops_at_time(it, t))
                    access_energy = scalar_access_energy if explicit_access_energy <= 1.0e-12 and not has_implicit else 0.0
                    if access_energy <= 0.0:
                        continue
                    allocated = False
                    access_station_ids = _access_station_ids(it, t)
                    if access_station_ids:
                        share = access_energy / max(1, len(access_station_ids))
                        for sid in access_station_ids:
                            access_ev.setdefault(sid, {tt: 0.0 for tt in times})
                            access_ev[sid][t] += share * float(time_map.get(t, 0.0))
                        allocated = True
                    if not allocated:
                        dep_station = it.get("dep_station")
                        if dep_station is None:
                            continue
                        access_ev.setdefault(dep_station, {tt: 0.0 for tt in times})
                        access_ev[dep_station][t] += access_energy * float(time_map.get(t, 0.0))

    total: Dict[str, Dict[int, float]] = {}
    stations = set(pure_ev.keys()) | set(access_ev.keys())
    for s in stations:
        total[s] = {t: float(pure_ev.get(s, {}).get(t, 0.0)) + float(access_ev.get(s, {}).get(t, 0.0)) for t in times}
    return {"pure_ev": pure_ev, "access_ev": access_ev, "total": total}


def aggregate_evtol_demand(
    flows: Dict[str, Dict[str, Dict[int, float]]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    d_route: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        it_id = it["id"]
        d_route[it_id] = {t: 0.0 for t in times}
        for _, time_map in flows.get(it_id, {}).items():
            for t in times:
                d_route[it_id][t] += float(time_map.get(t, 0.0))
    return d_route


def compute_evtol_energy_demand(
    d_route: Dict[str, Dict[int, float]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    e_dep: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        e_dep.setdefault(dep_station, {t: 0.0 for t in times})
        for t in times:
            e_dep[dep_station][t] += d_route.get(it["id"], {}).get(t, 0.0) * _value_by_time(it.get("e_per_pax", 0.0), t, 0.0)
    return e_dep


def aggregate_vt_departure_flow_by_class(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
    *,
    output_unit: str = "pax",
    vt_pax_per_departure_fast: float = 2.0,
    vt_pax_per_departure_slow: float = 4.0,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Aggregate eVTOL departures by station and service class.

    Args:
        flows: passenger flow map in pax/period.
        output_unit: ``'pax'`` (default) or ``'departures'``.

    Returns:
        Nested dict station -> class -> time -> value in requested unit.
    """
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        cls = get_evtol_service_class(it)
        out.setdefault(dep_station, {}).setdefault(cls, {t: 0.0 for t in times})
        pax_per_dep = vt_pax_per_departure_fast if cls == "fast" else vt_pax_per_departure_slow
        pax_per_dep = max(1.0e-6, float(pax_per_dep))
        for _, time_map in flows.get(it.get("id"), {}).items():
            for t in times:
                pax_flow = float(time_map.get(t, 0.0))
                val = pax_flow if output_unit == "pax" else pax_flow / pax_per_dep
                out[dep_station][cls][t] += val
    for dep in out:
        out[dep].setdefault("fast", {t: 0.0 for t in times})
        out[dep].setdefault("slow", {t: 0.0 for t in times})
    return out
