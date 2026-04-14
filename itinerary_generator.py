from typing import Any, Dict, List


def _default_evtol_fare(config: Dict[str, Any], service_class: str) -> float:
    fare_base = float(config.get("vt_fare_base", 4.0))
    premium = float(config.get("vt_fare_premium_fast", 1.2))
    return fare_base + premium if service_class == "fast" else fare_base


def generate_itineraries(
    data: Dict[str, Any],
    travel_times: Dict[str, Dict[int, float]],
    ev_station_waits: Dict[str, Dict[int, float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Experimental placeholder itinerary generator.

    Current production flow uses input itineraries from case files.
    This function intentionally returns an empty list unless explicitly enabled,
    and remains a scaffold for future generator implementations.
    """
    if not config.get("use_generator", False):
        return []
    hybrid = [str(x) for x in data.get("sets", {}).get("hybrid_stations", [])]
    params = data.get("parameters", {})
    dep_allowed = {s: bool(params.get("vt_departure_allowed", {}).get(s, True)) for s in hybrid}
    arr_allowed = {s: bool(params.get("vt_arrival_allowed", {}).get(s, True)) for s in hybrid}
    feasible_dep_stations = [s for s in hybrid if dep_allowed.get(s, True)]
    feasible_arr_stations = [s for s in hybrid if arr_allowed.get(s, True)]
    _ = feasible_dep_stations
    _ = feasible_arr_stations
    # Placeholder for V1+ MILP generation (not activated in toy data).
    # Future generator should support pure EV/eVTOL and multimodal EV_to_eVTOL fast/slow.
    # Any future VT itinerary generation should use only feasible_dep_stations for
    # departures while still allowing arrival-only vertiports in feasible_arr_stations.
    # Keep service-class-aware fallback fares for compatibility.
    _ = _default_evtol_fare(config, "fast")
    _ = _default_evtol_fare(config, "slow")
    return []
