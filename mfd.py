from typing import Dict, List, Tuple


def boundary_flows(
    arc_flows: Dict[str, Dict[int, float]],
    boundary_in: List[str],
    boundary_out: List[str],
    times: List[int],
) -> Tuple[List[float], List[float]]:
    inflow = []
    outflow = []
    for t in times:
        inflow.append(sum(float(arc_flows.get(a, {}).get(t, 0.0)) for a in boundary_in))
        outflow.append(sum(float(arc_flows.get(a, {}).get(t, 0.0)) for a in boundary_out))
    return inflow, outflow


def update_accumulation(n0: float, inflow: List[float], outflow: List[float], delta_t: float) -> List[float]:
    n = [float(n0)]
    for i in range(len(inflow)):
        n_next = max(0.0, float(n[-1]) + float(delta_t) * (float(inflow[i]) - float(outflow[i])))
        n.append(n_next)
    return n


def compute_g(n_series: List[float], mfd_params: Dict[str, float]) -> List[float]:
    n_crit = max(1.0e-6, float(mfd_params.get("n_crit", 1.0)))
    g_max = max(1.0e-6, float(mfd_params.get("g_max", 1.0)))
    beta = max(1.0, float(mfd_params.get("beta", 2.0)))
    return [max(0.0, 1.0 - min(1.0, (max(0.0, float(n)) / n_crit) ** beta) * (1.0 - g_max)) for n in n_series]
