import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    project_root = Path(__file__).resolve().parents[1]
    if candidate.parts and candidate.parts[0] == project_root.name:
        candidate = Path(*candidate.parts[1:])

    rooted = project_root / candidate
    if rooted.exists():
        return rooted

    return Path(path)


def load_yaml(path: str) -> Dict[str, Any]:
    resolved_path = _resolve_path(path)
    try:
        import yaml  # type: ignore

        with open(resolved_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ModuleNotFoundError:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"PyYAML is not installed and file is not valid JSON: {resolved_path}. "
                "Install with: pip install pyyaml"
            ) from exc


def require_paths(data: Dict[str, Any], paths: Iterable[str]) -> None:
    for path in paths:
        cursor: Any = data
        for key in path.split("."):
            if isinstance(cursor, dict) and key in cursor:
                cursor = cursor[key]
            else:
                raise ValueError(f"Missing required key path: {path}")


def logsumexp(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("logsumexp requires at least one value")
    max_val = max(values_list)
    if math.isinf(max_val):
        return max_val
    total = sum(math.exp(v - max_val) for v in values_list)
    return max_val + math.log(total)


def ensure_time_indexed(mapping: Dict[Any, Dict[int, float]], times: List[int]) -> None:
    for key, time_map in mapping.items():
        for t in times:
            if t not in time_map:
                raise ValueError(f"Missing time {t} for key {key}")
