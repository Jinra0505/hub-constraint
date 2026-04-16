Final runnable package manifest (paper branch)

Required runner:
- runner_hub_charging_multivot.py

Included files:
- assignment.py
- charging_hiGHS_and_gurobi_bound_fix.py
- data_loader.py
- runner_hub_charging_multivot.py
- cases/case_simple_validation.json
- cases/out_simple_validation.json
- cases/schema_min.json

Intentionally excluded legacy/unused files:
- mfd.py
- runner_new_background_boundary_flow.py
- any boundary-flow / MFD-only outputs

Exact rerun command (from repository root):
python3 - <<'PY'
import importlib.util, pathlib, json, types, sys
root=pathlib.Path('/workspace/hub-constraint')
runner_path=root/'runner_hub_charging_multivot.py'
case_path=root/'cases/case_simple_validation.json'
schema_path=root/'cases/schema_min.json'
out_path=root/'cases/out_simple_validation.json'
pkg=types.ModuleType('hub_constraint'); pkg.__path__=[str(root)]; sys.modules['hub_constraint']=pkg
spec=importlib.util.spec_from_file_location('hub_constraint.runner_hub_charging_multivot', runner_path)
mod=importlib.util.module_from_spec(spec); sys.modules['hub_constraint.runner_hub_charging_multivot']=mod; spec.loader.exec_module(mod)
data=mod.load_data(str(case_path), str(schema_path))
res=mod.run_hub_charging_multivot(data)
out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {out_path}')
PY

This folder is a full runnable package for the current correct hub-charging pipeline,
not just the changed-file list.
