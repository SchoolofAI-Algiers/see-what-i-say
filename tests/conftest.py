"""skip model/API tests automatically when the heavy optional deps are not installed"""
from importlib.util import find_spec

collect_ignore = []
if find_spec("torch") is None or find_spec("transformers") is None:
    collect_ignore += ["unit/test_grounder.py", "unit/test_contracts_compliance.py"]
if find_spec("fastapi") is None or find_spec("torch") is None:
    collect_ignore += ["unit/test_server_api.py"]