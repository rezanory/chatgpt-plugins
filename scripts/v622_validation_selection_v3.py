from __future__ import annotations

import importlib.util
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
V2_PATH = HERE / "v622_validation_selection_v2.py"

spec = importlib.util.spec_from_file_location("v622_validation_selection_v2", V2_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load V6.2.2 selection v2 module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

_original_request = module._request
_governance_names = set(module.SOURCE_NAMES)
_expected_hashes = dict(module.EXPECTED_SOURCE_HASHES)


def resilient_request(worker_id: str, artifact_name: str, timeout: int = 120):
    if artifact_name not in _governance_names:
        return _original_request(worker_id, artifact_name, timeout=timeout)

    last_error: Exception | None = None
    for attempt in range(1, 31):
        try:
            value, content = _original_request(worker_id, artifact_name, timeout=timeout)
            if value.get("governance_source_package_pinned") is not True:
                raise RuntimeError(
                    f"stale bridge edge for {artifact_name}: governance_source_package_pinned != true; "
                    f"file={value.get('file_name')} sha={value.get('sha256')}"
                )
            expected = _expected_hashes.get(artifact_name)
            if expected and value.get("sha256") != expected:
                raise RuntimeError(
                    f"noncanonical governance artifact for {artifact_name}: "
                    f"{value.get('sha256')} != {expected}; file={value.get('file_name')}"
                )
            if attempt > 1:
                print(f"Governance edge converged for {artifact_name} on attempt {attempt}/30", flush=True)
            return value, content
        except Exception as exc:
            last_error = exc
            print(f"Governance request {artifact_name} attempt {attempt}/30 not ready: {exc}", flush=True)
            time.sleep(3)

    raise RuntimeError(f"Canonical governance artifact did not converge for {artifact_name}: {last_error}")


module._request = resilient_request
module.main()
