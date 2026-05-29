from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_json


def pip_version() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return {"ok": True, "value": proc.stdout.strip()}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    payload: dict[str, Any] = {
        "name": name,
        "found": spec is not None,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
    }
    if spec is None:
        payload.update({"import_ok": False, "version": None, "module_file": None, "error": "module_not_found"})
        return payload
    try:
        module = importlib.import_module(name)
        payload["import_ok"] = True
        payload["module_file"] = getattr(module, "__file__", None)
        payload["version"] = getattr(module, "__version__", None)
        if payload["version"] is None:
            try:
                payload["version"] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                payload["version"] = None
        payload["error"] = None
    except Exception as exc:
        payload.update(
            {
                "import_ok": False,
                "version": None,
                "module_file": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return payload


def main() -> None:
    payload = {
        "platform": platform.platform(),
        "sys_version": sys.version,
        "sys_executable": sys.executable,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "env": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH"),
        },
        "pip": pip_version(),
        "modules": {
            "openfhe": module_status("openfhe"),
            "openfhe_numpy": module_status("openfhe_numpy"),
        },
    }
    write_json(RESULTS / "openfhe_numpy_env_probe.json", payload)
    print(RESULTS / "openfhe_numpy_env_probe.json")


if __name__ == "__main__":
    main()
