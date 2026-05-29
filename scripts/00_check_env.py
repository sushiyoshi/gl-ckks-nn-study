from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, env_info, write_json


def main() -> None:
    payload = env_info()
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "desilofhe"], text=True, capture_output=True)
    payload["pip_install_desilofhe"] = {
        "returncode": pip.returncode,
        "stdout_tail": pip.stdout[-2000:],
        "stderr_tail": pip.stderr[-2000:],
    }
    try:
        import desilofhe

        payload["import_desilofhe"] = True
        payload["desilofhe_version_attr"] = getattr(desilofhe, "__version__", None)
        payload["desilofhe_version_metadata"] = importlib.metadata.version("desilofhe")
    except Exception as exc:
        payload["import_desilofhe"] = False
        payload["desilofhe_exception"] = f"{type(exc).__name__}: {exc}"
    write_json(RESULTS / "env.json", payload)
    print(RESULTS / "env.json")


if __name__ == "__main__":
    main()
