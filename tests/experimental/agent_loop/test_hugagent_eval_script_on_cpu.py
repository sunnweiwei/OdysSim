# Copyright 2025 Individual Contributor: OdysSim Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "recipe/ditto/eval.sh"


@pytest.mark.parametrize("include", ["unset", "0", "1", "missing", "invalid"])
def test_hugagent_eval_is_opt_in_and_checks_data(tmp_path, include):
    """Default evaluation needs no gated data; explicit opt-in validates paths before launching."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "arguments.json"
    python = bin_dir / "python3"
    python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['CAPTURE_ARGUMENTS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE_ARGUMENTS": str(capture),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "DATA_DIR": str(tmp_path / "data"),
    }
    env.pop("INCLUDE_HUGAGENT", None)
    env.pop("HUGAGENT_DATA_DIR", None)
    if include != "unset":
        env["INCLUDE_HUGAGENT"] = "1" if include == "missing" else include
    if include == "1":
        data_dir = tmp_path / "custom-hugagent"
        data_dir.mkdir()
        for task in ("bsi", "bdu"):
            (data_dir / f"hugagent_{task}_val.parquet").touch()
        env["HUGAGENT_DATA_DIR"] = str(data_dir)
    result = subprocess.run(["bash", str(SCRIPT), "api"], env=env, capture_output=True, text=True)
    if include in ("missing", "invalid"):
        assert result.returncode == 1
        assert not capture.exists()
        assert ("Missing HugAgent data" if include == "missing" else "must be 0 or 1") in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        args = json.loads(capture.read_text())
        val_files = next(arg.removeprefix("data.val_files=") for arg in args if arg.startswith("data.val_files="))
        files = val_files.strip("[]").split(",")
        assert len(files) == (29 if include == "1" else 27)
        assert sum(Path(path).name.startswith("hugagent_") for path in files) == (2 if include == "1" else 0)
        if include == "1":
            assert files[-2:] == [str(data_dir / f"hugagent_{task}_val.parquet") for task in ("bsi", "bdu")]
