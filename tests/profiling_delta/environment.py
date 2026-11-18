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
"""The environment a frozen delta bound is valid in.

A delta is interpretune's arm minus the native arm, so it moves with every library either arm allocates through,
not only with interpretune. The bounds in `delta_bounds.yaml` were once frozen under torch 2.15 and checked by a gate
whose image carries torch 2.13: the gate reported a 24 MiB regression that interpretune did not cause. Each vendor's
bound is therefore valid only in the environment it was frozen in, recorded in `frozen_environments.yaml`, and the
test refuses to compare against a bound frozen elsewhere.

interpretune is deliberately not part of the environment: its changes are what the bound exists to catch.

Record the environment of a set of pair artifacts (after freezing the bounds from them)::

    python -m tests.profiling_delta.environment --artifacts <nvidia dir> <amd dir>
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import yaml

ENVIRONMENTS_FILE = Path(__file__).with_name("frozen_environments.yaml")

#: The distributions either arm allocates accelerator memory through. A version string carries its local build tag
#: (``+cu130``, a VCS hash), so a rebuilt wheel or a moved git pin counts as a different environment.
SALIENT_DISTRIBUTIONS: tuple[str, ...] = (
    "torch",
    "transformers",
    "accelerate",
    "transformer-lens",
    "nnsight",
    "circuit-tracer",
    "interp-engine",
)


def salient_versions() -> dict[str, str | None]:
    """The installed version of each salient distribution, ``None`` where it is not installed."""
    versions: dict[str, str | None] = {}
    for name in SALIENT_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def frozen_environment(vendor: str) -> dict[str, str | None] | None:
    """The environment ``vendor``'s bounds were frozen in, or ``None`` when none is recorded."""
    environments = yaml.safe_load(ENVIRONMENTS_FILE.read_text()) if ENVIRONMENTS_FILE.exists() else None
    return (environments or {}).get(vendor)


def environment_mismatch(frozen: dict[str, str | None], running: dict[str, str | None]) -> list[str]:
    """Each salient distribution whose version moved, as ``name: frozen -> running``."""
    return [
        f"{name}: {frozen.get(name)} -> {running.get(name)}"
        for name in SALIENT_DISTRIBUTIONS
        if frozen.get(name) != running.get(name)
    ]


def record(paths: list[Path]) -> dict[str, dict[str, str | None]]:
    """Each vendor's environment from its pair artifacts, refusing a vendor whose artifacts disagree."""
    files = [f for p in paths for f in (sorted(p.rglob("*.pair.json")) if p.is_dir() else [p])]
    if not files:
        raise SystemExit(f"no *.pair.json artifacts under {[str(p) for p in paths]}")
    by_vendor: dict[str, dict[str, str | None]] = {}
    for f in files:
        pair = json.loads(f.read_text())
        if "environment" not in pair:
            raise SystemExit(f"{f} records no environment (measured before environments were recorded): re-measure")
        seen = by_vendor.setdefault(pair["vendor"], pair["environment"])
        if seen != pair["environment"]:
            raise SystemExit(
                f"{pair['vendor']} artifacts were measured in different environments: "
                + "; ".join(environment_mismatch(seen, pair["environment"]))
            )
    return by_vendor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ENVIRONMENTS_FILE)
    args = parser.parse_args(argv)
    existing = yaml.safe_load(args.out.read_text()) if args.out.exists() else None
    environments = {**(existing or {}), **record(args.artifacts)}
    header = "# Each vendor's frozen delta bounds are valid only in this environment (see environment.py).\n"
    args.out.write_text(header + yaml.safe_dump(environments, sort_keys=True))
    print(f"wrote {args.out}: {sorted(environments)}")


if __name__ == "__main__":
    main()
