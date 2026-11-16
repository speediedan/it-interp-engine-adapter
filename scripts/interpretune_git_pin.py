#!/usr/bin/env python3
"""Print the requirement interpretune pins for a package in its ``git-deps`` dependency group.

usage: python scripts/interpretune_git_pin.py <package> [<interpretune ref>]

CI installs ``interpretune @ main``, and a PEP 735 dependency group does not transit a git install, so a package
interpretune carries only there (a git URL, which PyPI metadata cannot hold) has to be installed beside it. Reading the
pin from interpretune's ``pyproject.toml`` at the ref CI installs keeps one pin rather than two: a copied SHA looks
authoritative and drifts silently. ``.azure-pipelines/gpu-tests.yml`` derives circuit-tracer's the same way, inline.

finetuning-scheduler is read here until 2.15.0, the first release with ``TunableView`` and its conformance cases, is
on PyPI and interpretune's ``conformance`` extra carries it (speediedan/interpretune#760). Then this install, and this
script if nothing else uses it, should be deleted in favour of the extra.

Exits non-zero unless exactly one entry names the package, so a renamed or duplicated pin fails the install loudly.
"""

import sys
import tomllib
import urllib.request


def main() -> None:
    package = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else "main"
    url = f"https://raw.githubusercontent.com/speediedan/interpretune/{ref}/pyproject.toml"
    data = tomllib.loads(urllib.request.urlopen(url, timeout=30).read().decode())
    entries = [
        e
        for group in data.get("dependency-groups", {}).values()
        for e in group
        if isinstance(e, str) and e.split("@")[0].split(">")[0].split("=")[0].strip() == package
    ]
    if len(entries) != 1:
        sys.exit(
            f"expected exactly one {package} entry in interpretune's dependency-groups, found {len(entries)}: {entries}"
        )
    print(entries[0])


if __name__ == "__main__":
    main()
