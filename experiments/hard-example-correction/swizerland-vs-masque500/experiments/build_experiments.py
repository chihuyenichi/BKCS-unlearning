#!/usr/bin/env python3
"""Build the self-contained Kaggle scripts of the three follow-up experiments.

Each script is pipeline.py (unchanged apart from two asserted edits) followed by
ext_common.py and one experiment extension that redefines/extends names of the pipeline
namespace.  pipeline.py / pipeline.ipynb and the earlier runs are therefore untouched.

    python build_experiments.py [--out build]
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "pipeline.py"

TAIL = 'if __name__ == "__main__":\n    main()\n'
OPTIMIZER_OLD = '''    optimizer = torch.optim.Adam(
        branch.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
'''
OPTIMIZER_NEW = "    optimizer = make_optimizer(branch, config)\n"

EXPERIMENTS = {
    "exp_sam": ("ext_sam.py", "main_sam", ()),
    "exp_unfreeze": ("ext_unfreeze.py", "main_unfreeze", ()),
    "exp_head": ("ext_head.py", "main_head", (("NUM_WORKERS = 2\n", "NUM_WORKERS = 3\n"),)),
}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of: {old[:70]!r}")
    return text.replace(old, new)


def build(name: str) -> str:
    extension_file, entry, extra_edits = EXPERIMENTS[name]
    source = PIPELINE.read_text(encoding="utf-8")
    source = replace_once(source, TAIL, "")
    source = replace_once(source, OPTIMIZER_OLD, OPTIMIZER_NEW)
    for old, new in extra_edits:
        source = replace_once(source, old, new)
    parts = [
        source.rstrip("\n"),
        (HERE / "ext_common.py").read_text(encoding="utf-8").strip("\n"),
        (HERE / extension_file).read_text(encoding="utf-8").strip("\n"),
        f'if __name__ == "__main__":\n    {entry}()\n',
    ]
    text = "\n\n\n".join(parts)
    compile(text, name + ".py", "exec")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "build")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENTS:
        text = build(name)
        (args.out / f"{name}.py").write_text(text, encoding="utf-8", newline="\n")
        print(f"{name}.py: {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
