from __future__ import annotations

"""清理迁移进 scores 的旧仓库结果（log_path 包含 results_old）。"""

import shutil
from pathlib import Path


def main() -> int:
    root = Path("results/scores")
    backup = root / "_backup_old_imports"
    backup.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in sorted(root.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "results_old" not in text:
            continue
        try:
            shutil.copy2(path, backup / path.name)
        except Exception:
            pass
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    print(f"removed {removed} files; backup -> {backup}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
