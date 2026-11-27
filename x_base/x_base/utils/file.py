import os
import json
from pathlib import Path
import random


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def list_cases(input_dir: str, *, sample_size: int = 0, seed: int = 42):
    """
    返回 case_dir 列表：只取包含 image.jpg 与 prompt.txt 的子目录
    先过滤，再排序，避免 .DS_Store 等混入导致排序报错
    """
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    case_names = []

    for name in os.listdir(input_dir):
        case_dir = os.path.join(input_dir, name)
        if not os.path.isdir(case_dir):
            continue

        img = os.path.join(case_dir, "image.jpg")
        txt = os.path.join(case_dir, "prompt.txt")
        if os.path.isfile(img) and os.path.isfile(txt):
            case_names.append(name)

    # 再排序：数字按数值排，其它按字符串排（稳）
    case_names = sorted(case_names, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    cases = [os.path.join(input_dir, name) for name in case_names]
    if 0 < sample_size < len(cases):
        return random.Random(seed).sample(cases, sample_size)
    return cases

def atomic_json(path, value):
    """Replace a JSON output only after the complete temporary file is written."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
