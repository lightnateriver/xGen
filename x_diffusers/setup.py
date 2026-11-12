import os
import sys

from setuptools import setup, find_packages


def load_requirements(req_file: str) -> list:
    """Load install_requires from a requirements file."""
    req_path = os.path.join(os.path.dirname(__file__), req_file)
    if not os.path.exists(req_path):
        print(f"[ERROR] {req_file} not found at {req_path}")
        sys.exit(1)
    with open(req_path, "r") as f:
        deps = []
        for line in f.read().splitlines():
            if line.strip() and not line.startswith("#"):
                deps.append(line)
    return deps


# 通过环境变量选择 torch 版本
#   X_DIFFUSERS_TORCH_VERSION=210 → requirements.txt,      包名 x_diffusers
#   X_DIFFUSERS_TORCH_VERSION=29  → requirements_29.txt,    包名 x_diffusers
#   未设置 → 默认 torch 2.10.0
_torch_version = os.environ.get("X_DIFFUSERS_TORCH_VERSION", "210")

if _torch_version == "210":
    _req_file = "requirements.txt"
    _pkg_display = "2.10.0"
elif _torch_version == "29":
    _req_file = "requirements_29.txt"
    _pkg_display = "2.9.0"
else:
    print(f"[ERROR] Unknown X_DIFFUSERS_TORCH_VERSION: {_torch_version}")
    print(f"  Supported values: 29, 210")
    sys.exit(1)

install_requires = load_requirements(_req_file)
_profile = os.environ.get("X_DIFFUSERS_PROFILE", "video")
if _profile == "h3-image":
    install_requires = [dep for dep in install_requires if not dep.startswith("diffusers")]
    install_requires += load_requirements("requirements-h3-image.txt")
elif _profile != "video":
    sys.exit(f"Unknown X_DIFFUSERS_PROFILE: {_profile}; expected video or h3-image")

setup(
    name="x_diffusers",
    version="3.0.7",
    description=f"x diffusers (torch {_pkg_display})",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
