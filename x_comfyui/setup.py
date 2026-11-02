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


setup(
    name="x_comfyui",
    version="0.1.0",
    description="NPU acceleration adapter for ComfyUI — consumes x_base",
    packages=find_packages(),
    install_requires=["x_base"] + load_requirements("requirements.txt"),
    python_requires=">=3.9",
    data_files=[("share/x_comfyui", ["main_multi.py"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
