import os
from pathlib import Path
from setuptools import Distribution, setup, find_packages


class NPUDistribution(Distribution):
    def has_ext_modules(self):
        kernels = Path(__file__).parent / 'x_base/operator/npu_fusion'
        return super().has_ext_modules() or any(kernels.glob('*/build/*.so'))

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
    install_requires = []
    for line in f.read().splitlines():
        if line.strip() and not line.startswith('#'):
            install_requires.append(line)

setup(
    distclass=NPUDistribution,
    name="x_base",
    version="3.0.6",
    description="x base.",
    packages=find_packages(),
    # 包含非 Python 文件（配置文件等）
    package_data={
        'x_base': [
            'config/*.yaml',
            'operator/npu_fusion/*/CMakeLists.txt',
            'operator/npu_fusion/*/op_extension/*.asc',
            'operator/npu_fusion/*/op_kernel/*.asc',
            'operator/npu_fusion/*/op_kernel/*.h',
            'operator/npu_fusion/*/template/*',
            'operator/npu_fusion/*/build/*.so',
            'operator/npu_fusion/*.md',
            'operator/npu_fusion/cmake/*.cmake',
        ]
    },
    # 确保包数据被包含
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.11',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
