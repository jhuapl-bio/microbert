# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

from setuptools import find_packages, setup

# pip install requirements
with open("requirements.txt") as f:
    required = f.read().splitlines()
    print(required)

setup(
    name="analysis",
    version="0.1.0",
    description="A pip installable package supporting Microbert experiment runs",
    install_requires=required,
    packages=find_packages(),
)
