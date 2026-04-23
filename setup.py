import runpy
from setuptools import setup, find_packages

version = runpy.run_path("oct_tools/version.py")["__version__"]
setup(
    name="oct-sam",
    packages=find_packages(exclude=["test"]),
    version=version,
    author="Constantin Pape; Martin Schilling",
    license="MIT",
    entry_points={
        "console_scripts": [
            "oct_tools.interactive = oct_tools.cli:interactive",
        ]
    }
)
