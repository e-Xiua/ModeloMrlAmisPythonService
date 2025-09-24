from setuptools import setup, find_packages

setup(
    name="runmodel",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
        "numpy>=1.26,<2",
        "pandas>=2.0,<3",
        "openpyxl>=3.1.2",
        "scikit-learn",
        "matplotlib"
    ],
)