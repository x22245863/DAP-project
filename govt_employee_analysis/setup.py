from setuptools import find_packages, setup

setup(
    name="govt_employee_analysis",
    packages=find_packages(exclude=["govt_employee_analysis_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
