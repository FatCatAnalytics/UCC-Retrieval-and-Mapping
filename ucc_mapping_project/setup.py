from setuptools import find_packages, setup

setup(
    name="ucc_mapping_project",
    packages=find_packages(exclude=["ucc_mapping_project_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "pandas",
        "numpy",
        "rapidfuzz",
        "sentence-transformers",
        "faiss-cpu",
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
