from setuptools import setup, find_packages

setup(
    name="FENICE",
    version="0.1.0",
    author="Matteo Catania",
    description="My research scripts + data (depends on patched oemof fork)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # point to your forked oemof and branch:
        "oemof @ git+https://github.com/catania-matteo/oemof-solph.git@fix-for-myproject",
        "pandas>=1.0",
        "numpy>=1.20",
    ],
    package_data={"FENICE": ["data/*.xlsx"]},
)
