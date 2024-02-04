""" setup file to install package """

import os

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    _readme = f.read()


def read(version_filepath: str) -> str:
    """
    Get the content of the version file.

    Args:
        version_filepath: relative path to the current file setup.py

    Returns:
        The content of the version file.

    """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, version_filepath), "r") as fp:
        return fp.read()


def get_version(version_filepath: str) -> str:
    """
    Get the version in version file

    Args:
        version_filepath: path of the version file.

    Returns:
        The version of the application

    """
    for line in read(version_filepath).splitlines():
        if line.startswith("VERSION"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("./requirements.txt", encoding="utf8") as file:
    dependencies = file.read().splitlines()


setup(
    name="machinery-diag",
    version=get_version("machinery/version.py"),
    author="Khaled Benaggoune",
    author_email="khaled.mommi@gmail.com",
    description=("Package for diagnostic open data set"),
    long_description=_readme,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="phm, diagnostic, machinery",
    packages=find_packages("."),
    install_requires=dependencies,
    package_dir={"": "."},
)
