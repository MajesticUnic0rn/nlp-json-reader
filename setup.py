from setuptools import find_packages, setup
from similarity_matching import __name__, __version__


try:
    requirements = open("requirements.txt").readlines()
except FileNotFoundError:
    requirements = []

try:
    dev_requirements = open("dev_requirements.txt").readlines()
except FileNotFoundError:
    dev_requirements = []

LIBRARIES = [*requirements, *dev_requirements]

setup(
    name=__name__,
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requirements + requirements, "all": LIBRARIES},
)
