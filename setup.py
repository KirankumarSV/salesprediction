from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements() -> List[str]:
    with open('requirements.txt') as f:
        requirements_list = f.read().splitlines()

        if HYPHEN_E_DOT in requirements_list:
            requirements_list.remove(HYPHEN_E_DOT)

    return requirements_list

setup(
    name='salesprediction',
    version='0.0.1',
    author='Kiran',
    author_email='kumarsv.kiran@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
