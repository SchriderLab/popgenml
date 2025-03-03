# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='popgenml',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'popgenml': ['slim/*']
    },
    include_package_data=True,
)
