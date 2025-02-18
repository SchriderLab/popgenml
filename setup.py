# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='popgenml',
    version='0.0.1',
    packages=find_packages(include = ['popgenml', 'popgenml/*']),
    package_data={
        'my_package': ['data/*.csv', 'config.ini']
    },
)
