#!/usr/bin/env python
import os
from setuptools import setup, find_packages

with open('./requirements.txt', 'r') as f:
    requirements = f.read().strip().split('\n')

setup(
    name    ='topos',
    version ='0.2',
    description ="Statistics and Topology",
    author      ="Olivier Peltre",
    author_email='opeltre@gmail.com',
    url     ='https://github.com/opeltre/topos',
    license ='MIT',
    install_requires=requirements,
    packages = ['topos', 
                'topos.base', 
                'topos.core',
                'topos.domain',
                'topos.exceptions']
)
