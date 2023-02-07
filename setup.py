#!/usr/bin/env python

from distutils.core import setup

setup(name='qens',
    version='1.0.0',
    description='Quantile ensemble models',
    author='Serena Wang, Evan L. Ray',
    author_email='elray@umass.edu',
    url='https://github.com/reichlab/qenspy',
    py_modules=['qens'],
    install_requires=[
        'numpy',
        'tensorflow>=2',
        'tensorflow_probability>=0.16.0'
    ]
)
