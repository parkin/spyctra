#!/usr/bin/env python

from distutils.core import setup

setup(name='spyctra',
        version='0.1.0',
        description='Analysis tools for Raman and other spectroscopy data',
        author='Will Parkin',
        author_email='wmparkin@gmail.com',
        package_dir={'spyctra': 'spyctra'},
        packages=['spyctra'],
        install_requires=[
            "numpy",
            "scipy",
            ],
        )
