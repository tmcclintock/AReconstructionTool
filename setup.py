import sys, os, glob
from setuptools import setup, Extension
import subprocess

dist = setup(name="resampler",
             author="Tom McClintock",
             author_email="mcclintock@bnl.gov",
             description="Tool for resampling MCMC chains.",
             license="GNU General Public License v2.0",
             url="https://github.com/tmcclintock/GP_resampler",
             packages=['resampler'],
             install_requires=['cffi','numpy','scipy','george'],
             setup_requires=['pytest_runner'],
             tests_require=['pytest'])
