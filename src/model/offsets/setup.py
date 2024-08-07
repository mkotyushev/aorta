from setuptools import setup, Extension

module = Extension('offsets', sources=['offsets.c'])

setup(name='offsets',
      version='1.0',
      description='Python Package with C extension for calculating 3D offsets',
      ext_modules=[module])