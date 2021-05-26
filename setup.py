import importlib
from pathlib import Path
from setuptools import setup
from setuptools import find_packages

def get_ivis_version():
    spec = importlib.util.spec_from_file_location(
        'ivis.version', str(Path('ivis/version.py')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VERSION

VERSION = get_ivis_version()

with open('README.md') as f:
    long_description = f.read()

setup(name='ivis',
      version=VERSION,
      description='Artificial neural network-driven visualization of high-dimensional data using triplets.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/beringresearch/ivis',
      author='Benjamin Szubert, Ignat Drozdov',
      author_email='bszubert@beringresearch.com, idrozdov@beringresearch.com',
      license='Apache License, Version 2.0',
      packages=find_packages(),
      python_requires='>=3.5',
      install_requires=[
          'numpy',
          'scikit-learn>0.20.0',
          'annoy>=1.15.2',
          'tqdm',
          'dill'
      ],
      extras_require={
          'tests': ['pytest'],
          'visualization': ['matplotlib', 'seaborn'],
          'cpu': ['tensorflow-cpu'],
          'gpu': ['tensorflow']
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ],
      zip_safe=False)
