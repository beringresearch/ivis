import os
import imp
from setuptools import setup
from setuptools import find_packages

VERSION = imp.load_source(
        'ivis.version', os.path.join('ivis', 'version.py')).VERSION

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
      license='GNU General Public License v2.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scikit-learn>0.20.0',
          'annoy>=1.15.2',
          'tqdm'
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
