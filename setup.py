from setuptools import setup
from setuptools import find_packages

setup(name='ivis',
      version='1.1.2',
      description='Artificial neural network-driven visualization of high-dimensional data using triplets.',
      url='http://github.com/beringresearch/ivis',
      author='Benjamin Szubert, Ignat Drozdov',
      author_email='bszubert@beringresearch.com, idrozdov@beringresearch.com',
      license='Creative Commons Attribution-NonCommercial-NoDerivs 3.0',
      packages=find_packages(),
      install_requires=[
          'tensorflow',
          'keras',
          'numpy',
          'scikit-learn',
          'annoy',
          'tqdm'
      ],
      zip_safe=False)
