from setuptools import setup
from setuptools import find_packages

setup(name='ivis',
      version='1.1.4',
      description='Artificial neural network-driven visualization of high-dimensional data using triplets.',
      url='http://github.com/beringresearch/ivis',
      author='Benjamin Szubert, Ignat Drozdov',
      author_email='bszubert@beringresearch.com, idrozdov@beringresearch.com',
      license='GNU General Public License v2.0',
      packages=find_packages(),
      install_requires=[
          'tensorflow',
          'keras',
          'numpy',
          'scikit-learn',
          'annoy>=1.15.2',
          'tqdm'
      ],
      extras_require={
          'tests': ['pytest']
      },
      zip_safe=False)
