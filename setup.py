from setuptools import setup

setup(name='ivis',
      version='1.0',
      description='Artificial neural network-driven visualization of high-dimensional data using triplets.',
      url='http://github.com/beringresearch/dimensionality_reduction',
      author='Benjamin Szubert, Ignat Drozdov',
      author_email='bszubert@beringresearch.com, idrozdov@beringresearch.com',
      license='MIT',
      packages=['ivis'],
      install_requires=[
          'tensorflow',
          'keras',
          'numpy',
          'scikit-learn',
          'annoy',
      ],
      zip_safe=False)