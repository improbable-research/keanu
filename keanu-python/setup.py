from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='keanu',
      description='A probabilistic approach from an Improbabilistic company',
      long_description=readme(),
      version='0.0.1',
      author='Improbable Worlds',
      author_email='keanu-engineering@improbable.io',
      url='https://github.com/improbable-research/keanu',
      license='MIT',
      packages=['keanu'],
      install_requires=[
          'py4j',
          'numpy'
      ],
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6'
      ],
      zip_safe=False)
