from setuptools import setup, find_packages

def readme():
    with open('README.rst', "r") as f:
        return f.read()

setup(name='keanu',
      description='A probabilistic approach from an Improbabilistic company',
      long_description=readme(),
      version='0.0.14.dev4',
      author='Improbable Worlds',
      author_email='keanu-engineering@improbable.io',
      url='https://github.com/improbable-research/keanu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'py4j',
          'numpy'
      ],
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6'
      ],
      zip_safe=False)
