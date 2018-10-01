from setuptools import setup

setup(name='keanu',
      version='0.1',
      author='Improbable Worlds',
      author_email='UNKNOWN',
      packages=['keanu', 'keanu.generated'],
      install_requires=[
          'py4j',
          'pandas',
          'numpy'
      ],
      zip_safe=False)
