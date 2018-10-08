from setuptools import setup

setup(name='keanu',
      version='0.1',
      author='Improbable Worlds',
      author_email='keanu-engineering@improbable.io',
      packages=['keanu', 'keanu.generated'],
      install_requires=[
          'py4j',
          'numpy'
      ],
      zip_safe=False)
