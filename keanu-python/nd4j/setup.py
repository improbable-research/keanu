import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, 'nd4j', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

setup(
    name='nd4j',
    description='A package of the Nd4j jars for use by packages that use Py4j.',
    version=about['__version__'],
    author='Improbable Worlds',
    author_email='keanu-engineering@improbable.io',
    url='https://improbable-research.github.io/keanu/python',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
