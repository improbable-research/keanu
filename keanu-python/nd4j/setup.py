from setuptools import setup, find_packages

setup(
    name='nd4j',
    description='A package of the Nd4j jars for use by packages that use Py4j.',
    version='1.0.0-beta3',
    author='Improbable Worlds',
    author_email='keanu-engineering@improbable.io',
    url='https://improbable-research.github.io/keanu/python',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)