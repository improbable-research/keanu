from setuptools import setup, find_packages

setup(
    name='nd4j',
    description='A package of the nd4j jars',
    version='1.0.0-beta3',
    author='Improbable Worlds',
    author_email='keanu-engineering@improbable.io',
    url='https://improbable-research.github.io/keanu/python',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)