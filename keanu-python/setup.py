from setuptools import setup, find_packages
import shutil
import os


def readme() -> str:
    with open('README.rst', "r") as f:
        return f.read()


here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, 'keanu', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

# If you don't remove the old directories, it tends to put the excluded module "examples" into the bdist
for dir_name in ("keanu-%s.dist-info" % about['__version__'], "keanu.egg-info", "build", "dist"):
    shutil.rmtree(dir_name, ignore_errors=True)

setup(
    name='keanu',
    description='A probabilistic approach from an Improbabilistic company',
    long_description=readme(),
    version=about['__version__'],
    author='Improbable Worlds',
    author_email='keanu-engineering@improbable.io',
    url='https://improbable-research.github.io/keanu/python',
    license='MIT',
    packages=find_packages(exclude=["examples"]),
    include_package_data=True,
    install_requires=['py4j', 'numpy', 'pandas', 'nd4j==1.0.0-beta3'],
    classifiers=[
        'Development Status :: 1 - Planning', 'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    zip_safe=False)
