from setuptools import setup, find_packages
import shutil

def readme():
    with open('README.rst', "r") as f:
        return f.read()

version_string = '0.0.15.dev1'

# If you don't remove the old directories, it tends to put the excluded module "examples" into the bdist
for dir_name in ("keanu-%s.dist-info" % version_string, "keanu.egg-info", "build", "dist"):
    shutil.rmtree(dir_name, ignore_errors=True)

setup(name='keanu',
      description='A probabilistic approach from an Improbabilistic company',
      long_description=readme(),
      version=version_string,
      author='Improbable Worlds',
      author_email='keanu-engineering@improbable.io',
      url='https://improbable-research.github.io/keanu/python',
      license='MIT',
      packages=find_packages(exclude=["examples"]),
      include_package_data=True,
      install_requires=[
          'py4j',
          'numpy',
          'pandas'
      ],
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6'
      ],
      zip_safe=False)
