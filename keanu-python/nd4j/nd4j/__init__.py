from pkg_resources import resource_filename

from .__version__ import __version__


def get_classpath():
    classpath = resource_filename(__name__, 'classpath')
    return classpath
