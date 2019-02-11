from pkg_resources import resource_filename


def get_classpath():
    classpath = resource_filename(__name__, 'classpath')
    return classpath
