import re

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')

def _get_java_name(name):
    """
    >>> _get_java_name("snake_case_name")
    'snakeCaseName'
    >>> _get_java_name("camelCaseName")
    'camelCaseName'
    """
    first, *rest = name.split('_')
    return first + ''.join(word.capitalize() for word in rest)

def _get_python_name(name):
    """
    >>> _get_python_name("camelCaseName")
    'camel_case_name'
    >>> _get_python_name("snake_case_name")
    'snake_case_name'
    """
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()
