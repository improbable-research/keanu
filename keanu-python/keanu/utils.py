import re

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')

def get_java_name(name):
    """
    >>> get_java_name("snake_case_name")
    'snakeCaseName'
    >>> get_java_name("camelCaseName")
    'camelCaseName'
    """
    first, *rest = name.split('_')
    return first + ''.join(word.capitalize() for word in rest)

def is_python_name(name):
    """
    >>> is_python_name("snake_case_name")
    True
    >>> is_python_name("snakecasename")
    True
    >>> is_python_name("camelCaseName")
    False
    >>> is_python_name("camelCaseNAME")
    False
    """
    s1 = first_cap_re.sub(r'\1_\2', name)
    python_name = all_cap_re.sub(r'\1_\2', s1).lower()

    return python_name == name
