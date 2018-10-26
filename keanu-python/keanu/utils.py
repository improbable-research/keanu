def get_java_name(name):
    """
    >>> get_java_name("snake_case_name")
    'snakeCaseName'
    >>> get_java_name("camelCaseName")
    'camelCaseName'
    """
    first, *rest = name.split('_')
    return first + ''.join(word.capitalize() for word in rest)
