import re

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def _to_camel_case_name(name: str) -> str:
    """
    >>> _to_camel_case_name("snake_case_name")
    'snakeCaseName'
    >>> _to_camel_case_name("camelCaseName")
    'camelCaseName'
    """
    first, *rest = name.split('_')
    return first + ''.join(word.capitalize() for word in rest)


def _to_snake_case_name(name: str) -> str:
    """
    >>> _to_snake_case_name("camelCaseName")
    'camel_case_name'
    >>> _to_snake_case_name("snake_case_name")
    'snake_case_name'
    """
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()
