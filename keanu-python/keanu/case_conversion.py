def _to_camel_case_name(name: str) -> str:
    """
    >>> _to_camel_case_name("snake_case_name")
    'snakeCaseName'
    >>> _to_camel_case_name("camelCaseName")
    'camelCaseName'
    """
    first, *rest = name.split('_')
    return first + ''.join(word.capitalize() for word in rest)
