def shorten_hash(original_hash: int) -> int:
    """
    This function removes final bytes from a hash to get a hash that can be parsed in java as an int
    :param original_hash: the original hash of an object. This should have 36 bytes.
    :return: A in int parsed from the first 32 bytes of the original hash
    """
    max_int_32_bit = (2 ** 32) - 1
    return abs(original_hash) % max_int_32_bit
