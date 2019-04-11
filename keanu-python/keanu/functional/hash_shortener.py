def shorten_hash(original_hash: int) -> int:
    """
    This function removes final bytes from a hash to get a hash that can be parsed in java as an int
    :param original_hash: the original hash of an object. This should have 36 bytes.
    :return: A in int parsed from the first 32 bytes of the original hash
    """
    absolute_hash_long = abs(original_hash)
    byte_array = absolute_hash_long.to_bytes(36, byteorder='big')
    cropped_byte_array = byte_array[:32]
    return int.from_bytes(cropped_byte_array, byteorder='big')
