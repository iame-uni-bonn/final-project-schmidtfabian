def write_mse_to_file(value, filepath):
    """
    This function takes a mean squared value and a filepath and writes a .txt file only containing this value.
    
    Args:
    value(integer or float): The value that should be saved in the .txt file.
    filepath(Path-like object): The directory in which the .txt file should be written in.

    """
    _fail_if_invalid_input(value=value, filepath=filepath)
    with open(filepath, 'w') as file:
        file.write(str(value))

def _fail_if_invalid_input(value, filepath):
    _fail_if_wrong_datatype(value=value, filepath=filepath)

def _fail_if_wrong_datatype(value, filepath):
    if not isinstance(filepath, str) and not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise TypeError("'filepath' must be a valid filepath and 'value' an int, float or string.")