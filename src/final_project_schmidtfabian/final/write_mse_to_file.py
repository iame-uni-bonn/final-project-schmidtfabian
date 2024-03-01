def write_mse_to_file(value, filepath):
    """
    This function takes a mean squared value and a filepath and writes a .txt file only containing this value.
    
    Args:
    value(integer or float): The value that should be saved in the .txt file.
    filepath(Path-like object): The directory in which the .txt file should be written in.

    """
    with open(filepath, 'w') as file:
        file.write(str(value))