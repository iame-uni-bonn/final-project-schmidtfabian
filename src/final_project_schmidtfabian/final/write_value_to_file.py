import os


def write_value_to_file(value, filepath):
    """Takes a value and a filepath and writes a .txt file only containing this value.

    Args:
        - value(integer or float): The value that should be saved in the .txt file.
        - filepath(Path-like object): The directory in which the .txt file should be written in.

    """
    _fail_if_invalid_input(value=value, filepath=filepath)
    with open(filepath, "w") as file:
        file.write(str(value))


def _fail_if_invalid_input(value, filepath):
    """Throws an error if inputs are invalid."""
    _fail_if_wrong_datatype(value=value, filepath=filepath)
    _fail_if_filepath_is_not_txt_file(filepath=filepath)


def _fail_if_wrong_datatype(value, filepath):
    """Throws an error if inputs have wrong data types."""
    if (
        not os.path.isabs(filepath)
        or not isinstance(value, int | float | str)
        or isinstance(value, bool)
    ):
        msg = (
            "'filepath' must be a valid absolute filepath"
            " and 'value' an int, float or string."
        )
        raise TypeError(
            msg,
        )


def _fail_if_filepath_is_not_txt_file(filepath):
    """Throws an error if 'filepath' does not with '.txt'."""
    filepath_str = str(filepath)
    if not filepath_str.endswith(".txt"):
        msg = "'filepath' must end with '.txt'."
        raise ValueError(msg)
