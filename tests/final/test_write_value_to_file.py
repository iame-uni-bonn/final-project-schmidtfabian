import pytest
from pathlib import Path

from final_project_schmidtfabian.final.write_value_to_file import write_value_to_file
from final_project_schmidtfabian.config import BLD

wrong_dtypes_filepath = [5, 1.2, True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", wrong_dtypes_filepath)
def test_write_value_to_file_wrong_dtypes_filepath(input):
    with pytest.raises(TypeError):
        write_value_to_file(value=1, filepath=input)

wrong_dtypes_value = [True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", wrong_dtypes_value)
def test_write_value_to_file_wrong_dtypes_filepath(input):
    with pytest.raises(TypeError):
        write_value_to_file(value=input, filepath= BLD / "tests" / "mse_test.txt")

def test_write_value_to_file_correct_file_created():
    test_filepath = BLD / "tests" / "mse_test.txt"
    parent_folder_filepath = BLD / "tests"
    parent_folder_filepath.mkdir(parents=True, exist_ok=True)
    write_value_to_file(value=1, filepath= test_filepath)
    with open(test_filepath, 'r') as file:
        file_content = file.read().strip()
    
    assert Path(test_filepath).is_file() and file_content == str(1), \
    "File does not exist or contains the wrong content."