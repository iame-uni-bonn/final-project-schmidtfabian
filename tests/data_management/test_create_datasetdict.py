import pandas as pd
import pytest

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.data_management.create_datasetdict import (
    create_datasetdict,
)

WRONG_DTYPES = [5, 1.2, "hello", True, None, [1, 2, 3], {"a": 1, "b": 1}]


@pytest.mark.parametrize("input", WRONG_DTYPES)
def test_create_datasetdict_wrong_dtypes(input):
    """Tests if function raises a TypeError for inputs with the wrong datatype."""
    with pytest.raises(TypeError):
        create_datasetdict(input)


test_dictionary_not_enough_elements = {
    "date": ["25.02.2024", "26.02.2024", "27.02.2024"],
    "values": ["Hello", "World", "!"],
}
test_dataframe_not_enough_elements = pd.DataFrame(test_dictionary_not_enough_elements)


def test_create_datasetdict_not_enough_elements():
    """Tests if function raises an error if the number of elements in the column
    'values' is too small."""
    with pytest.raises(ValueError):
        create_datasetdict(test_dataframe_not_enough_elements)


test_dictionary_correct_dataframe = {
    "text": ["Hello", "World", "!", "EPPE", "is", "fun"],
}
test_dataframe_correct_dataframe = pd.DataFrame(test_dictionary_correct_dataframe)


def test_create_datasetdict_correct_length_datasets():
    """Tests if the datasets in the DatasetDict have the correct length."""
    test_datasetdict = create_datasetdict(test_dataframe_correct_dataframe)
    assert (
        len(test_datasetdict["train"]["text"]) == 4
        and len(test_datasetdict["test"]["text"]) == 1
        and len(test_datasetdict["validation"]["text"]) == 1
    ), "The train, test and validation dataset should have a length of 4,1 and 1 respectively."


def test_create_datasetdict_correct_columns():
    """Tests if the datasets in the DatasetDict only contain the right columns."""
    test_datasetdict = create_datasetdict(test_dataframe_correct_dataframe)
    feature_names_train = test_datasetdict["train"].features.keys()
    feature_names_test = test_datasetdict["test"].features.keys()
    feature_names_validation = test_datasetdict["validation"].features.keys()
    assert (
        len(feature_names_train) == 1
        and "text" in feature_names_train
        and feature_names_train == feature_names_test
        and feature_names_train == feature_names_validation
    ), "Error: 'text' is not the only feature or not found in the dataset."


def test_create_datasetdict_no_same_value():
    """Tests if there are no intersecting values in all datasets in the DatasetDict."""
    test_datasetdict = create_datasetdict(test_dataframe_correct_dataframe)
    train_set = set(test_datasetdict["train"]["text"])
    test_set = set(test_datasetdict["test"]["text"])
    validation_set = set(test_datasetdict["validation"]["text"])
    assert (
        len(train_set.intersection(test_set)) == 0
        and len(train_set.intersection(validation_set)) == 0
        and len(test_set.intersection(validation_set)) == 0
    ), "Train and test datasets have overlapping values"
