import pandas as pd
import pytest
from datasets import Dataset
from final_project_schmidtfabian.analysis.create_labels_for_summary_statistics import (
    create_labels_for_summary_statistics,
)

wrong_dtypes = [5, 1.2, "hello", True, None, [1, 2, 3], {"a": 1, "b": 1}]


@pytest.mark.parametrize("input", wrong_dtypes)
def test_create_labels_for_summary_statistics_wrong_dtypes(input):
    with pytest.raises(TypeError):
        create_labels_for_summary_statistics(test_data_datasetdict_headlines=input)


dataframe_wrong_column_text = pd.DataFrame(
    {"incorrect_column": ["text1", "text2"], "label": [0, 1]},
)
dataframe_wrong_column_label = pd.DataFrame(
    {"text": ["text1", "text2"], "no label": [0, 1]},
)
dataset_wrong_column_text = Dataset.from_pandas(dataframe_wrong_column_text)
dataset_wrong_column_label = Dataset.from_pandas(dataframe_wrong_column_label)

datasets_wrong_columns = [dataset_wrong_column_text, dataset_wrong_column_label]


@pytest.mark.parametrize("input", datasets_wrong_columns)
def test_create_labels_for_summary_statistics_wrong_columns(input):
    with pytest.raises(ValueError):
        create_labels_for_summary_statistics(test_data_datasetdict_headlines=input)


dataframe_text_not_string = pd.DataFrame({"text": [1, 2], "label": [0, 1]})
dataset_text_not_string = Dataset.from_pandas(dataframe_text_not_string)


def test_create_labels_for_summary_statistics_wrong_datatype_column_text():
    with pytest.raises(ValueError):
        create_labels_for_summary_statistics(dataset_text_not_string)


dataframe_label_not_int = pd.DataFrame(
    {"text": ["text1", "text2"], "label": ["0", "1"]},
)
dataset_label_not_int = Dataset.from_pandas(dataframe_label_not_int)


def test_create_labels_for_summary_statistics_wrong_datatype_column_label():
    with pytest.raises(ValueError):
        create_labels_for_summary_statistics(dataset_label_not_int)
