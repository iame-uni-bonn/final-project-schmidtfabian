import pandas as pd
import pytest
from datasets import Dataset, DatasetDict
from final_project_schmidtfabian.analysis.train_text_classification_model import (
    create_trainer_for_text_classification_model,
)
from final_project_schmidtfabian.config import BLD

wrong_dtypes = [5, 1.2, "hello", True, None, [1, 2, 3], {"a": 1, "b": 1}]

dataframe_correct_values = pd.DataFrame({"text": ["text1", "text2"], "label": [0, 1]})

data_dict = {
    "train": Dataset.from_pandas(dataframe_correct_values),
    "validation": Dataset.from_pandas(dataframe_correct_values),
}
test_dataset_dict = DatasetDict(data_dict)


@pytest.mark.parametrize("input", wrong_dtypes)
def test_create_trainer_for_text_classification_model_wrong_dtype_saving_location(
    input,
):
    with pytest.raises(TypeError):
        create_trainer_for_text_classification_model(
            saving_location_model=input,
            dataset_dict_headlines=test_dataset_dict,
        )


@pytest.mark.parametrize("input", wrong_dtypes)
def test_create_trainer_for_text_classification_model_wrong_dtype_dataset_dict_headlines(
    input,
):
    with pytest.raises(TypeError):
        create_trainer_for_text_classification_model(
            saving_location_model=BLD,
            dataset_dict_headlines=input,
        )


data_dict_no_validation = {
    "train": Dataset.from_pandas(dataframe_correct_values),
    "test": Dataset.from_pandas(dataframe_correct_values),
}
dataset_dict_no_validation = DatasetDict(data_dict_no_validation)


def test_create_trainer_for_text_classification_model_wrong_datasets():
    with pytest.raises(ValueError):
        create_trainer_for_text_classification_model(
            saving_location_model=BLD,
            dataset_dict_headlines=dataset_dict_no_validation,
        )
