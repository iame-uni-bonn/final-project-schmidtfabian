import pandas as pd
import pytest
from final_project_schmidtfabian.time_series_analysis.train_and_predict_VAR_model import (
    create_predictions_dataframe_VAR_model,
    create_summary_statistics_string,
    train_VAR_model,
)
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

wrong_dtypes = [5, 1.2, "hello", True, None, [1, 2, 3], {"a": 1, "b": 1}]


@pytest.mark.parametrize("input", wrong_dtypes)
def test_train_and_predict_VAR_model_wrong_dtype(input):
    with pytest.raises(TypeError):
        train_VAR_model(merged_train_dataset=input)


def test_train_and_predict_VAR_model_only_one_column():
    with pytest.raises(ValueError):
        train_VAR_model(merged_train_dataset=pd.DataFrame({"A": [1, 2, 3]}))


def test_train_and_predict_VAR_model_wrong_datatype_columns():
    with pytest.raises(ValueError):
        train_VAR_model(
            merged_train_dataset=pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        )


@pytest.mark.parametrize("input", wrong_dtypes)
def test_create_summary_statistics_string_wrong_dtype(input):
    with pytest.raises(TypeError):
        create_summary_statistics_string(input)


train_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
dummy_varresultswrapper = VARResultsWrapper(123)


@pytest.mark.parametrize("input", wrong_dtypes)
def test_create_predictions_dataframe_VAR_model_wrong_dtype(input):
    with pytest.raises(TypeError):
        create_predictions_dataframe_VAR_model(
            merged_train_dataset=train_data,
            VAR_model=dummy_varresultswrapper,
            test_data=input,
        )


test_data_empty_index = pd.DataFrame(columns=["A", "B"])


def test_create_predictions_dataframe_VAR_model_wrong_dtype():
    with pytest.raises(ValueError):
        create_predictions_dataframe_VAR_model(
            merged_train_dataset=train_data,
            VAR_model=dummy_varresultswrapper,
            test_data=test_data_empty_index,
        )
