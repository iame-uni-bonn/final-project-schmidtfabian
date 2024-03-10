import pytest
import pandas as pd
import pmdarima as pm

from final_project_schmidtfabian.time_series_analysis.train_arima_model import\
train_arima_model, create_summary_statistics_dataframe, create_predictions_dataframe_ARIMA_model

wrong_dtypes = [5, 1.2, "hello", True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", wrong_dtypes)
def test_train_arima_model_wrong_dtype(input):
    with pytest.raises(TypeError):
        train_arima_model(train_dataset=input)

@pytest.mark.parametrize("input", wrong_dtypes)
def test_create_summary_statistics_dataframe_wrong_dtype(input):
    with pytest.raises(TypeError):
        create_summary_statistics_dataframe(input)

dummy_arima_object = pm.ARIMA(order=(2,0,1))

@pytest.mark.parametrize("input", wrong_dtypes)
def test_create_predictions_dataframe_ARIMA_model_wrong_dtype_test_data(input):
    with pytest.raises(TypeError):
        create_predictions_dataframe_ARIMA_model(ARIMA_model=dummy_arima_object,
                                                 test_data=input)

def test_create_predictions_dataframe_ARIMA_model_empty_index_test_data():
    with pytest.raises(ValueError):
        create_predictions_dataframe_ARIMA_model(ARIMA_model=dummy_arima_object,
                                                 test_data=pd.DataFrame(columns=['A', 'B']))
