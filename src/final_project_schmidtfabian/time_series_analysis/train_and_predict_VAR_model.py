from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

def train_VAR_model(merged_train_dataset):
    """Trains VAR model to the provided data.
    
    Trains a VAR model to the provided data assuming daily data and using the Akaike
    Information Criterion to select the correct amount of lags. We need to convert the
    data types in both columns to np.float64 since the VAR function cannot handle
    pd.Float64Dtype().

    Args:
        - merged_train_dataset(pd.DataFrame): The DataFrame the VAR model should be
        trained with.

    Returns:
        - model_VAR_fitted(VAR.VARResultsWrapper)

    """
    _fail_if_wrong_input_training_data(dataframe=merged_train_dataset)
    merged_train_dataset_float64 = merged_train_dataset.astype(np.float64)
    model_VAR = VAR(merged_train_dataset_float64,freq="D")
    model_VAR_fitted = model_VAR.fit(maxlags=15, ic='aic')
    return model_VAR_fitted

def create_summary_statistics_string(VAR_model):
    """Creates string containing summary of VAR model.

    Args:
        - VAR_model(VARResultsWrapper): The results of the training process of the
        VAR model.
    
    Returns:
        - VAR_model_summary_statistics_string(str)

    """
    _fail_if_wrong_dtype(VAR_model=VAR_model)
    VAR_model_summary_statistics = VAR_model.summary()
    VAR_model_summary_statistics_string = str(VAR_model_summary_statistics)

    return VAR_model_summary_statistics_string

def create_predictions_dataframe_VAR_model(merged_train_dataset,VAR_model, test_data):
    """Creates predictions using a VAR model equal to length of test data.

    Creates predictions from the training data using the specified VAR model.
    Forecasts equally many steps ahead as the length of the test data.

    Args:
        - merged_train_dataset(pd.DataFrame): A DataFrame containing the training data.
        - VAR_model(VARResultsWrapper): The results of the fitted VAR model.
        - test_data(pd.DataFrame): A DataFrame containing the test data.

    Returns:
        - VAR_forecasts_dataframe(pd.DataFrame)

    """
    _fail_if_wrong_dtype(VAR_model=VAR_model)
    _fail_if_wrong_input_training_data(merged_train_dataset)
    _fail_if_wrong_input_test_data(dataframe=test_data)
    merged_train_dataset_float64 = merged_train_dataset.astype(np.float64)
    VAR_forecasts = VAR_model.forecast(merged_train_dataset_float64.values
                                       ,steps = len(test_data.index))
    VAR_forecasts_dataframe = pd.DataFrame(data = VAR_forecasts,
                                            index = test_data.index,
                                            columns = ['forecast_cycle_values', 'forecast_sentiment'])
    
    return VAR_forecasts_dataframe


def _fail_if_wrong_dtype(VAR_model):
    if not isinstance(VAR_model, VARResultsWrapper):
        current_datatype = type(VAR_model)
        raise TypeError(f"'VAR_model' has to be an 'VARResultsWrapper', but is currently \
                        '{current_datatype}'")

def _fail_if_wrong_input_training_data(dataframe):
    """Throws an error if the input is invalid."""
    _fail_if_not_dataframe(dataframe=dataframe)
    _fail_if_dataframe_has_less_than_two_columns(dataframe=dataframe)
    _fail_wrong_datatype_columns(dataframe=dataframe)

def _fail_if_wrong_input_test_data(dataframe):
    _fail_if_not_dataframe(dataframe=dataframe)
    _fail_if_dataframe_index_is_empty(dataframe=dataframe)

def _fail_if_not_dataframe(dataframe):
    """Throws an error if the input is not a pandas dataframe."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Input is not a dataframe, it is {type(dataframe)}.")

def _fail_if_dataframe_has_less_than_two_columns(dataframe):
    if dataframe.shape[1] < 2:
        raise ValueError("'train_dataset' has to have more than one column.")
    
def _fail_if_dataframe_index_is_empty(dataframe):
    if dataframe.index.empty:
        raise ValueError("'test_data' has to have a non-empty index.")

def _fail_wrong_datatype_columns(dataframe):
    non_Float64_columns = [col for col, dtype in dataframe.dtypes.items() if not isinstance(dtype, pd.Float64Dtype)]
    if non_Float64_columns:
        raise ValueError(f"The following columns do not have datatype pd.Float64Dtype: {', '.join(non_Float64_columns)}.")