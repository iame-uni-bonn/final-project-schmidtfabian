import pmdarima as pm
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

def train_arima_model(train_dataset):
    """Trains ARIMA model to the provided data.
    
    Trains an ARIMA model using the auto_arima function of the pmdarima library.
    Uses the Augmented Dickey-Fuller test to determine the inclusion of a trend
    component. The data should be daily, and not seasonal. Maximum number of lags
    and moving-average components is 15.

    Args:
        - train_dataset(pd.Series): DataSeries used for training ARIMA model.

    Returns:
        - ARIMAmodel(pm.ARIMA)

    """
    _fail_if_wrong_input_training(series=train_dataset)
    ARIMAmodel = pm.auto_arima(train_dataset, start_p=0, start_q=0,
                      test='adf',
                      max_p=15, max_q=15,
                      m=1,
                      seasonal=False,
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    
    return ARIMAmodel

def create_summary_statistics_dataframe(ARIMA_model):
    """Creates summary statistics of ARIMA model and saves them in a DataFrame.
    
    Args:
        - ARIMA_model(pm.ARIMA): The ARIMA model for which summary statistics should be
        created.
    
    Returns:
        - ARIMA_model_summary_statistics_dataframe(pd.DataFrame)

    """
    _fail_if_wrong_dtype(ARIMA_model=ARIMA_model)
    ARIMA_model_summary_statistics = ARIMA_model.summary()
    ARIMA_model_summary_statistics_dataframe = pd.DataFrame(ARIMA_model_summary_statistics.tables[1])

    return ARIMA_model_summary_statistics_dataframe

def create_predictions_dataframe_ARIMA_model(ARIMA_model, test_data):
    """Createss predictions using ARIMA model and puts them into a DataFrame.
    
    Creates predictions using the trained ARIMA model trained with the training data
    and adds predictions to a pandas DataFrame. The amount of steps predicted ahead is
    equal to the length of the provided test data.

    Args:
        - ARIMA_model(pm.ARIMA): The trained ARIMA model that should be used to make
        predictions.
        - test_data(pd.DataFrame): The dataframe containing the test data.

    Returns:
        -ARIMA_forecasts_dataframe(pd.DataFrame)

    """
    _fail_if_wrong_dtype(ARIMA_model=ARIMA_model)
    _fail_if_wrong_input_test(dataframe=test_data)
    ARIMA_forecasts = ARIMA_model.predict(len(test_data.index))
    ARIMA_forecasts_dataframe = pd.DataFrame(data = ARIMA_forecasts,
                                            index = test_data.index,
                                            columns = ["forecast_cycle_values"])
    
    return ARIMA_forecasts_dataframe


def _fail_if_wrong_dtype(ARIMA_model):
    """Throws an error if argument is not an 'ARIMA' object."""
    if not isinstance(ARIMA_model, pm.ARIMA):
        current_datatype = type(ARIMA_model)
        raise TypeError(f"'ARIMA_model' has to be an 'pm.ARIMA' object, but is currently \
                        '{current_datatype}'")

def _fail_if_wrong_input_training(series):
    """Throws an error if the input is invalid."""
    _fail_if_not_series(series)

def _fail_if_wrong_input_test(dataframe):
    _fail_if_not_dataframe(dataframe=dataframe)
    _fail_if_dataframe_index_is_empty(dataframe=dataframe)

def _fail_if_not_series(series):
    """Throws an error if the input is not a pandas series."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"Input is not a series, it is {type(series)}.")
    
def _fail_if_not_dataframe(dataframe):
    """Throws an error if the input is not a pandas series."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Input is not a DataFrame, it is {type(dataframe)}.")
    
def _fail_if_dataframe_index_is_empty(dataframe):
    if dataframe.index.empty:
        raise ValueError("'test_data' has to have a non-empty index.")