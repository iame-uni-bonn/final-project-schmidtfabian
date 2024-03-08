import pmdarima as pm
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

def train_arima_model(train_dataset):
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
    _fail_if_wrong_dtype(ARIMA_model=ARIMA_model)
    ARIMA_model_summary_statistics = ARIMA_model.summary()
    ARIMA_model_summary_statistics_dataframe = pd.DataFrame(ARIMA_model_summary_statistics.tables[1])

    return ARIMA_model_summary_statistics_dataframe

def create_predictions_dataframe_ARIMA_model(ARIMA_model, test_data):
    _fail_if_wrong_dtype(ARIMA_model=ARIMA_model)
    ARIMA_forecasts = ARIMA_model.predict(len(test_data.index))
    ARIMA_forecasts_dataframe = pd.DataFrame(data = ARIMA_forecasts,
                                            index = test_data.index,
                                            columns = ["forecast_cycle_values"])
    
    return ARIMA_forecasts_dataframe


def _fail_if_wrong_dtype(ARIMA_model):
    if not isinstance(ARIMA_model, pm.ARIMA):
        current_datatype = type(ARIMA_model)
        raise TypeError(f"'ARIMA_model' has to be an 'pm.ARIMA' object, but is currently \
                        '{current_datatype}'")