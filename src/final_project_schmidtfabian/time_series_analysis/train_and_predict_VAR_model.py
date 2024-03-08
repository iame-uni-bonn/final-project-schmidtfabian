from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

def train_VAR_model(merged_train_dataset):
    model_VAR = VAR(merged_train_dataset,freq="D")
    model_VAR_fitted = model_VAR.fit(maxlags=15, ic='aic')
    return model_VAR_fitted

def create_summary_statistics_string(VAR_model):
    _fail_if_wrong_dtype(VAR_model=VAR_model)
    VAR_model_summary_statistics = VAR_model.summary()
    VAR_model_summary_statistics_string = str(VAR_model_summary_statistics)

    return VAR_model_summary_statistics_string

def create_predictions_dataframe_VAR_model(merged_train_dataset,VAR_model, test_data):
    _fail_if_wrong_dtype(VAR_model=VAR_model)
    VAR_forecasts = VAR_model.forecast(merged_train_dataset.values ,steps = len(test_data.index))
    VAR_forecasts_dataframe = pd.DataFrame(data = VAR_forecasts,
                                            index = test_data.index,
                                            columns = ['forecast_cycle_values', 'forecast_sentiment'])
    
    return VAR_forecasts_dataframe


def _fail_if_wrong_dtype(VAR_model):
    if not isinstance(VAR_model, VARResultsWrapper):
        current_datatype = type(VAR_model)
        raise TypeError(f"'VAR_model' has to be an 'VAR' object, but is currently \
                        '{current_datatype}'")