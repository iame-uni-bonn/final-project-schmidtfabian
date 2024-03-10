import pytask
import pandas as pd
from statsmodels.tsa.api import VAR
from pmdarima import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import  ValueWarning

warnings.simplefilter('ignore', ValueWarning)

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD

def task_analyze_forecast_performance_cv(
        depends_on = BLD / "data" / "merged_data_sentiment_analyzed_finetuned_model.arrow",
        produces = BLD / "data" / "mean_squared_errors_time_series_models_cv.arrow"
        ):
    merged_dataset = pd.read_feather(depends_on)
    merged_dataset_float64 = pd.DataFrame()
    merged_dataset_float64["cycle_values"] = merged_dataset["cycle_values"].astype(np.float64)
    merged_dataset_float64["sentiment"] = merged_dataset["sentiment"].astype(np.float64)
    merged_dataset_float64.index = merged_dataset.index
    number_of_splits = 10
    tss = TimeSeriesSplit(n_splits=number_of_splits, test_size=3, gap=0)
    mses_ARIMA_model = []
    mses_VAR_model = []
    for train_idx, val_idx in tss.split(merged_dataset_float64["cycle_values"]):
        train = merged_dataset_float64.iloc[train_idx]
        test = merged_dataset_float64.iloc[val_idx]

        train_ARIMA_model_data = train["cycle_values"]
        train_ARIMA_model_data.index = train.index


        test_data = test["cycle_values"]
        
        ARIMA_model = ARIMA(
                    order=(2,0,1),
                    seasonal_order=(0, 0, 0, 0)
                    )
        ARIMA_model_fitted = ARIMA_model.fit(train_ARIMA_model_data)
        predictions_ARIMA_model = ARIMA_model_fitted.predict(
            n_periods=len(test_data))
        mse_ARIMA_model = mean_squared_error(test_data, predictions_ARIMA_model)
        mses_ARIMA_model.append(mse_ARIMA_model)

        COMPONENTS=["cycle_values","sentiment"]

        train_VAR_model_data = train[COMPONENTS]
        train_VAR_model_data.index = train.index
        
        model = VAR(train_VAR_model_data, freq="D")
        model_fit_VAR=model.fit(maxlags=1)
        forecasts_VAR = model_fit_VAR.forecast(
            train_VAR_model_data.values,
            steps=len(test_data)
            )
        columns = ['forecast_cycle_values', 'forecast_sentiment']
        VAR_forecast_df = pd.DataFrame(forecasts_VAR, columns=columns, index=test_data.index)
        mse_VAR_model = mean_squared_error(test_data, VAR_forecast_df["forecast_cycle_values"])
        mses_VAR_model.append(mse_VAR_model)
        dict_mses = {
            "mses ARIMA model" : mses_ARIMA_model,
            "mses VAR model" : mses_VAR_model
        }
        dataframe_mses_wrong_dtype = pd.DataFrame(dict_mses)
        dataframe_mses = dataframe_mses_wrong_dtype.astype(pd.Float64Dtype())
        dataframe_mses.to_feather(produces)

