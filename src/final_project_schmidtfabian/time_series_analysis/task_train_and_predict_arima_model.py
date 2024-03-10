from pathlib import Path

import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.time_series_analysis.train_arima_model import (
    create_predictions_dataframe_ARIMA_model,
    create_summary_statistics_dataframe,
    train_arima_model,
)

arima_model_deps = {
    "train data time-series-forecasting": BLD / "data" / "train_data_time_series.arrow",
    "test data time-series-forecasting": BLD / "data" / "test_data_time_series.arrow",
    "scripts": Path("train_arima_model.py"),
}

arima_model_products = {
    "summary statistics table": BLD / "tables" / "summary_statistics_ARIMA_model.tex",
    "forecast dataframe ARIMA model": BLD / "data" / "data_forecast_ARIMA_model.arrow",
}


def task_train_and_predict_arima_model(
    depends_on=arima_model_deps,
    produces=arima_model_products,
):
    """Trains ARIMA model on time-series data and then makes predictions."""
    train = pd.read_feather(depends_on["train data time-series-forecasting"])
    test = pd.read_feather(depends_on["test data time-series-forecasting"])
    ARIMA_model = train_arima_model(train_dataset=train["cycle_values"])
    ARIMA_model_summary_statistics_dataframe = create_summary_statistics_dataframe(
        ARIMA_model=ARIMA_model,
    )
    ARIMA_model_summary_statistics_dataframe.to_latex(
        produces["summary statistics table"],
        index=False,
    )
    ARIMA_forecasts_dataframe = create_predictions_dataframe_ARIMA_model(
        ARIMA_model=ARIMA_model,
        test_data=test,
    )
    ARIMA_forecasts_dataframe.to_feather(produces["forecast dataframe ARIMA model"])
