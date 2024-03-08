import pytask
import pandas as pd
from pathlib import Path

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.time_series_analysis.train_and_predict_VAR_model import train_VAR_model, \
create_predictions_dataframe_VAR_model, create_summary_statistics_string
from final_project_schmidtfabian.final.write_value_to_file import write_value_to_file

VAR_model_deps = {
    "train data time-series-forecasting" : BLD / "data" / "train_data_time_series.arrow",
    "test data time-series-forecasting" : BLD / "data" / "test_data_time_series.arrow",
    "scripts" : Path("train_and_predict_VAR_model.py")
}

VAR_model_products = {
    "summary statistics table" : BLD / "tables" / "summary_statistics_VAR_model.txt",
    "forecast dataframe VAR model" : BLD / "data" / "data_forecast_VAR_model.arrow"
}

def task_train_and_predict_VAR_model(
        depends_on = VAR_model_deps,
        produces = VAR_model_products):
    train = pd.read_feather(depends_on["train data time-series-forecasting"])
    test = pd.read_feather(depends_on["test data time-series-forecasting"])
    VAR_model = train_VAR_model(merged_train_dataset=train)
    VAR_model_summary_statistics = create_summary_statistics_string(VAR_model=VAR_model)
    write_value_to_file(VAR_model_summary_statistics,produces["summary statistics table"])
    VAR_forecasts_dataframe = create_predictions_dataframe_VAR_model(
        merged_train_dataset=train,
        VAR_model=VAR_model,
        test_data=test
        )
    VAR_forecasts_dataframe.to_feather(produces["forecast dataframe VAR model"])