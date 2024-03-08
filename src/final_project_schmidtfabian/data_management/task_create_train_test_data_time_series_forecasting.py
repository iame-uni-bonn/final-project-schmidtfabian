import pytask
import pandas as pd
from pmdarima.model_selection import train_test_split

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD

create_train_test_products = {
    "train data time-series-forecasting" : BLD / "data" / "train_data_time_series.arrow",
    "test data time-series-forecasting" : BLD / "data" / "test_data_time_series.arrow",
}
def task_create_train_test_data_time_series_forecasting(
        depends_on = BLD / "data" / "merged_data_sentiment_analyzed_finetuned_model.arrow",
        produces = create_train_test_products):
    merged_dataframe = pd.read_feather(depends_on)
    train, test = train_test_split(
        merged_dataframe[["cycle_values","sentiment"]], test_size=25)
    train.to_feather(produces["train data time-series-forecasting"])
    test.to_feather(produces["test data time-series-forecasting"])
    
