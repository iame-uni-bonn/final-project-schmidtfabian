import pytask
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.final.write_value_to_file import write_value_to_file

adf_statistic_deps = {
    "train dataframe merged data" : BLD / "data" / "train_data_time_series.arrow",
    "scripts" : Path("write_value_to_file.py")
}
for category in ["cycle_values", "sentiment"]:
    adf_statistics_products = {
        "ADF statistic p-value" : BLD / "output" / f"p_value_ADF_statistic_{category}.txt",
    }

    @pytask.task(id=category)
    def task_calculate_adf_statistic(
            depends_on = adf_statistic_deps,
            produces = adf_statistics_products,
            category = category
    ):
        train = pd.read_feather(depends_on["train dataframe merged data"])
        ADF_result_cycle_values = adfuller(train[f"{category}"].values)
        p_value_ADF_result_cycle_values = ADF_result_cycle_values[1]
        write_value_to_file(p_value_ADF_result_cycle_values,produces["ADF statistic p-value"])