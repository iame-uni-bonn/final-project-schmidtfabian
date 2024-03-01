import pytask
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD

merge_datasets_deps = {
    "dataset_headlines" : BLD / "data" / "dataset_headlines.arrow",
    "dataset_economic_indicator": BLD / "data" / "detrended_economic_indicator.arrow"
}

def task_merge_datasets(
        depends_on = merge_datasets_deps,
        produces = BLD / "data" / "merged_dataset_headlines_economic_indicator.arrow"
        ):
    dataframe_headlines = pd.read_feather(depends_on["dataset_headlines"])
    dataframe_economic_indicator = pd.read_feather(depends_on["dataset_economic_indicator"])
    merged_dataframe = dataframe_headlines.merge(dataframe_economic_indicator, left_index=True, right_index=True, how='inner')
    merged_dataframe.to_feather(produces)