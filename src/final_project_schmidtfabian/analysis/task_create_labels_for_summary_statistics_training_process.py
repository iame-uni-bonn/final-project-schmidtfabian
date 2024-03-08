import pytask
from datasets import load_from_disk
import pandas as pd
from pathlib import Path

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.analysis.create_labels_for_summary_statistics\
    import create_labels_for_summary_statistics

create_labels_deps = {
    "scripts": Path("create_labels_for_summary_statistics.py"),
    "dataset dictionary" : BLD / "data" / "dataset_dict.json",
    "finetuned model output" : BLD / "finetuned_text_classification_model" / "info.txt"
}

def task_create_labels_for_summary_statistics_training_process(
        depends_on = create_labels_deps,
        produces = BLD / "data" / "dataframe_labels_test_data.arrow"
        ):
    datasetdict_headlines = load_from_disk(BLD / "data")
    test_data_datasetdict_headlines = datasetdict_headlines["test"]
    dataframe_labels_test_data = create_labels_for_summary_statistics(
        test_data_datasetdict_headlines=test_data_datasetdict_headlines
        )
    dataframe_labels_test_data.to_feather(produces)
