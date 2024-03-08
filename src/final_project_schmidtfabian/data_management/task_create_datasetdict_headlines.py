import pytask
from datasets import Dataset, DatasetDict
import pandas as pd
from pathlib import Path

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import SRC, BLD
from final_project_schmidtfabian.data_management.create_datasetdict import create_datasetdict

create_datasetdict_deps = {
    "data handlabeled": SRC / "data" / "headlinesdatasetfinetuninghandlabeled.xlsx",
    "scripts" : Path("create_datasetdict.py")
}

def task_create_datasetdict_for_training(
        depends_on = create_datasetdict_deps,
        produces = BLD/"data"/"dataset_dict.json"
        ):
    handlabeled_dataset_headlines = pd.read_excel(depends_on["data handlabeled"],sheet_name=0, usecols="A,B",skiprows=1392, header=None, names=["text","label"], nrows=300)
    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    handlabeled_dataset_headlines['label'] = handlabeled_dataset_headlines['label'].map(label_mapping)
    datasetdict_headlines = create_datasetdict(handlabeled_dataset_headlines)
    datasetdict_headlines.save_to_disk(BLD/"data")