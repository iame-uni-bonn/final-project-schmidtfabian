import pytask
from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.analysis.train_text_classification_model import \
    create_trainer_for_text_classification_model
from final_project_schmidtfabian.final.write_value_to_file import write_value_to_file

train_classification_model_deps = {
    "dataset dictionary" : BLD / "data" / "dataset_dict.json",
    "scripts": Path("train_text_classification_model.py")
}

def task_train_text_classification_model(
        depends_on = train_classification_model_deps,
        produces = BLD / "finetuned_text_classification_model" / "info.txt"
        ):
    datasetdict_headlines = load_from_disk(BLD / "data")
    trainer_text_classification_model = create_trainer_for_text_classification_model(
        BLD / "finetuned_text_classification_model",
        dataset_dict_headlines=datasetdict_headlines)
    trainer_text_classification_model.train()
    write_value_to_file("The fine-tuned model of this project is located in this folder.",produces)