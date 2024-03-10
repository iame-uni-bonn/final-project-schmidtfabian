import pandas as pd
import pytask
from sklearn.metrics import classification_report, confusion_matrix

from final_project_schmidtfabian.config import BLD

create_summary_statistics_deps = {
    "test data headlines labeled": BLD / "data" / "dataframe_labels_test_data.arrow",
}
for model in ["finetuned", "zero_shot_classification"]:
    create_summary_statistics_products = {
        "classification report": BLD
        / "tables"
        / f"classification_report_{model}_model.tex",
        "confusion matrix": BLD / "tables" / f"confusion_matrix_{model}_model.tex",
    }

    @pytask.task(id=model)
    def task_create_summmary_statistics_training_process(
        depends_on=create_summary_statistics_deps,
        produces=create_summary_statistics_products,
        model=model,
    ):
        """Creates classification report and confusion matrix for models."""
        dataframe_labels_test_data = pd.read_feather(
            depends_on["test data headlines labeled"],
        )
        classification_report_model = classification_report(
            dataframe_labels_test_data["label"],
            dataframe_labels_test_data[f"label {model} model"],
            output_dict=True,
        )
        classification_report_model_dataframe = pd.DataFrame(
            classification_report_model,
        ).transpose()
        classification_report_model_dataframe.to_latex(
            produces["classification report"],
            float_format="%.2f",
        )
        confusion_matrix_model = confusion_matrix(
            dataframe_labels_test_data["label"],
            dataframe_labels_test_data[f"label {model} model"],
        )
        confusion_matrix_model_dataframe = pd.DataFrame(
            confusion_matrix_model,
            index=["Actual Negative", "Actual Neutral", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"],
        )
        confusion_matrix_model_dataframe.to_latex(
            produces["confusion matrix"],
            float_format="%.2f",
        )
