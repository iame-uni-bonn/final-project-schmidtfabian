"""Tasks running the core analyses."""

from pathlib import Path

import pandas as pd
import pytask

from final_project_schmidtfabian.analysis.model import fit_logit_model, load_model
from final_project_schmidtfabian.analysis.predict import predict_prob_by_age
from final_project_schmidtfabian.config import BLD, GROUPS, SRC
from final_project_schmidtfabian.utilities import read_yaml

fit_model_deps = {
    "scripts": [Path("model.py"), Path("predict.py")],
    "data": BLD / "python" / "data" / "data_clean.csv",
    "data_info": SRC / "data_management" / "data_info.yaml",
}


def task_fit_model_python(
    depends_on=fit_model_deps,
    produces=BLD / "python" / "models" / "model.pickle",
):
    """Fit a logistic regression model (Python version)."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    model = fit_logit_model(data, data_info, model_type="linear")
    model.save(produces)


for group in GROUPS:
    predict_deps = {
        "data": BLD / "python" / "data" / "data_clean.csv",
        "model": BLD / "python" / "models" / "model.pickle",
    }

    @pytask.task(id=group)
    def task_predict_python(
        group=group,
        depends_on=predict_deps,
        produces=BLD / "python" / "predictions" / f"{group}.csv",
    ):
        """Predict based on the model estimates (Python version)."""
        model = load_model(depends_on["model"])
        data = pd.read_csv(depends_on["data"])
        predicted_prob = predict_prob_by_age(data, model, group)
        predicted_prob.to_csv(produces, index=False)
