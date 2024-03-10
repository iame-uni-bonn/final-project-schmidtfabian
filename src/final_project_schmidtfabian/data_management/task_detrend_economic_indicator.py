from pathlib import Path

import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.data_management.detrend_economic_indicator import (
    detrend_economic_indicator,
)

detrend_economic_activity_deps = {
    "scripts": Path("clean_economic_activity_indicator.py"),
    "data": BLD / "data" / "cleaned_economic_indicator.arrow",
}


def task_detrend_economic_indicator(
    depends_on=detrend_economic_activity_deps,
    produces=BLD / "data" / "detrended_economic_indicator.arrow",
):
    """Detrends the economic activity indicator using the Hodrick-Prescott Filter."""
    cleaned_economic_indicator = pd.read_feather(depends_on["data"])
    detrended_data = detrend_economic_indicator(cleaned_economic_indicator)
    detrended_data.to_feather(produces)
