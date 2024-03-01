import pandas as pd
import pytask
from pathlib import Path

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD, SRC
from final_project_schmidtfabian.data_management.clean_economic_activity_indicator import clean_economic_activity_indicator

import_economic_activity_deps = {
    "scripts": Path("clean_economic_activity_indicator.py"),
    "data": SRC / "data" / "Lkw-Maut-Fahrleistungsindex-Daten.xlsx",
}

def task_import_economic_indicator(
        depends_on = import_economic_activity_deps,
        produces = BLD / "data" / "cleaned_economic_indicator.arrow"
        ):
    """Imports and cleans the data containing the economic activity indicator and then saves it in a new file format."""
    economic_indicator_dataframe = pd.read_excel(depends_on["data"], sheet_name=1, usecols="A,E",skiprows=5)
    cleaned_data = clean_economic_activity_indicator(economic_indicator_dataframe)
    cleaned_data.to_feather(produces)