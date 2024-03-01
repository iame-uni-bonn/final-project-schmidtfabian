import pytest
import pandas as pd

from final_project_schmidtfabian.data_management import detrend_economic_indicator

WRONG_DTYPES = [5, 1.2, "hello", True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", WRONG_DTYPES)
def test_clean_economic_indicator_wrong_dtypes(input):
    with pytest.raises(TypeError):
        detrend_economic_indicator(input)

test_dictionary_wrong_columns =  {"Datum": ["25.02.2024", "26.02.2024", "27.02.2024"], "Kalender- und saisonbereinigt (KSB)":[100.5,99.7,95.4]}
test_dataframe_wrong_columns = pd.DataFrame(test_dictionary_wrong_columns)

def test_clean_economic_indicator_wrong_columns():
    with pytest.raises(ValueError):
        detrend_economic_indicator(test_dataframe_wrong_columns)