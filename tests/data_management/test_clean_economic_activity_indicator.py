import pytest
import pandas as pd

from final_project_schmidtfabian.data_management import clean_economic_activity_indicator

wrong_dtypes = [5, 1.2, "hello", True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", wrong_dtypes)
def test_clean_economic_indicator_wrong_dtypes(input):
    with pytest.raises(TypeError):
        clean_economic_activity_indicator(input)

test_dictionary_wrong_columns =  {"Datum gestern": ["25.02.2024", "26.02.2024", "27.02.2024"], "Kalender-, aber nicht saisonbereinigt (KNSB)":[100.5,99.7,95.4]}
test_dataframe_wrong_columns = pd.DataFrame(data = test_dictionary_wrong_columns)

def test_clean_economic_indicator_wrong_columns():
    with pytest.raises(ValueError):
        clean_economic_activity_indicator(test_dataframe_wrong_columns)


test_dictionary_correct_columns =  {"Datum": ["25.02.2024", "26.02.2024", "27.02.2024"], "Kalender- und saisonbereinigt (KSB)":[100.5,99.7,95.4]}
test_dataframe_correct_columns = pd.DataFrame(test_dictionary_correct_columns)

def test_clean_economic_indicator_correct_dtypes():
    cleaned_data = clean_economic_activity_indicator(test_dataframe_correct_columns)
    assert cleaned_data["date"].dtype == "datetime" and cleaned_data["values"].dtype == "float64"

def test_clean_economic_indicator_rename():
    cleaned_data = clean_economic_activity_indicator(test_dataframe_correct_columns)
    assert cleaned_data.names == ["date", "values"]

def test_clean_economic_indicator_correct_index():
    cleaned_data = clean_economic_activity_indicator(test_dataframe_correct_columns)
    assert cleaned_data.index == ["25.02.2024", "26.02.2024", "27.02.2024"]