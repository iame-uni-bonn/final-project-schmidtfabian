import pandas as pd
import pytest

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.analysis.analyze_sentiment_headlines import (
    analyze_sentiment_finetuned_model,
    analyze_sentiment_zero_shot_classification,
)
from final_project_schmidtfabian.config import BLD

wrong_dtypes = [5, 1.2, "hello", True, None, [1, 2, 3], {"a": 1, "b": 1}]


@pytest.mark.parametrize("input", wrong_dtypes)
def test_analyze_sentiment_zero_shot_classification_wrong_dtypes(input):
    with pytest.raises(TypeError):
        analyze_sentiment_zero_shot_classification(input)


@pytest.mark.parametrize("input", wrong_dtypes)
def test_analyze_sentiment_finetuned_wrong_dtypes_dataframe_headlines(input):
    with pytest.raises(TypeError):
        analyze_sentiment_finetuned_model(
            dataframe_headlines=input,
            location_finetuned_model=BLD,
        )


headlines = [
    "Local man invents teleportation device, accidentally teleports into zoo's lion enclosure",
    "Study finds that pizza is a better motivator than money",
    "Aliens invite Earth to join Intergalactic Dance-Off Competition",
    "Mayor declares every Friday 'Wear Your Pajamas to Work Day'",
    "Scientists discover that cats can speak fluent French, choose not to",
    "New smartphone app helps you find your soulmate based on pizza preferences",
    "World's first flying pig spotted over downtown",
    "Breaking News: Bananas officially declared Earth's funniest fruit",
    "Local grandma wins rap battle against famous rapper",
    "Researchers develop time machine, go back to prevent invention of pineapple pizza",
]
test_dataframe_headlines = pd.DataFrame({"headlines in a list": [headlines]})


@pytest.mark.parametrize("input", wrong_dtypes)
def test_analyze_sentiment_finetuned_wrong_dtypes_location_finetuned_model(input):
    with pytest.raises(TypeError):
        analyze_sentiment_finetuned_model(
            dataframe_headlines=test_dataframe_headlines,
            location_finetuned_model=input,
        )


numbers = [1, 2, 3, 4, 5]

test_dataframe_wrong_column = pd.DataFrame({"numbers": [numbers]})


def test_analyze_sentiment_zero_shot_classification_wrong_columns():
    with pytest.raises(ValueError):
        analyze_sentiment_zero_shot_classification(test_dataframe_wrong_column)


def test_analyze_sentiment_finetuned_wrong_columns():
    with pytest.raises(ValueError):
        analyze_sentiment_finetuned_model(
            dataframe_headlines=test_dataframe_wrong_column,
            location_finetuned_model=BLD,
        )


test_dataframe_wrong_datatypes_elements = pd.DataFrame(
    {"headlines in a list": [numbers]},
)


def test_analyze_sentiment_zero_shot_classification_wrong_datatype_elements_of_column():
    with pytest.raises(ValueError):
        analyze_sentiment_zero_shot_classification(
            test_dataframe_wrong_datatypes_elements,
        )


def test_analyze_sentiment_finetuned_model_wrong_datatype_elements_of_column():
    with pytest.raises(ValueError):
        analyze_sentiment_finetuned_model(
            dataframe_headlines=test_dataframe_wrong_datatypes_elements,
            location_finetuned_model=BLD,
        )
