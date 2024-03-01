import pandas as pd
from pathlib import Path
import pytask

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.analysis.analyze_sentiment_zero_classification import analyze_sentiment_zero_classification

analyze_sentiment_deps = {
    "scripts": Path("clean_economic_activity_indicator.py"),
    "data": BLD / "data" / "data_headlines.arrow",
}

def task_analyze_sentiment(
        depends_on = analyze_sentiment_deps,
        produces = BLD / "data" / "headlines_sentiment_analyzed_zero_shot_classification.arrow"
        ):
    data_headlines = pd.read_feather(depends_on["data"])
    data_headlines_sentiment_analyzed = analyze_sentiment_zero_classification(data_headlines)
    data_headlines_sentiment_analyzed.to_feather(produces)