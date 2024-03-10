from pathlib import Path

import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.analysis.analyze_sentiment_headlines import (
    analyze_sentiment_finetuned_model,
)
from final_project_schmidtfabian.config import BLD

analyze_sentiment_deps = {
    "scripts": Path("analyze_sentiment_headlines.py"),
    "data": BLD / "data" / "merged_dataset_headlines_economic_indicator.arrow",
    "model": BLD / "finetuned_text_classification_model" / "info.txt",
}


def task_analyze_sentiment(
    depends_on=analyze_sentiment_deps,
    produces=BLD / "data" / "merged_data_sentiment_analyzed_finetuned_model.arrow",
):
    """Analyzes sentiment of headlines using finetuned model."""
    data_headlines = pd.read_feather(depends_on["data"])
    data_headlines_sentiment_analyzed = analyze_sentiment_finetuned_model(
        data_headlines,
        BLD / "finetuned_text_classification_model" / "checkpoint-4",
    )
    data_headlines_sentiment_analyzed.to_feather(produces)
