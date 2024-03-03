import pytask
import pandas as pd
from sklearn.metrics import mean_squared_error

from final_project_schmidtfabian.config import SRC, BLD
from final_project_schmidtfabian.final.write_value_to_file import write_value_to_file

mse_zero_shot_classification_deps = {
    "handlabeled dataframe": SRC / "final_project_schmidtfabian" / "data" / "newspaperheadlines_manually_labeled.xlsx",
    "data_headlines_sentiment_analyzed_zero_shot_classification_model" : BLD / "data" / "headlines_sentiment_analyzed_zero_shot_classification.arrow"
}

def task_calculate_mse_zero_shot_classification_model(
        depends_on = mse_zero_shot_classification_deps,
        produces = BLD / "output" / "mse_zero_shot_classification.txt"):
    handlabled_dataframe_headlines=pd.read_excel(depends_on["handlabeled dataframe"], index_col=0, nrows=30)
    label_mapping = {
        "negative": 1,
        "neutral": 2,
        "positive": 3
    }
    handlabled_dataframe_headlines['label'] = handlabled_dataframe_headlines['sentiment manually label'].map(label_mapping)
    data_headlines_sentiment_analyzed = pd.read_feather(depends_on["data_headlines_sentiment_analyzed_zero_shot_classification_model"])
    mse_daily_zeroshot= mean_squared_error(data_headlines_sentiment_analyzed["sentiment"][:30], handlabled_dataframe_headlines['label'])
    write_value_to_file(mse_daily_zeroshot,produces)


