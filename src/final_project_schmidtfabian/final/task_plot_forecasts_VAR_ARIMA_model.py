import pytask
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_squared_error

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.final.write_value_to_file import write_value_to_file

for model in ["VAR", "ARIMA"]:
    plot_forecasts_deps = {
        "train dataframe merged data" : BLD / "data" / "train_data_time_series.arrow",
        "test dataframe merged data" : BLD / "data" / "test_data_time_series.arrow",
        "forecasts data" : BLD / "data" / f"data_forecast_{model}_model.arrow"
    }
    plot_forecasts_products = {
        "plot forecasts" : BLD / "figures" / f"{model}_forecasts_cycle.png",
        "mse seven days forecast" : BLD / "output" / f"mse_{model}_seven_day_forecasts.txt"
    }
    @pytask.task(id=model)
    def task_plot_forecasts_VAR_ARIMA_model(
        depends_on = plot_forecasts_deps,
        produces = plot_forecasts_products,
        model = model
    ):
        train = pd.read_feather(depends_on["train dataframe merged data"])
        test = pd.read_feather(depends_on["test dataframe merged data"])
        model_forecasts = pd.read_feather(depends_on["forecasts data"])
        trace_training_data = go.Scatter(x=train.index, y=train["cycle_values"], mode='lines',
                            name='training data', line=dict(color='blue'))
        
        trace_forecast_data = go.Scatter(x=model_forecasts.index, y=model_forecasts["forecast_cycle_values"],
                            mode='lines', name='forecasts', line=dict(color='green'))
        
        trace_test_data = go.Scatter(x=test.index, y=test["cycle_values"], mode='lines', name='test data',
                            line=dict(color='red'))

        figure_layout = go.Layout(
            title=f'Test vs. Forecast Values: Truck Toll Mileage Index || {model} model',
            xaxis=dict(title='Dates'),
            yaxis=dict(title='Cycle Values Truck Toll Mileage Index'),
            showlegend=True
        )
        figure_forecasts_model = go.Figure(
            data=[trace_training_data, trace_forecast_data, trace_test_data],
            layout=figure_layout
            )
        figure_forecasts_model.write_image(produces["plot forecasts"])

        mse_forecasts_seven_days = mean_squared_error(
            test["cycle_values"][:6],
            model_forecasts["forecast_cycle_values"][:6])
        write_value_to_file(mse_forecasts_seven_days, produces["mse seven days forecast"])