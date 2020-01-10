"""Create Plots of Cut signals

This module contains functions that can be used to plot various aspects
of the signals.

"""

# Authors: Tim von Hahn <18tcvh@queensu.ca>
#
# License: MIT License


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


def plot_signals_plotly(df, tool_no, cut_name, variables_to_trend):
    # plot just the currents, speeds, and cut_signal

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if tool_no == None:
        df1 = df.reset_index(drop=True)
        title_plot = str(cut_name)

    else:

        df1 = df[df["tool_no"] == tool_no]
        df1 = df1.reset_index(drop=True)

        title_plot = str(cut_name)

    for i in variables_to_trend:
        fig.add_trace(
            go.Scatter(x=df1[i].index.values, y=df1[i], mode="lines", name=i),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=df1["cut_signal"].index.values,
            y=df1["cut_signal"],
            mode="lines",
            name="cut_signal",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df1["tool_no"].index.values,
            y=df1["tool_no"],
            mode="lines",
            name="tool_no",
        ),
        secondary_y=True,
    )

    try:
        fig.add_trace(
            go.Scatter(
                x=df1["speed_stable"].index.values,
                y=df1["speed_stable"],
                mode="lines",
                name="speed_stable",
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=df1["true_metal_cut"].index.values,
                y=df1["true_metal_cut"],
                mode="lines",
                name="true_metal_cut",
            ),
            secondary_y=True,
        )

    except:
        pass

    pio.write_html(fig, file=title_plot, auto_open=True)

