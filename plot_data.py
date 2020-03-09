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


def plot_signals_plotly(
    df,
    tool_no,
    cut_name,
    variables_to_trend=["current_main", "current_sub", "speed_main", "speed_sub"],
    y_axis_showgrid=True,
    x_axis_showgrid=True,
    shaded_cut_region=True,
):
    # plot just the currents, speeds, and cut_signal

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if tool_no == None:
        df1 = df.reset_index(drop=True)
        title_plot = str(cut_name)

        # finded the indices for the shaded region where metal cutting occurs
        try:
            v = np.array(df1["cut_signal"])
            cut_signal_index = np.where(v[:-1] != v[1:])[0]
        except:
            pass

    else:

        df1 = df[df["tool_no"] == tool_no]
        df1 = df1.reset_index(drop=True)

        # finded the indices for the shaded region where metal cutting occurs
        try:
            v = np.array(df1["cut_signal"])
            cut_signal_index = np.where(v[:-1] != v[1:])[0]
        except:
            pass

        title_plot = str(cut_name)

    for i in variables_to_trend:
        fig.add_trace(
            go.Scatter(x=df1[i].index.values, y=df1[i], mode="lines", name=i),
            secondary_y=False,
        )

    try:
        fig.add_trace(
            go.Scatter(
                x=df1["cut_signal"].index.values,
                y=df1["cut_signal"],
                mode="lines",
                name="cut_signal",
            ),
            secondary_y=True,
        )
    except:
        pass

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

    if shaded_cut_region == True:
        # add shaded region for representing cut area https://plot.ly/python/shapes/
        shaded_region_list = []
        for cut_region in range(0, int(len(cut_signal_index) / 2)):
            shaded_region_list.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=cut_signal_index[cut_region * 2],
                    y0=0,
                    x1=cut_signal_index[cut_region * 2 + 1],
                    y1=1,
                    fillcolor="grey",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            )

        fig.update_layout(shapes=shaded_region_list)

    # hide secondary grid https://community.plot.ly/t/hide-grid-of-secondary-y-axis/28781
    fig["layout"]["yaxis2"]["showgrid"] = False
    fig["layout"]["yaxis2"]["visible"] = True
    fig["layout"]["yaxis"]["visible"] = True
    fig["layout"]["xaxis"]["visible"] = True
    fig["layout"]["yaxis"]["showgrid"] = y_axis_showgrid
    fig["layout"]["xaxis"]["showgrid"] = x_axis_showgrid
    fig["layout"]["plot_bgcolor"] = "white"

    # adjust theme https://plot.ly/python/templates/
    fig.update_layout(template="plotly_white")
    # fig.update_layout(template="seaborn")

    pio.write_html(fig, file=title_plot, auto_open=True)


def plot_features_by_average_index_plotly(df, feat_to_trend, tool_no=54, index_list=[1,2],chart_height=9000):
    '''Function to plot the feature table results'''
    
    def convert_to_datetime(cols):
        unix_date = cols[0]
        value = datetime.fromtimestamp(unix_date)
        return value
    
    df = df[(df['tool']==tool_no) & (df['index'].isin(index_list))].groupby(['unix_date'], as_index=False).mean()
    df = df.reset_index(drop=True).sort_values('unix_date')
    df['date'] = df[['unix_date']].apply(convert_to_datetime, axis=1)
    df['date_ymd'] = pd.to_datetime(df['date'],unit='s').dt.to_period('D')
    df['failed'].fillna(0, inplace=True, downcast='int')
    df.to_csv('groupby_csv_test.csv')
    
    # get date-changes
    # https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy
    v=np.array(df['date_ymd'], dtype=datetime)
    date_change_list = np.where(v[:-1] != v[1:])[0]
    
    feat_list = []
    feat_title_list = []
    for feat in feat_to_trend:
        feat_list.append([feat, feat_to_trend[feat]])
        feat_title_list.append(feat_to_trend[feat])

    index_str = ''
    index_str_file = ''
    for i in index_list:
        index_str += str(i)+', '
        index_str_file += str(i)+'_'

    title_chart = 'Features for Tool {}, Averaged Across Splits (splits on metal-cutting)'.format(tool_no)

#     title_chart = 'Features for Tool {}, Averaged Across Cut Splits {}'.format(tool_no, index_str[:-2])

    file_name = 'tool_{}_avg_splits_{}.html'.format(tool_no,index_str_file[:-1])

    cols = 1
    rows = int(len(feat_to_trend) / cols) 
    if (len(feat_to_trend) % cols) != 0:
        rows += 1

    # Initialize figure with subplots
    fig = make_subplots(rows=rows, cols=cols,subplot_titles=feat_title_list)  

    # get lenght of features
    l = len(feat_to_trend)
    len_data = len(df)

    # iterate through each feature number and plot on chart
    counter = 0
    for i in range(rows):
        for j in range(cols):
            if counter < l:
                trend_value = feat_list[counter][0]
                title_plot = feat_title_list[counter]
                
                min_plot_val = np.min(df[trend_value])
                max_plot_val = np.max(df[trend_value])
                len_trend_data = len(df[trend_value])

                if title_plot != feat_title_list[counter]:

                    print('ERROR IN SUB-PLOT TITLE')

                for failed_label in range(0,3):
                    df1 = df[df['failed'] == failed_label].copy()
                    
                    # define the colors for the markers
                    # can find default colors for theme via:
                    # >> plotly_template = pio.templates["plotly_white"]
                    # >> plotly_template.layout
                    
                    color_label = ['#636efa','#EF553B','#00cc96','#00cc96']

                    x = df1.index.values
                    y = df1[trend_value]
                    p = df1[['unix_date','date']].values
                    point_name=[]
                    for d in p:
                        if failed_label == 1:
                            point_name.append('Label: Wear, '+ str(d[1])+", "+str(d[0]))
                        elif failed_label == 2:
                            point_name.append('Label: Ignore, '+ str(d[1])+", "+str(d[0]))
                        else:
                            point_name.append(str(d[1])+", "+str(d[0]))
            
                    # Add traces
                    # how to add hover text, https://plot.ly/python/hover-text-and-formatting/
                    # Scattergl documentation: 
                        # https://plot.ly/python-api-reference/generated/plotly.graph_objects.Scattergl.html
                    fig.add_trace(go.Scattergl(x=x, y=y, mode='markers',
                                               showlegend=False,
                                               hovertext=point_name, 
                                               hoverinfo='text', marker=dict(color = color_label[failed_label])), 
                                  row=(i+1), 
                                  col=(j+1), )
                    
                # add the vertical lines
                for date_change in date_change_list:
                    date_change_text = 'Date change to '+str(df['date_ymd'].to_numpy()[date_change+1])
                    fig.add_trace(go.Scattergl(x=np.ones(10)*date_change,y=np.linspace(min_plot_val,max_plot_val,10), 
                                               mode='lines',
                                               hovertext=date_change_text,
                                               hoverinfo='text',
                                               opacity=0.2,
                                               showlegend=False,  marker=dict(color = 'black')), 
                                  row=(i+1), 
                                  col=(j+1), )
                
                if counter == 0:
                    fig['layout']['yaxis'].update(range=[min_plot_val, max_plot_val])
                    fig['layout']['xaxis'].update(range=[0, len_trend_data])
                else:
                    fig['layout']['yaxis{}'.format(counter)].update(range=[min_plot_val, max_plot_val], 
                                                                    autorange=True)
                    fig['layout']['xaxis{}'.format(counter)].update(range=[-10, len_trend_data+10])

                counter += 1
            else:
                pass

#     fig.update_layout(title_text=title_chart, template="plotly_white", autosize=True, 
#                       height=chart_height, 
#                       margin=go.layout.Margin(autoexpand=True,l=10,r=10,b=2,t=100,pad=4),)
    
    fig.update_layout(title={'text':title_chart, 
                             'font': dict(size=20)}, 
                      template="plotly_white", 
                      autosize=True, 
                      height=chart_height, 
                      margin=go.layout.Margin(autoexpand=True,l=10,r=10,b=2,t=100,pad=4),)    

    pio.write_html(fig,file=file_name,auto_open=True)

