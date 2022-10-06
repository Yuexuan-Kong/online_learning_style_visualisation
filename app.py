import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import random


# ---------------------------------- Read data --------------------------------
features_norm = pd.read_csv("data/features_norm_test_viz.csv")
features_norm["f_sen"] = round((features_norm["f_sen_1"]+features_norm["f_sen_1"])/2, 2)
features_norm["f_per"] = round((features_norm["f_per_1"]+features_norm["f_per_1"])/2, 2)
features_norm["f_lea"] = round((features_norm["f_lea_1"]+features_norm["f_lea_1"])/2, 2)

username = random.randint(1, 1)
feature = features_norm.iloc[username]
dropout = pd.read_csv("data/dropout_rate.csv")
clicks = pd.read_csv("data/all_count_per_type_transformed.csv")

# for line chart over time
click_all = clicks.drop(columns=['username'])
click_all = click_all.groupby(['interval',"event_class", "event"]).size().reset_index()
click_all = click_all.rename(columns={0:"count"})
# for pie chart
click_single = clicks[clicks["username"] == username]

f_act = features_norm["f_act"]
f_ref = features_norm["f_ref"]
f_sen = round((features_norm["f_sen_1"]+features_norm["f_sen_1"])/2, 2)
f_int = features_norm["f_int"]
f_glo = features_norm["f_glo"]
f_seq = features_norm["f_seq"]
f_vis = features_norm["f_vis"]
f_ver = features_norm["f_ver"]
f_ded = features_norm["f_ded"]
f_ind = features_norm["f_ind"]
f_per = round((features_norm["f_per_1"]+features_norm["f_per_2"])/2, 2)
f_lea = round((features_norm["f_lea_1"]+features_norm["f_lea_2"])/2, 2)

dic_fea = {
        "dim1":[f_act, f_ref],
        "dim2":[f_sen, f_int],
        "dim3":[f_glo, f_seq],
        "dim4":[f_vis, f_ver],
        "dim5":[f_ded, f_ind],
        "dim6":[f_per, f_lea]
        }

fea_name = {
        "dim1":["f_act", "f_ref"],
        "dim2":["f_sen", "f_int"],
        "dim3":["f_glo", "f_seq"],
        "dim4":["f_vis", "f_ver"],
        "dim5":["f_ded", "f_ind"],
        "dim6":["f_per", "f_lea"]
        }

dic_pre = {
        'dim1':['Active活跃型','Reflective沉思型'],
        'dim2':['Sensitive感知型','Intuitive直觉型'],
        'dim3':['Global整体型','Sequential序列型'],
        'dim4':['Visual视觉型','Verbal言语型'],
        'dim5':['Deductive推理型','Inductive演绎型'],
        'dim6':['Performance表现型','Learning学习型']
        }

# ---------------------------------- Charts -----------------------------------
radio = dbc.RadioItems(
    id="choose-dim",
    className="radio",
    options=[
        {'label': 'Processing信息加工', 'value': 'dim1'},
        {'label': 'Perception感知维度', 'value': 'dim2'},
        {'label': 'Understanding理解维度', 'value': 'dim3'},
        {'label': 'Input输入维度', 'value': 'dim4'},
        {'label': 'Organisation组织维度', 'value': 'dim5'},
        {'label': 'Motivation动机维度', 'value': 'dim6'},
    ],
    value='dim1'
)

# slider map for number of days
slider_map = daq.Slider(
    id="slider_map",
    handleLabel={"showCurrentValue": True, "label": "Day"},
    min=1,
    max=36,
    value=36,
    size=450,
    color="#4B9072",
)

# slider map for number of days
slider_map_all = daq.Slider(
    id="slider_map_all",
    handleLabel={"showCurrentValue": True, "label": "Day"},
    min=1,
    max=36,
    value=36,
    size=450,
    color="#4B9072",
)
# ------------------------------- Radar features --------------------------------
radar_features = go.Figure()

categories = []
for i in dic_pre.keys():
    categories.extend(dic_pre[i])

# TODO position of the legend of the radar graph is too low
radar_features.add_trace(go.Scatterpolar(
      r=feature.to_list(),
      theta=categories,
      fill='toself',
      name='Your preferences你的偏好'
))
radar_features.add_trace(go.Scatterpolar(
      r=[0.5 for _ in range(0,13)],
      theta=categories,
      fill='toself',
      name='Average preferences平均偏好'
))

radar_features.update_layout(
        width=500,
        height=400,
        margin=dict(l=40, r=40, t=0, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color="black",
        font_size=10, 
        polar=dict(
          radialaxis=dict(
            visible=True,
            range=[0, 1]
          )),
)

# ---------------------------------- Layout -----------------------------------
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(children="YOUR ONLINE LEARNING STYLE"),
                html.Label(
                    "We are interested in investigating the food products that have the biggest impact on environment. Here you can understand which are the products whose productions emit more greenhouse gases and associate this with each supply chain step, their worldwide productions, and the water use.",
                    style={"color": "rgb(33 36 35)"},
                ),
                html.Img(
                    src=app.get_asset_url("supply_chain.png"),
                    style={
                        "position": "relative",
                        "width": "180%",
                        "left": "-83px",
                        "top": "-20px",
                    },
                ),
            ],
            className="side_bar",
        ),
        
        # first row
        html.Div(
            [
                # first widget: radar chart
                html.Div([
                    dcc.Graph(
                        id='radar-figure', 
                        figure=radar_features,
                        )
                    ],
                    id='radar-box', 
                    className='box', 
                    style={"width":"40%", 'vertical-align': 'middle', 'horizontal-align': 'middle'}
                ),
                
                # second widget: description of cluster
                html.Div([
                        html.P(id='cluster_text', children='You belong to cluster A')
                        ], id='cluster_div', className='box_comment', style={"width":"70%"})
                ],
                id="row1",
                className="row"
            ),

        # second row
        html.Div(
            [
                # second widget: choose dimension 
                html.Div(
                    [
                        html.Label("Choose dimension:"), 
                        html.Br(),
                        html.Label("选择观察的维度："),
                        html.Br(),
                        radio,
                    ],
                    id='chooseDim',
                    className="box",
                    style={
                        "margin": "10px",
                        "padding-top": "15px",
                        "padding-bottom": "15px",
                    },
                ),
                
                # second big widget, three parts: two preferences and desciption
                html.Div([

                    # first row for two preferences
                    html.Div([
                        # header
                        html.Label(
                        "You among all people 你在人群中的位置",
                        style={"font-size": "medium"},
                                ),
                        html.Br(),
                        html.Br(),

                        # preference 1 
                        html.Div([
                            dcc.Graph(
                                id="graph-pref1"),
                            html.Div(
                                [html.P(id='comment-pref1')],
                                className='box_comment',
                                )
                                ], id='preference1'),
                        # preference 2
                        html.Div([
                            dcc.Graph(
                                id="graph-pref2"),
                            html.Div(
                                [html.P(id='comment-pref2')],
                                className='box_comment',
                                )
                                
                            ], id='preference2')
                        ],
                        id='preferenceChart',
                        className='row',
                        style={"height":"80%"}
                    ),
                    
                    # second row for desciption of this dimension
                    ],
                    id ='row2col2',
                    className='box',
                    style={"width":"80%"}
                )
                ],
                id='row2',
                className="row",
                style={"height": "200%", "width":"100%"}
            ),

        # third row
        html.Div(
            [
                # first widget: sum of clicks on different types over time
                html.Div([
                    html.Label(
                        "Number of clicks on each type of material",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "各种类材料点击数量分布",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(
                        id='pie-chart-single', 
                        )
                    ],
                    id='overtime',
                    className='box',
                    style={"width":"40%"}
                ),
                
                # second widget: pie chart for diffent kinds of clicks
                html.Div([
                    html.Label(
                        "Number of clicks over time",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "各种类材料点击数量随时间变化",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(
                        id='line-chart-single', 
                        ),
                    html.Br(),
                    slider_map
                    ],
                    id='pieChart',
                    className='box',
                    style={"width":"60%"}
                )
                ],
                id="row3",
                className="row",
            ),

        # fourth row
        html.Div(
            [
                # first widget: sum of clicks on different types over time
                html.Div([
                    html.Label(
                        "Number of clicks over time",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "各种类材料点击数量随时间变化",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(
                        id='line-chart-all', 
                        ),
                    html.Br(),
                    slider_map_all
                    ],
                    id='pieChart-all',
                    className='box',
                    style={"width":"60%"}
                ),
                
                # second widget: pie chart for diffent kinds of clicks
                html.Div([
                    html.Label(
                        "Number of clicks on each type of material",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "各种类材料点击数量分布",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(
                        id='pie-chart-all', 
                        )
                    ],
                    id='overtime-all',
                    className='box',
                    style={"width":"40%"}
                )],
                id="row4",
                className="row",
            ),

        # fifth row
        html.Div(
            [
                # first widget: bar chart of dropout rate
                html.Div([
                    
                    ],
                    id='dropout',
                    className="box",
                    style={"width": "50%"}
                ),
                
                # second widget: bar chart of number of courses choses
                html.Div([

                    ],
                    id='numberClass',
                    className="box",
                    style={"width": "50%"}
                )
                ],
                id='row5',
                className="row"
            )

    ],
    className='main'
)


# ----------------------------------- Callbacks -------------------------------
@app.callback(
    Output("graph-pref1", "figure"),
    Output("graph-pref2", "figure"),
    Input("choose-dim", "value")
)
def display_histogram_prefs(chooseDim):

    global feature
    global dic_fea
    global fea_name
    global dic_pre

    pref1, pref2 = dic_fea[chooseDim]
    title1, title2 = dic_pre[chooseDim]
    str1, str2 = fea_name[chooseDim]

    fig_pref1 = px.histogram(pref1, log_y=True,histnorm='probability density', range_x=[0, 1], color_discrete_sequence=['goldenrod'], opacity=0.6)
    fig_pref2 = px.histogram(pref2, log_y=True, histnorm='probability density', range_x=[0, 1], color_discrete_sequence=['goldenrod'], opacity=0.6)

    fig_pref1.update_layout(
            showlegend=False,
            title= title1,
            yaxis={'title':'Number of people人数','visible': True, 'showticklabels': True, 
                    'tickmode' : 'array',
        'tickvals' : [0.1,2],
        'ticktext' : ['few','numerous']},
            xaxis={'title':'Preference偏好','visible': True, 'showticklabels': True,
                    'tickmode' : 'array',
        'tickvals' : [0.1,0.9],
        'ticktext' : ['weak','strong']},
        width=400,
        height=250,
        margin=dict(l=10, r=60, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="black",
        font_size=10, 
          )
    
    fig_pref1.add_vline(
            x=feature[str1], 
            line_width=3,
            line_dash = 'dash', 
            line_color = 'firebrick'
            )

    fig_pref1.add_annotation(x=feature[str1], y=2,
            text="Your position你的位置",
            showarrow=True,
            arrowhead=1)

    fig_pref2.update_layout(
            showlegend=False,
            title=title2,
            yaxis={'title':'Number of people人数','visible': True, 'showticklabels': True, 
                    'tickmode' : 'array',
        'tickvals' : [0.1,2],
        'ticktext' : ['few','numerous']},
            xaxis={'title':'Preference偏好','visible': True, 'showticklabels': True,
                    'tickmode' : 'array',
        'tickvals' : [0.1,0.9],
        'ticktext' : ['weak','strong']},
        width=400,
        height=250,
        margin=dict(l=10, r=60, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="black",
        font_size=10, 
          )

    fig_pref2.add_vline(
            x=feature[str2], 
            line_width=3,
            line_dash = 'dash', 
            line_color = 'firebrick'
            )

    fig_pref2.add_annotation(x=feature[str2], y=2,
            text="Your position你的位置",
            showarrow=True,
            arrowhead=1)

    return fig_pref1, fig_pref2

# callback for xxxxx over time
@app.callback(
    Output("pie-chart-single", "figure"),
    Output("line-chart-single", "figure"),
    Input("slider_map", "value")
)
def display_charts_single(day):


    global click_single   
    if not click_single[click_single.interval<=day].empty:
        click_single_sun = click_single[click_single.interval<=day]
    else:
        click_single_sun = pd.DataFrame({'event_class':['no activity'], 'interval':[0],'event':['no activity'], '0':[0]})


    pie_chart_click = px.sunburst(
                click_single_sun,
                path=["event_class", "event"],
                values="0",
                color="event_class",
                color_discrete_sequence=px.colors.sequential.haline_r,
            ).update_traces(hovertemplate="%{label}<br>" + "Number of clicks: %{value} times")


    pie_chart_click = pie_chart_click.update_layout(
        {
            "margin": dict(t=0, l=0, r=0, b=10),
            "paper_bgcolor": "#F9F9F8",
            "font_color": "#363535",
        }
    )


    click_single_line = click_single_sun.drop(columns=['event'])
    click_single_line = click_single_line.groupby(['interval', 'event_class']).sum().reset_index()
    line_chart_click = px.bar(
            click_single_line, 
            x="interval", 
            y="0", 
            color="event_class", 
            text_auto=True,
            labels={'interval':'Day 天数', 'event_class':'event class事件类型', '0':'clicks点击数量'}
            ).update_traces(hovertemplate="%{label}<br>" + "Number of clicks点击数量: %{y}")

    line_chart_click.update_xaxes(range=[1, day])
    line_chart_click = line_chart_click.update_layout(
            hovermode='x',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',

    )

    return pie_chart_click, line_chart_click


# callback for all over time
@app.callback(
    Output("pie-chart-all", "figure"),
    Output("line-chart-all", "figure"),
    Input("slider_map_all", "value")
)
def display_charts_all(day):

    global click_all   

    click_all_sun = click_all[click_all.interval<=day]

    pie_chart_click = px.sunburst(
                click_all_sun,
                path=["event_class", "event"],
                values='count',
                color="event_class",
                color_discrete_sequence=px.colors.sequential.Peach_r,
            ).update_traces(hovertemplate="%{label}<br>" + "Number of clicks: %{value} times")


    pie_chart_click = pie_chart_click.update_layout(
        {
            "margin": dict(t=0, l=0, r=0, b=10),
            "paper_bgcolor": "#F9F9F8",
            "font_color": "#363535",
        }
    )


    click_all_line = click_all_sun.drop(columns=['event'])
    click_all_line = click_all_line.groupby(['interval', 'event_class']).sum().groupby(['event_class']).cumsum().reset_index()
    line_chart_click = px.bar(
            click_all_line, 
            x="interval", 
            y='count', 
            color="event_class", 
            text_auto=True,
            labels={'interval':'Day 天数', 'event_class':'event class事件类型', 'count':'clicks点击数量'}
            ).update_traces(hovertemplate="%{label}<br>" + "Number of clicks点击数量: %{y}")

    line_chart_click.update_xaxes(range=[1, day])
    line_chart_click = line_chart_click.update_layout(
            hovermode='x',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return pie_chart_click, line_chart_click

if __name__ == "__main__":
    app.run_server(debug=True)
