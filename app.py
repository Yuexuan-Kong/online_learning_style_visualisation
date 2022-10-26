import os

import dash
import math
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import random


# ---------------------------------- Read data --------------------------------
features_norm = pd.read_csv("data/features_norm_row.csv")
features_norm = features_norm.rename(columns={"f_glo_1":"f_glo", "f_seq_1":"f_seq"})
features_norm["f_act"] = round((features_norm["f_act_1"]+features_norm["f_act_2"])/2, 2)
features_norm["f_sen"] = round((features_norm["f_sen_1"]+features_norm["f_sen_2"])/2, 2)
features_norm["f_vis"] = round((features_norm["f_vis_1"]+features_norm["f_vis_2"])/2, 2)
features_norm["f_ver"] = round((features_norm["f_ver_1"]+features_norm["f_ver_2"])/2, 2)
features_norm["f_per"] = round((features_norm["f_per_1"]+features_norm["f_per_2"])/2, 2)
features_norm["f_lea"] = round((features_norm["f_lea_1"]+features_norm["f_lea_2"])/2, 2)

username = random.randint(1, 1)+3000 # type 6
feature = features_norm.iloc[username]
feature = feature[["f_act", "f_ref", "f_sen","f_int","f_glo","f_seq","f_vis","f_ver","f_ded","f_ind","f_per","f_lea"]]
dropout = pd.read_csv("data/dropout_rate.csv")
dropout_mean = dropout["truth"].mean()
dropout_user = dropout["truth"].iloc[username]
count = pd.read_csv("data/summary_clicks.csv")
count_mean = count["count_course"].mean()
count_user = count["count_course"].iloc[username]
clicks = pd.read_csv("data/all_count_per_type_transformed.csv")

# for line chart over time
click_all = clicks.drop(columns=['username'])
click_all = click_all.groupby(['interval',"event_class", "event"]).size().reset_index()
click_all = click_all.rename(columns={0:"count"})
# for pie chart
click_single = clicks[clicks["username"] == username]

f_act = features_norm["f_act"]
f_ref = features_norm["f_ref"]
f_sen = features_norm["f_sen"]
f_int = features_norm["f_int"]
f_glo = features_norm["f_glo"]
f_seq = features_norm["f_seq"]
f_vis = features_norm["f_vis"]
f_ver = features_norm["f_ver"]
f_ded = features_norm["f_ded"]
f_ind = features_norm["f_ind"]
f_per = features_norm["f_per"]
f_lea = features_norm["f_lea"]

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
        'dim5':['Deductive演绎型','Inductive归纳型'],
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

clusterRadio = dbc.RadioItems(
    id="choose-cluster",
    className="radio",
    options=[
        {'label': 'What', 'value': 'dim1'},
        {'label': 'Should', 'value': 'dim2'},
        {'label': 'Go', 'value': 'dim3'},
        {'label': 'In', 'value': 'dim4'},
        {'label': 'These', 'value': 'dim5'},
        {'label': 'Thingies', 'value': 'dim6'},
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


radar_features.add_trace(go.Scatterpolar(
      r=feature.to_list(),
      theta=categories,
      fill='toself',
      name='Your preferences你的偏好'
))
radar_features.add_trace(go.Scatterpolar(
      r=[0.23, 0.47, 0.52, 0.39, 0.16, 0.29, 0.51, 0.59, 0.41, 0.59, 0.43, 0.44],
      theta=categories,
      fill='toself',
      name='Average preferences平均偏好'
))

# DONE: I fine tuned the "x" value in the "legend" direct

# TODO: the position of the radar graph is not centered :)
radar_features.update_layout(
        width=500,
        height=400,
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=32),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=0.94),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color="black",
        font_size=10, 
        polar=dict(
          radialaxis=dict(
            visible=True,
            range=[0, 1]
          )),
)

# Horizontal bar chart for number of people in different clusters
bar_num = go.Figure()
bar_num.add_trace(go.Bar(
    y=['I型', 'II型', 'III型', 'IV型', 'V型', 'VI型', 'VII型'],
    x=[4326, 2977, 2905, 4570, 3865, 2539, 5741],
    name='number of people',
    orientation='h',
    marker=dict(
        color='rgba(246, 78, 139, 0.6)',
        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
    ),
))

bar_num.update_layout(
    autosize=True,
    height=400,
    width=300,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
    paper_bgcolor="rgba(0, 0, 0, 0)",
)
bar_num.update_traces(
        texttemplate = [4326, 2977, 2905, 4570, 3865, 2539, 5741],textposition = "inside"
        )

# Bar charts for dropout rate and count of chosen courses

animals=['Dropout rate辍课数', 'Courses chosen总选课数']
bar_drop = go.Figure()

bar_drop.add_trace(go.Bar(
    name='You你的数据', x=animals, y=[math.ceil(dropout_user*count_user), count_user],
    marker_color='indianred'
))
bar_drop.add_trace(go.Bar(
    name='Average平均数据', x=animals, y=[dropout_mean*count_mean, count_mean], 
    marker_color='lightsalmon'
))
# Change the bar mode
bar_drop.update_layout(autosize=True, margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
    xaxis_tickangle=0, barmode='group', paper_bgcolor="rgba(0, 0, 0, 0)"
)

# ---------------------------------- Layout -----------------------------------
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(children="你的在线学习风格展示"),
                html.H1(children="YOUR ONLINE LEARNING STYLE"),
                html.Label(
                    "你想不想知道，你是擅长像福尔摩斯一样归纳推理、还是擅长像高斯一样推理数学定理呢？你喜欢跳跃性的学习知识，还是喜欢循序渐进的一步步来呢？",
                    style={"color": "rgb(33 36 35)"},
                ),
                html.Label(
                    "在这里，你可以找到你想要的答案。在这里，你可以看到自己的在线学习风格偏好；在这里，你可以看到自己与他人的比较；在这里，你可以看到自己在线学习时更喜欢点击什么材料。",
                    style={"color": "rgb(33 36 35)"},
                ),
                # TODO: can you find an image here for learning style and/or online learning? 
                html.Img(
                    src=app.get_asset_url("supply_chain.png"),
                    style={
                        "position": "relative",
                        "width": "115%",
                        "left": "-20px",
                        "top": "40px",
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
                    html.Label(
                        "你的各偏好在人群中的比较",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "Your preferences compared to average",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(
                        id='radar-figure', 
                        figure=radar_features,
                        )
                    ],
                    id='radar-box', 
                    className='box', 
                    style={"width":"40%", 'vertical-align': 'middle', 'horizontal-align': 'middle'}
                ),


                # first widget: bar chart of dropout rate
                html.Div([
                    html.Label(
                        "你的各偏好在人群中的比较",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "Your preferences compared to average",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(
                        id='dropout_figure', 
                        figure=bar_drop,
                        )
                    ],
                    id='dropout-box', 
                    className='box', 
                    style={"width":"50%", 'vertical-align': 'middle', 'horizontal-align': 'middle'}
                ),
                
                ],
                id="row1",
                className="row"
            ),

        # fifth row
        html.Div(
            [
                
                # TODO: what i wanted is a bar chart hehe, not buttons, a horizontal bar chart that show how many students there are in each cluster :)
                # second widget: description of cluster
                html.Div(
                    [
                        html.Label(
                            "各类型学习者的人数",
                            style={"font-size": "medium"},
                        ),
                        html.Br(),
                        html.Label(
                            "Number of students in each cluster",
                            style={"font-size": "medium"},
                        ),
                        html.Br(),
                        html.Br(),
                        dcc.Graph(
                            id='bar-num',
                            figure=bar_num,
                            )
                    ],
                    id='bar-num-box',
                    className="box",
                    style={"width":"30%", 'vertical-align': 'middle', 'horizontal-align': 'middle'}
                ),
                # third widget: description of cluster
                # TODO: How about the color now of this box now?
                html.Div([
                        html.P(id='cluster_text', 
                            # children="Hi babe. You are probably sleepy and reading this wondering how come there is so much text here. Fear not. It is I! Your boyfriend (dramatic cat pose (m)O_O(m) ). Before you start coding, I wanted to say that I admire your capability to say: you know what? Getting an actual dash board here with interactive plots and loads of visual user info sounds like a good idea. And then you go ahead and do it. What if you don´t have much experience on it, what if you do not know about the available tools and don´t have much time? You go ahead and do it anyway. Focusing single mindedly on it. I admire it. It is for things like this that I know that things will go well for you in the end. It is for things like this that I know you will be an awesome PhD student. You are getting close, babe. Te amo <3."
                            children=""
                            ),
                        dcc.Markdown('''
                            # 你属于VI型在线学习者

                            ## 类别六的在线学习者对文字材料有着非常强烈的偏好，也是极强的表现力驱动的学习者，也同时是沉思型和序列型的学习者。你们更喜欢独自学习，而不是与同学进行热烈的讨论。你们喜欢对文字型材料进行学习，或者音频类型的语音类课件。你们喜欢按部就班地一章一章的推进学习工作，并且对获得好成绩、超越他人展现出一定的高欲望。
                            ## VI型学习者是最罕见的学习者，你们在人群中是大熊猫般的存在。
                            ## 在线学习风格模型一共具有六个不同的维度，想知道每个维度代表了什么吗？想知道你在每个维度上的偏好吗？那就继续往下阅读吧！你会找到答案的哦。
                            ''')# Previous text 'You belong to cluster A'

                        ], id='cluster_div', className='box_comment', style={"width":"30%"}),
                # second widget: bar chart of number of courses choses
                html.Div([
                    html.Img(
                        src=app.get_asset_url("Food.png"),
                        style={
                            "position": "relative",
                            "width": "95%",
                            "left": "20px",
                            "top": "40px",
                        },
                        ),
                    ]),
                ],
                id='row5',
                className="row"
            ),
        # second row
        html.Div(
            [
                # second widget: choose dimension 
                # DONE: I added an extra html.Br()
                html.Div(
                    [
                        html.Label("选择观察的维度"),
                        html.Br(),
                        html.Label("Choose dimension"), 
                        html.Br(),
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
                        # DONE: Like this? (Dani moves hands as if presenting a cool product)
                        # exactly like this
                        # header
                        html.Label(
                        "你在人群中的位置 You among all people ",
                           style={"font-size": "medium"},
                        ),
                      ],
                      style={"text-align": "center"},
                    ),
                    html.Div([
                        # preference 1 
                        html.Div([
                            dcc.Graph(
                                id="graph-pref1"),
                            html.Div(
                                [dcc.Markdown(id='comment-pref1')],
                                className='box_comment',
                                )
                                ], id='preference1'),
                        # preference 2
                        html.Div([
                            dcc.Graph(
                                id="graph-pref2"),
                            html.Div(
                                [dcc.Markdown(id='comment-pref2')],
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
                    style={"width":"80%", "text-align": "center"}
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
                        "各种类材料点击数量分布",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "Number of clicks on each type of material",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
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
                        "各种类材料点击数量随时间变化",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "Number of clicks over time",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
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
                        "各种类材料点击数量随时间变化",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "Number of clicks over time",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
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
                        "各种类材料点击数量分布",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Label(
                        "Number of clicks on each type of material",
                        style={"font-size": "medium"},
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label(
                        "点击获取更多消息",
                        style={"font-size": "9px"},
                    ),
                    html.Br(),
                    html.Label(
                        "Click on it to know more!",
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
        'tickvals' : [0.01,2.5],
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
            arrowhead=1,
            ax=60,
            ay=10)

    fig_pref2.update_layout(
            showlegend=False,
            title=title2,
            yaxis={'title':'Number of people人数','visible': True, 'showticklabels': True, 
                    'tickmode' : 'array',
        'tickvals' : [0.01,2.5],
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
            arrowhead=1,
            ax=60,
            ay=10)

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
            ).update_traces(
              hovertemplate="%{label}<br>" + "Number of clicks: %{value} times",
              textinfo="label+percent entry"
            )


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
    
    # DONE: We had to add a "textinfo" param in update_traces
    # Look at this: https://stackoverflow.com/questions/71902624/plotly-sunburst-chart-with-percentages

    pie_chart_click = px.sunburst(
                click_all_sun,
                path=["event_class", "event"],
                values='count',
                color="event_class",
                color_discrete_sequence=px.colors.sequential.Peach_r,
            ).update_traces(
                hovertemplate="%{label}<br>" + "Number of clicks: %{value} times",
                textinfo="label+percent parent"
            )


    pie_chart_click = pie_chart_click.update_layout(
        {
            "margin": dict(t=0, l=0, r=0, b=10),
            "paper_bgcolor": "#F9F9F8",
            "font_color": "#363535",
        }
    )


    click_all_line = click_all_sun.drop(columns=['event'])
    click_all_line = click_all_line.groupby(['interval', 'event_class']).sum().groupby(['event_class']).cumsum().reset_index()
    # TODO: is it possible to change number of clicks to percentage of clicks on this material against all materials? because now the number is tooooo big
    # NOT DONE: It already shows percentages, right?
    # Muxho: it shows the number of clicks instead of percentages, but i think it's not too grave
    # this is for the cool bar chart on the fourth row
    line_chart_click = px.bar(
            click_all_line, 
            x="interval", 
            y='count', 
            color="event_class", 
            text_auto=True,
            labels={'interval':'Day 天数', 'event_class':'event class事件类型', 'count':'clicks点击数量'}
            ).update_traces(
              hovertemplate="%{label}<br>" + "Number of clicks点击数量: %{y}",
            )

    line_chart_click.update_xaxes(range=[1, day])
    line_chart_click = line_chart_click.update_layout(
            hovermode='x',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return pie_chart_click, line_chart_click

# Callbakcs for description of different preferences
@app.callback(
    Output("comment-pref1", "children"),
    Output("comment-pref2", "children"),
    Input("choose-dim", "value")
)
def description(chooseDim):
    if chooseDim == "dim1":
        text1 = "信息加工维度是测量你喜欢如何在大脑中处理信息的维度。你在此维度上具有沉思型偏好"
        text2 = "你喜欢一个人安静的学习，而不是在在线论坛上与大家进行讨论。你喜欢首先安静地独自思考问题，“我们先好好想想吧”是你的口头禅。"
    if chooseDim == "dim2":
        text1 = "感知维度是测量你喜欢用什么样的方式去接受新知识的维度。你在此维度上是平衡型的。"
        text2 = "你有时喜欢学习事实，有时又喜欢学习抽象的数学公式。你有时很注意细节，有时又很有创新性。"
    if chooseDim == "dim3":
        text1 = "理解维度是测量你喜欢以怎么样的顺序去学习知识的维度。你在此维度上具有序列型的偏好。"
        text2 = "你习惯按线性步骤理解问题，每一步都合乎逻辑地紧跟前一步。你倾向于按部就班地寻找答案。"
    if chooseDim == "dim4":
        text1 = "输入维度是测量你喜欢用什么途径去获得新知识的维度。你在此维度上具有言语型偏好。"
        text2 = "你擅长从文字和口头的解释中获取信息，通过阅读文章或听广播是你获取知识的主要途径。"
    if chooseDim == "dim5":
        text1 = "理解维度是测量你喜欢用什么样的方式从获得的信息转换为内部更深层次的理解的维度，你在此维度上具有演绎型的偏好。"
        text2 = "你推理的时候，喜欢从广泛的大事实入手，得到结果。“如果A等于B，同时C又是A，那么C就是B。如果A不等于B，那么C不可能等于B” 是你喜欢的推理方式。"
    if chooseDim == "dim6":
        text1 = "动机维度是测量你的主要动机来源的维度，你在此维度上具有表现力偏好。"
        text2 = "你努力学习主要是为了出人头地，想比别人更有成就，你喜欢在学习中竞争， 也喜欢在竞争中学习。"
    
    return text1, text2 

if __name__ == "__main__":
    app.run_server(debug=True)
    # TODO: remember that muxho is very very grateful to her huobao <3 I love you.
