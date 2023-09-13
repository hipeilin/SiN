import dash
from dash import dcc
from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
# from dash import dcc
from plotly.subplots import make_subplots

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fetch_csv(selection):
    date_columns = ['start', 'end']
    df_temp = pd.read_csv(selection, parse_dates=date_columns)

    return df_temp


# participant_number = 1
# participant = 'P' + str(participant_number)
# df = fetch_csv(participant + '.csv')
# df_vc = fetch_csv(participant + '_vc.csv')
# df_id_attr = fetch_csv(participant + '_id_attr.csv')
# df_id_attr['ID'] = df_id_attr['ID'].astype(str)
# df_id_attr = df_id_attr.sort_values(by='start')
# df_bubble_bg = fetch_csv(participant + '_bubble_bg.csv')

df_list, df_vc_list, df_id_attr_list, df_bubble_bg_list = [], [], [], []
for i in range(1, 8):
    df_ = fetch_csv('P' + str(i) + '.csv')
    df_list.append(df_)
    df_vc_ = fetch_csv('P' + str(i) + '_vc.csv')
    df_vc_list.append(df_vc_)
    df_id_attr_ = fetch_csv('P' + str(i) + '_id_attr.csv')
    df_id_attr_list.append(df_id_attr_)
    df_bubble_bg_ = fetch_csv('P' + str(i) + '_bubble_bg.csv')
    df_bubble_bg_list.append(df_bubble_bg_)

df = df_list[0]
df_vc = df_vc_list[0]
df_id_attr = df_id_attr_list[0]
df_id_attr['ID'] = df_id_attr['ID'].astype(str)
df_id_attr = df_id_attr.sort_values(by='start')
df_bubble_bg = df_bubble_bg_list[0]

# --------------------------------------------------------------------------
# front end
# --------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    children=[
        # title
        html.H1("SiN (Sweden in Number) Project"),

        html.Div(
            children=[
                html.H3('Individual View'),

                # First column
                html.Div([
                    # Select participant
                    html.P("Select participant"),
                    dcc.Dropdown(
                        id='participant_selection',
                        multi=False,
                        clearable=False,
                        options=[
                            {'label': 'P' + str(i), 'value': i} for i in range(1, 8)
                        ],
                        value=1,
                        style={'width': '200px'}
                    ),
                ], style={'display': 'inline-block', 'verticalAlign': 'top'}),

                # Second column
                html.Div([
                    # Choose what plots to show
                    dcc.Checklist(
                        id='view_switch',
                        options=[{'label': 'Show views', 'value': True}],
                        value=[True],
                    ),
                    dcc.Checklist(
                        id='bubble_switch',
                        options=[{'label': 'Show bubbles', 'value': True}],
                        value=[True],
                    ),
                    dcc.Checklist(
                        id='year_switch',
                        options=[{'label': 'Show "year" attributes', 'value': True}],
                        value=[],
                    ),
                    dcc.Checklist(
                        id='municipality_switch',
                        options=[{'label': 'Show "municipality" attributes', 'value': True}],
                        value=[],
                    ),
                    dcc.Checklist(
                        id='gender_switch',
                        options=[{'label': 'Show "gender" attributes', 'value': True}],
                        value=[],
                    ),
                    dcc.Checklist(
                        id='complete_bubble_switch',
                        options=[{'label': 'Show incomplete bubbles', 'value': True}],
                        value=[True],
                    ),
                    dcc.Checklist(
                        id='color_switch',
                        options=[{'label': 'Toggle color by view (default by ID)', 'value': True, 'disabled': True}],
                        style={'color': 'lightGray'},
                        value=[],
                    ),
                ], style={'display': 'inline-block', 'width': '300px', 'border': '1px solid black',
                          'marginLeft': '20px', 'verticalAlign': 'top'}),

                # Third column
                html.Div([
                    # Task selection
                    html.P('Task selection', style={'paddingLeft': '4px'}),
                    dcc.RadioItems(
                        id='task_selection',
                        options=[{'label': 'All', 'value': 'All'},
                                 {'label': 'Task 1', 'value': 'Task 1'},
                                 {'label': 'Task 2', 'value': 'Task 2'},
                                 {'label': 'Task 3', 'value': 'Task 3'},
                                 {'label': 'Task 4', 'value': 'Task 4'},
                                 {'label': 'Task 5', 'value': 'Task 5'}],
                        value='All',
                    ),
                ], style={'display': 'inline-block', 'width': '300px',
                          'marginLeft': '20px', 'border': '1px solid black', 'verticalAlign': 'top'}),

                # Fourth column
                html.Div([
                    # Stats about the incomplete bubbles
                    html.Div([
                        html.Span('Number of bubbles in total: '),
                        html.Span(id='total_bubble_number'), html.Br(),
                        html.Span('Number of complete bubbles: '),
                        html.Span(id='complete_bubble_number'), html.Br(),
                        html.Span('Number of incomplete bubbles: '),
                        html.Span(id='incomplete_bubble_number'), html.Br(),
                        html.Span('Number of interaction: '),
                        html.Span(id='interaction_number'), html.Br(),
                        html.Span('Total duration: '),
                        html.Span(id='total_duration'), html.Br(),
                        html.Span('Total duration per task: '),
                        html.Span(id='task_duration'), html.Br(),
                    ],
                        style={'paddingLeft': '5px', 'border': '1px solid black'}
                    ),
                ], style={'display': 'inline-block', 'width': '300px',
                          'marginLeft': '20px', 'verticalAlign': 'top'}),

                html.Div([
                    # Task selection
                    html.P('Plot customization', style={'paddingLeft': '4px'}),
                    # dcc.RadioItems(
                    #     id='line_scatter_switch',
                    #     options=[{'label': 'Scatter', 'value': 'scatter'},
                    #              {'label': 'Line', 'value': 'line'}],
                    #     value='scatter',
                    #     style={'marginBottom': '3px'}
                    # ),

                    dcc.RadioItems(
                        id='line_scatter_switch_individual',
                        options=[{'label': 'Scatter', 'value': 'scatter'},
                                 {'label': 'Line', 'value': 'line'}],
                        value='scatter',
                        style={'marginBottom': '3px'}
                    ),
                    dcc.Checklist(
                        id='line_interpolation_individual',
                        options=[{'label': '', 'value': True}],
                        value=[],
                    ),
                    dcc.Checklist(
                        id='toggle_timestamp',
                        options=[{'label': 'Toggle task timestamp', 'value': True}],
                        value=[True],
                    ),
                ], style={'display': 'inline-block', 'width': '300px',
                          'marginLeft': '20px', 'border': '1px solid black', 'verticalAlign': 'top'}),

                # Fifth column
                # html.Div([
                #     html.P('Cosine similarity between'),
                #     dcc.Dropdown(
                #         id='cos_1',
                #         multi=False,
                #         clearable=False,
                #         options=[
                #             {'label': 'P' + str(i), 'value': i} for i in range(1, 8)
                #         ],
                #         value=1,
                #         style={'display': 'inline-block', 'width': '65px',
                #                'marginRight': '10px', 'verticalAlign': 'top'}
                #     ),
                #     html.Span('and', style={'display': 'inline-block', 'verticalAlign': 'top'}),
                #     dcc.Dropdown(
                #         id='cos_2',
                #         multi=False,
                #         clearable=False,
                #         options=[
                #             {'label': 'P' + str(i), 'value': i} for i in range(1, 8)
                #         ],
                #         value=1,
                #         style={'display': 'inline-block', 'width': '65px',
                #                'marginLeft': '10px'}
                #     ),
                #     html.P("on"),
                #
                #     dcc.RadioItems(
                #         id='attr_selection',
                #         options=[{'label': 'Year', 'value': 'year'},
                #                  {'label': 'Municipality', 'value': 'municipality'},
                #                  {'label': 'Gender', 'value': 'gender'}],
                #         value='year',
                #     ),
                #
                #     html.P('1', id='cos_result', style={'fontWeight': 'bold'}),
                #
                #     html.Button("Compare", id='compare_exe', style={'display': 'block'}),
                #
                # ], style={'display': 'inline-block', 'width': '300px',
                #           'marginLeft': '20px', 'verticalAlign': 'top'}),

                # Main plot
                dcc.Graph(id='timeline_plot'),

                # ================================================================================================
                html.Hr(),
                html.H3('Comparison View'),

                # Control panel for comparison plot
                html.Div([
                    # Task selection
                    html.P('Task selection', style={'paddingLeft': '4px'}),
                    dcc.RadioItems(
                        id='comp_attribute_selection',
                        options=[
                            {'label': 'Gender', 'value': 'gender'},
                            {'label': 'Year', 'value': 'year'},
                            {'label': 'Municipality', 'value': 'municipality'}
                        ],
                        value='gender',
                    ),
                ], style={'display': 'inline-block', 'width': '300px',
                          'border': '1px solid black', 'verticalAlign': 'top'}),

                html.Div([
                    # Task selection
                    html.P('Attribute selection', style={'paddingLeft': '4px'}),
                    dcc.RadioItems(
                        id='comp_task_selection',
                        options=[{'label': 'All', 'value': 'All'},
                                 {'label': 'Task 1', 'value': 'Task 1'},
                                 {'label': 'Task 2', 'value': 'Task 2'},
                                 {'label': 'Task 3', 'value': 'Task 3'},
                                 {'label': 'Task 4', 'value': 'Task 4'},
                                 {'label': 'Task 5', 'value': 'Task 5'}],
                        value='All',
                        style={'marginBottom': '3px'}
                    ),
                    dcc.Checklist(
                        id='task_aligned',
                        options=[{'label': '', 'value': True}],
                        value=[],
                    ),
                ], style={'display': 'inline-block', 'width': '300px',
                          'marginLeft': '20px', 'border': '1px solid black', 'verticalAlign': 'top'}),

                html.Div([
                    # Task selection
                    html.P('Plot customization', style={'paddingLeft': '4px'}),
                    dcc.RadioItems(
                        id='line_scatter_switch_comp',
                        options=[{'label': 'Scatter', 'value': 'scatter'},
                                 {'label': 'Line', 'value': 'line'}],
                        value='scatter',
                        style={'marginBottom': '3px'}
                    ),
                    dcc.Checklist(
                        id='line_interpolation',
                        options=[{'label': '', 'value': True}],
                        value=[],
                    ),
                ], style={'display': 'inline-block', 'width': '300px',
                          'marginLeft': '20px', 'border': '1px solid black', 'verticalAlign': 'top'}),

                # Concept plot
                dcc.Graph(id='test_plot'),

                # dummy element
                html.P(id='dummy', style={'display': 'none'}),
            ]
        )
    ]
)


# Switch for the 'Line Interpolation' option in comparison view
@app.callback(
    Output('line_interpolation', 'options'),
    Output('line_interpolation', 'style'),
    Input('line_scatter_switch_comp', 'value'),
)
def color_button_toggle(line_scatter_switch_comp):
    if line_scatter_switch_comp == 'scatter':
        new_options = [{'label': 'Attribute interpolation', 'value': False, 'disabled': True}]
        new_style = {'color': 'lightGray'}
        return new_options, new_style
    else:
        new_options = [{'label': 'Attribute interpolation', 'value': True, 'disabled': False}]
        new_style = {}
        return new_options, new_style


# Switch for the 'Individual or Whole Task' option
@app.callback(
    Output('task_aligned', 'options'),
    Output('task_aligned', 'style'),
    Input('comp_task_selection', 'value'),
)
def color_button_toggle(comp_task_selection):
    if comp_task_selection == 'All':
        new_options = [{'label': 'Align by task', 'value': False, 'disabled': True}]
        new_style = {'color': 'lightGray'}
        return new_options, new_style
    else:
        new_options = [{'label': 'Align by task', 'value': True, 'disabled': False}]
        new_style = {}
        return new_options, new_style


# Comparison plot
@app.callback(
    Output('test_plot', 'figure'),
    Input('comp_attribute_selection', 'value'),
    Input('comp_task_selection', 'value'),
    Input('task_aligned', 'value'),
    Input('line_scatter_switch_comp', 'value'),
    Input('line_interpolation', 'value'),
)
def comparison_plot_control(comp_attribute_selection, comp_task_selection, task_aligned, line_scatter_switch_comp,
                            line_interpolation):
    num_dfs = len(df_bubble_bg_list)
    fig = make_subplots(shared_xaxes=True, specs=[[{"secondary_y": True}]] * num_dfs, rows=num_dfs, cols=1,
                        vertical_spacing=0.01)

    for i_ in range(0, 7):

        bubble_df = df_bubble_bg_list[i_]
        attr_df = df_id_attr_list[i_]
        bubble_df['ID'] = bubble_df['ID'].astype(str)

        # Preprocess the DF if task selection is active
        if comp_task_selection != 'All':
            current_task = int(comp_task_selection[-1])
            next_task = 'Task ' + str(current_task + 1)

            # Filter between the tasks for bubble BG
            bubble_df_current_task_index = bubble_df[bubble_df['event'] == comp_task_selection].index[0]
            bubble_df_next_task_index = bubble_df[bubble_df['event'] == next_task].index[0]
            id_attr_df_filtered = bubble_df.loc[bubble_df_current_task_index + 1:bubble_df_next_task_index - 1]

            # Filter between the tasks for attribute plot
            attr_df_current_task_index = attr_df[attr_df['event'] == comp_task_selection].index[0]
            attr_df_next_task_index = attr_df[attr_df['event'] == next_task].index[0]
            attr_df_filtered = attr_df.loc[attr_df_current_task_index + 1:attr_df_next_task_index - 1]

            if task_aligned:
                # Align
                align_datetime = datetime.strptime('2023-05-04 00:00:00', '%Y-%m-%d %H:%M:%S')
                datetime_offset_bubble = id_attr_df_filtered['start'].iloc[0] - align_datetime
                id_attr_df_filtered_aligned = id_attr_df_filtered.copy()
                id_attr_df_filtered_aligned['start'] = id_attr_df_filtered['start'] - datetime_offset_bubble
                id_attr_df_filtered_aligned['end'] = id_attr_df_filtered['end'] - datetime_offset_bubble
                id_attr_df_filtered = id_attr_df_filtered_aligned.copy()

                datetime_offset = attr_df_filtered['start'].iloc[0] - align_datetime
                attr_df_filtered_aligned = attr_df_filtered.copy()
                attr_df_filtered_aligned['start'] = attr_df_filtered['start'] - datetime_offset
                attr_df_filtered = attr_df_filtered_aligned.copy()

        else:
            id_attr_df_filtered = bubble_df.copy()
            attr_df_filtered = attr_df.copy()

        # Preprocess and start drawing
        bubble_df_filtered = id_attr_df_filtered[
            ~id_attr_df_filtered.apply(lambda row: row.astype(str).str.contains('Task').any(), axis=1)]
        bubble_df_filtered.reset_index(drop=True, inplace=True)

        # Attribute plots
        if line_scatter_switch_comp == 'scatter':
            attr_plot = px.scatter(attr_df_filtered, x='start', y=comp_attribute_selection, color='participant')
        else:
            if line_interpolation:
                attr_plot = px.line(attr_df_filtered, x='start', y=comp_attribute_selection, color='participant',
                                    line_shape='hv')
            else:
                attr_plot = px.line(attr_df_filtered, x='start', y=comp_attribute_selection, color='participant')
            attr_plot.update_traces(line=dict(dash='solid', color='black'))

        attr_plot.update_traces(yaxis='y2')

        bg = px.timeline(bubble_df_filtered, x_start='start', x_end='end', y='participant', color='ID',
                         color_discrete_sequence=px.colors.qualitative.Set3)

        for trace in attr_plot.data:
            fig.add_trace(trace, row=i_ + 1, col=1, secondary_y=True)
            # fig.update_layout(yaxis_range=[-3, 3])
        for trace in bg.data:
            fig.add_trace(trace, row=i_ + 1, col=1)

        # Add the vertical task dotted lines if showing all tasks
        if comp_task_selection == 'All':
            task_rows = attr_df_filtered[attr_df_filtered['event'].str.contains('Task')]
            # Store the respective 'start' column values into a normal Python list
            tasks = task_rows['start'].tolist()
            for task in tasks:
                fig.add_vline(x=task, line_width=3, line_dash="dot", line_color="black", row=i_ + 1, col=1)

        # for trace in attr_plot.data + bg.data:
        #     fig.add_trace(trace, row=i_+1, col=1)

    # # Make the ID as string to show as category, else it will be treated as continuous
    # df_bubble_bg['ID'] = df_bubble_bg['ID'].astype(str)
    # # Use boolean indexing to exclude rows containing 'Task'
    # df_bubble_bg_filtered = df_bubble_bg[
    #     ~df_bubble_bg.apply(lambda row: row.astype(str).str.contains('Task').any(), axis=1)]
    # df_bubble_bg_filtered.reset_index(drop=True, inplace=True)
    #
    # # Create a plot for year attribute
    # attr_plot = px.line(df_id_attr, x='start', y='year', color='participant')
    # # Change the color to black and set it as the secondary y axis
    # attr_plot.update_traces(line=dict(dash='solid', color='black'), yaxis='y2')
    #
    # # Create a bar chart for bubble, used as a background
    # bg = px.timeline(df_bubble_bg_filtered, x_start='start', x_end='end', y='participant', color='ID')
    #
    # # fig_1 = px.bar(df_bubble_bg, base='start_int', x='duration', y='participant', color='ID',
    # #               orientation='h', hover_data={'event': True})
    #
    # for trace in attr_plot.data:
    #     fig.add_trace(trace, row=1, col=1, secondary_y=True)
    # for trace in bg.data:
    #     fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(type='date')

    if comp_attribute_selection == 'gender':
        fig.update_yaxes(dtick=1)
        category_array = ['both', 'male', 'female']
        category_order = [-1, len(category_array)]
    elif comp_attribute_selection == 'year':
        fig.update_yaxes(dtick=4)
        category_array = [str(year) for year in range(1990, 2018)]
        category_order = [1988, 2018]
    else:
        fig.update_yaxes(dtick=1)
        category_array = [None, 'Vadstena', 'Motala', 'Kinda', 'Valdemarsvik', 'Finsp?ng', 'Mj?lby', 'Link?ping',
                          'Boxholm', 'Ydre', 'Katrineholm', '?tvidaberg', 'S?derk?ping', '?desh?g', 'Norrk?ping']
        category_order = [-1, len(category_array)]

    fig.update_yaxes(range=category_order,
                     categoryarray=category_array,
                     secondary_y=True)

    fig.update_layout(barmode='stack', bargap=0, autosize=False, xaxis_scaleanchor='y',
                      # yaxis=dict(categoryorder='category descending'),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=1080)

    fig.update_traces(showlegend=False)
    # Customize the x-axis and y-axis to make their lines black
    fig.update_xaxes(
        linecolor='black',
        linewidth=2,
    )

    # fig.update_traces(opacity=.4)

    # add horizontal lines
    # fig.add_hline(y=0)

    return fig


# Cosine similarity calculation
# @app.callback(
#     Output('cos_result', 'children'),
#     Input('compare_exe', 'n_clicks'),
#     State('cos_1', 'value'),
#     State('cos_2', 'value'),
#     State('attr_selection', 'value'),
#     prevent_initial_call=True
# )
# def set_participant(compare_exe, cos_1_value, cos_2_value, attr_selection):
#     # Testing with cos similarity
#     tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
#     attribute = str(attr_selection)
#
#     p1 = fetch_csv('P' + str(cos_1_value) + '_id_attr.csv')
#     p1 = p1[-p1['event'].isin(tasks)]
#     p2 = fetch_csv('P' + str(cos_2_value) + '_id_attr.csv')
#     p2 = p2[-p2['event'].isin(tasks)]
#
#     attr_vec_1 = p1[attribute]
#     attr_vec_1 = attr_vec_1.dropna()
#     attr_vec_1 = attr_vec_1.astype(str)
#
#     attr_vec_2 = p2[attribute]
#     attr_vec_2 = attr_vec_2.dropna()
#     attr_vec_2 = attr_vec_2.astype(str)
#
#     # Combine the vectors
#     combined_vector = [attr_vec_1, attr_vec_2]
#     # print(combined_vector)
#
#     # Initialize the TF-IDF vectorizer
#     vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
#
#     # Transform the combined vectors into TF-IDF vectors
#     tfidf_matrix = vectorizer.fit_transform(combined_vector)
#
#     # Convert the matrix to dense arrays
#     dense_tfidf = tfidf_matrix.toarray()
#
#     similarity = 1 - cosine(dense_tfidf[0], dense_tfidf[1])
#
#     return round(similarity, 2)  # f"Button clicked\nState values: cos_1 = {cos_1_value}, cos_2 = {cos_2_value}"

# Switch for the 'Line Interpolation' option in individual view
@app.callback(
    Output('line_interpolation_individual', 'options'),
    Output('line_interpolation_individual', 'style'),
    Input('line_scatter_switch_individual', 'value'),
)
def color_button_toggle(line_scatter_switch_individual):
    if line_scatter_switch_individual == 'scatter':
        new_options = [{'label': 'Attribute interpolation', 'value': False, 'disabled': True}]
        new_style = {'color': 'lightGray'}
        return new_options, new_style
    else:
        new_options = [{'label': 'Attribute interpolation', 'value': True, 'disabled': False}]
        new_style = {}
        return new_options, new_style


@app.callback(
    Output('color_switch', 'options'),
    Output('color_switch', 'style'),
    Input('complete_bubble_switch', 'value'),
    prevent_initial_call=True
)
def color_button_toggle(complete_bubble_switch):
    if complete_bubble_switch:
        new_options = [{'label': 'Toggle color by view (default by ID)', 'value': False, 'disabled': True}]
        new_style = {'color': 'lightGray'}
        return new_options, new_style
    else:
        new_options = [{'label': 'Toggle color by view (default by ID)', 'value': True, 'disabled': False}]
        new_style = {}

        return new_options, new_style


# Draw timeline plot
@app.callback(
    Output('timeline_plot', 'figure'),
    Output('total_bubble_number', 'children'),
    Output('complete_bubble_number', 'children'),
    Output('incomplete_bubble_number', 'children'),
    Output('interaction_number', 'children'),
    Output('total_duration', 'children'),
    Output('task_duration', 'children'),

    Input('view_switch', 'value'),
    Input('bubble_switch', 'value'),
    Input('year_switch', 'value'),
    Input('municipality_switch', 'value'),
    Input('gender_switch', 'value'),
    Input('complete_bubble_switch', 'value'),
    Input('color_switch', 'value'),
    Input('participant_selection', 'value'),
    Input('task_selection', 'value'),
    Input('toggle_timestamp', 'value'),
    Input('line_scatter_switch_individual', 'value'),
    Input('line_interpolation_individual', 'value'),
    # prevent_initial_call=True
)
def timeline_plot_control(view_switch, bubble_switch, year_switch, municipality_switch, gender_switch,
                          complete_bubble_switch, color_switch, participant_selection, task_selection,
                          toggle_timestamp, line_scatter_switch_individual, line_interpolation_individual):
    global df, df_vc, df_id_attr  # participant, participant_number,

    # print(task_selection)

    # Reset the dataframes
    participant_number = participant_selection
    participant = 'P' + str(participant_number)
    df = fetch_csv(participant + '.csv')
    df_vc = fetch_csv(participant + '_vc.csv')
    df_id_attr = fetch_csv(participant + '_id_attr.csv')
    df_id_attr['ID'] = df_id_attr['ID'].astype(str)
    df_id_attr = df_id_attr.sort_values(by='start')

    # Check the number of interaction
    interaction_number = len(df_id_attr) - 6  # Minus the number of 'Task' row

    # Extract the total duration of experiment
    whole_time_str = str(df_id_attr.iloc[-1]['start'])
    datetime_obj = datetime.strptime(whole_time_str, '%Y-%m-%d %H:%M:%S')
    whole_minutes = datetime_obj.minute
    whole_seconds = datetime_obj.second

    total_duration_str = str(whole_minutes) + ':' + str(whole_seconds)

    # Condition for toggle complete/incomplete bubbles
    # print(complete_bubble_switch)
    if color_switch:
        color = 'view'
    else:
        color = 'ID'

    # Create an empty plot
    place_holder = pd.DataFrame(columns=['start', 'end', 'participant'])
    fig = px.timeline(place_holder, x_start='start', x_end='end', y='participant')

    plot_trace = []
    color_mapping = {
        'EducationButton': 'orange',
        'IncomeButton': 'green',
        'AgeButton': 'magenta',
        'PopulationButton': 'cyan',
    }

    # Stats about the bubbles and incomplete bubbles
    complete_bubble_df = df_id_attr[df_id_attr['complete']]
    complete_bubble_number = complete_bubble_df['ID'].nunique()
    total_bubble_number = df_id_attr['ID'].nunique()
    incomplete_bubble_number = total_bubble_number - complete_bubble_number

    # print(df_id_attr_filtered.head())

    if task_selection != 'All':
        # print(task_selection)
        current_task = int(task_selection[-1])
        next_task = 'Task ' + str(current_task + 1)

        current_task_index = df_id_attr[df_id_attr['event'] == task_selection].index[0]
        next_task_index = df_id_attr[df_id_attr['event'] == next_task].index[0]

        df_id_attr_filtered = df_id_attr.loc[current_task_index + 1:next_task_index - 1]

        # Get the custom time range for x-axis
        start_time = df_id_attr[df_id_attr['event'] == task_selection].iloc[0]['start']
        end_time = df_id_attr[df_id_attr['event'] == next_task].iloc[0]['start']
        x_axis = [start_time, end_time]

        # Get the duration
        task_duration = end_time - start_time
        time_str = str(task_duration)
        components = time_str.split()
        minutes, seconds = map(int, components[2].split(':')[-2:])
        task_duration = str(minutes) + ':' + str(seconds)

    else:
        tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
        df_id_attr_filtered = df_id_attr[-df_id_attr['event'].isin(tasks)]

        # Get the full range for x-axis
        start_time = df_id_attr.iloc[0]['start']
        end_time = df_id_attr.iloc[-1]['start']
        x_axis = [start_time, end_time]

        # No need to show single task duration when viewing the full experiment
        task_duration = '(select a task first)'

    if bubble_switch:
        if complete_bubble_switch:
            id_trace = px.timeline(df_id_attr_filtered, x_start='start', x_end='end', y='ID', color=color,
                                   pattern_shape='complete', pattern_shape_sequence=['/', ''],
                                   color_discrete_map=color_mapping).data
        else:
            df_id_attr_filtered = df_id_attr_filtered[df_id_attr_filtered['complete']]
            id_trace = px.timeline(df_id_attr_filtered, x_start='start', x_end='end', y='ID', color=color,
                                   color_discrete_map=color_mapping).data
        plot_trace = id_trace

    if year_switch:
        if line_scatter_switch_individual == 'scatter':
            year_trace = px.scatter(df_id_attr_filtered, x='start', y='year', color=color,
                                    color_discrete_map=color_mapping).data
        else:
            if line_interpolation_individual:
                year_trace = px.line(df_id_attr_filtered, x='start', y='year', color=color,
                                     color_discrete_map=color_mapping, line_shape='hv').data
            else:
                year_trace = px.line(df_id_attr_filtered, x='start', y='year', color=color,
                                     color_discrete_map=color_mapping).data

        plot_trace += year_trace

    if municipality_switch:
        if line_scatter_switch_individual == 'scatter':
            municipality_trace = px.scatter(df_id_attr_filtered, x='start', y='municipality', color=color,
                                            color_discrete_map=color_mapping).data
        else:
            if line_interpolation_individual:
                municipality_trace = px.line(df_id_attr_filtered, x='start', y='municipality', color=color,
                                             color_discrete_map=color_mapping, line_shape='hv').data
            else:
                municipality_trace = px.line(df_id_attr_filtered, x='start', y='municipality', color=color,
                                             color_discrete_map=color_mapping).data
        plot_trace += municipality_trace

    if gender_switch:
        if line_scatter_switch_individual == 'scatter':
            gender_trace = px.scatter(df_id_attr_filtered, x='start', y='gender', color=color,
                                      color_discrete_map=color_mapping).data
        else:
            if line_interpolation_individual:
                gender_trace = px.line(df_id_attr_filtered, x='start', y='gender', color=color,
                                       color_discrete_map=color_mapping, line_shape='hv').data
            else:
                gender_trace = px.line(df_id_attr_filtered, x='start', y='gender', color=color,
                                       color_discrete_map=color_mapping).data
        plot_trace += gender_trace

    # print(df_vc)
    if view_switch:
        view_trace = px.timeline(df_vc, x_start='start', x_end='end', y='participant', color='button',
                                 color_discrete_map=color_mapping).data
        plot_trace += view_trace

    # Assemble the graph
    for trace in plot_trace:
        fig.add_trace(trace)

    # Add the vertical task dotted lines
    task_rows = df_vc[df_vc['button'].str.contains('Task')]
    # Store the respective 'start' column values into a normal Python list
    tasks = task_rows['start'].tolist()
    for task in tasks:
        fig.add_vline(  # x=datetime.strptime(str(task), '%Y-%m-%d %H:%M:%S').timestamp() * 1000,
            x=task,
            line_width=2, line_dash="dash", line_color="black",
            # annotation_text=str(task.minute) + ':' + str(task.second),
        )
        # Add text annotation for minute and second
        if toggle_timestamp:
            fig.add_annotation(
                # x=datetime.strptime(str(task), '%Y-%m-%d %H:%M:%S').timestamp() * 1000,
                x=task,
                y=-0.5,  # Adjust the Y-coordinate as needed
                text=f"<b>{task.minute:02d}:{task.second:02d}</b>",
                ax=30,
                ay=30,
                font=dict(color='white'),
                bgcolor="cadetBlue",
            )

    fig.update_xaxes(range=x_axis)

    # print(df_id_attr.head())
    # for i, row in df_id_attr.iterrows():
    #     if 'Task' in row['event']:
    #         fig.update_yaxes(tickvals=[row['event']], ticktext=[''])  # Hide the tick value

    # Remove background and increase height
    fig.update_layout(height=980,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      )
    return fig, total_bubble_number, complete_bubble_number, incomplete_bubble_number, interaction_number, \
           total_duration_str, task_duration


# --------------------------------------------------------------------------
# start the app
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
