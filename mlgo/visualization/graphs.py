import os
import numpy as np
import secrets
import pandas as pd
from flask import current_app
import plotly.offline as pof
import plotly.graph_objs as go
from plotly import tools
import base64
from mlgo.facets.facets_overview.python.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
#from generic_feature_statistics_generator import GenericFeatureStatisticsGenerator import

HTML_TEMPLATE = """<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="600"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem").data = data;
        </script>"""

HTML_TEMPLATE_OVERVIEW = """<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html" >
        <facets-overview id="elem1"></facets-overview>
        <script>
          document.querySelector("#elem1").protoInput = "{protostr}";
        </script>"""


def facets(data_filename):
    dv = dive(data_filename)
    ov = overview(data_filename)
    return dv, ov


def dive(data_filename):
    filepath = os.path.join(current_app.root_path, 'static/data', data_filename)
    data = pd.read_csv(filepath, header=0)
    data.reset_index()
    jsonstr = data.to_json(orient='records')
    html = HTML_TEMPLATE.format(jsonstr=jsonstr)
    random_hex = secrets.token_hex(8)
    fw = open('mlgo/templates/'+random_hex+'.html', 'w')
    fw.write(html)
    fw.close()
    return random_hex+'.html'


def overview(data_filename):
    filepath = os.path.join(current_app.root_path, 'static/data', data_filename)
    data = pd.read_csv(filepath, header=0)
    data.reset_index()

    gfsg = GenericFeatureStatisticsGenerator()
    proto = gfsg.ProtoFromDataFrames([{'name': 'train', 'table': data}])
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
    html = HTML_TEMPLATE_OVERVIEW.format(protostr=protostr)
    random_hex = secrets.token_hex(8)
    fw = open('mlgo/templates/' + random_hex + '.html', 'w')
    fw.write(html)
    fw.close()
    print(random_hex + '.html')
    return random_hex + '.html'


def get_labels(data):
    df = data
    column_names = list(df)
    df.columns = list(range(0, len(df.columns)))
    features = df.drop(columns=[len(df.columns) - 1])
    labels = df.get(len(df.columns) - 1)
    features.columns = column_names[:-1]
    labels.columns = column_names[-1]
    return features, labels


def scatter_subplots(data_filename):
    filepath = os.path.join(current_app.root_path, 'static/data', data_filename)
    data = pd.read_csv(filepath, header=0)
    data.reset_index()

    features, labels = get_labels(data)

    num_features = features.shape[1]
    if num_features % 2 != 0:
        rows = int(num_features / 2) + 1
    else:
        rows = int(num_features / 2)
    fig = tools.make_subplots(rows=rows, cols=2)
    f_list = list(features)
    i = 0
    for ft in f_list:
        trc = go.Scatter(
            x=features[ft],
            y=labels,
            mode='markers+text'
        )
        j = int(i / 2) + 1
        fig.append_trace(trc, j, (i % 2) + 1)
        i = i + 1
    random_hex = secrets.token_hex(8)
    filename = os.path.join(current_app.root_path, 'templates', random_hex+".html")
    print(random_hex + 'html')
    pof.plot(fig, filename=filename, auto_open=False)
    return random_hex+".html"


def scatter(data_filename):
    filepath = os.path.join(current_app.root_path, 'static/data', data_filename)
    data = pd.read_csv(filepath, header=0)
    data.reset_index()

    features, labels = get_labels(data)

    f_list = features.columns
    scatters = []
    i = 0
    name_list = []
    for ft in f_list:
        if i > 10:
            break
        i = i+1
        trc = go.Scatter(
            x=features[ft],
            y=labels,
            mode='markers+text',
            marker=dict(
                color=np.random.randn(500),
                colorscale='Viridis'
            ),
            textposition='bottom center'
        )
        layout = go.Layout(
            title='Scatter Plot',
            hovermode='closest',
            xaxis=dict(
                title=ft
            ),
            yaxis=dict(
                title=labels.columns
            ),
            showlegend=False
        )
        name_list.append(ft+" vs "+labels.columns)
        data = [trc]
        fig = go.Figure(data=data, layout=layout)
        random_hex = secrets.token_hex(8)
        filename = os.path.join(current_app.root_path, 'templates', random_hex + ".html")
        print(random_hex + 'html')
        pof.plot(fig, filename=filename, auto_open=False)
        scatters.append(random_hex+".html")

    return scatters, name_list
