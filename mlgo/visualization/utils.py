import os
import pandas as pd
from flask import current_app
from mlgo.visualization.graphs import get_labels
import plotly.offline as pof
import plotly.graph_objs as go
import numpy as np
import secrets


def pairwise_graph(data_filename, feature_list):
    #filepath = os.path.join(current_app.root_path, 'static/data', data_filename)
    filepath = data_filename
    data = pd.read_csv(filepath, header=0)
    #print((data['Fare']))

    features, labels = get_labels(data)
    var1 = feature_list[0]
    var2 = feature_list[1]

   # num_features = feature_list.shape[1]
    trc = go.Scatter(
        x=data[var1],
        y=data[var2],
        mode='markers+text',
        marker=dict(
            color=np.random.randn(500),
            colorscale='Viridis',
        ),
        textposition='bottom center'
    )
    layout = go.Layout(
        title='Scatter Plot',
        hovermode='closest',
        xaxis=dict(
            title=feature_list[0] + ' vs ' + feature_list[1]
        ),
        yaxis=dict(
            title=labels.columns
        ),
        showlegend=False
    )
    data = [trc]
    fig = go.Figure(data=data, layout=layout)
    random_hex = secrets.token_hex(8)
    #filename = os.path.join(current_app.root_path, 'templates', random_hex + ".html")
    filename = random_hex + '.html'
    print(random_hex + 'html')
    pof.plot(fig, filename=filename, auto_open=False)
    return filename