import os
import numpy as np
import secrets
import pandas as pd
from flask import current_app
import plotly.offline as pof
import plotly.graph_objs as go
from plotly import tools
import base64
from generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

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
    print(random_hex+'html')
    return random_hex+'.html'

