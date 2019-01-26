from flask import Blueprint, render_template, request, redirect, url_for
from mlgo.datatraining.analysis import Analysis
from mlgo.visualization.graphs import facets, scatter
from mlgo.main.utils import get_tooltip
import pandas as pd
import seaborn as sns
import os

visualization = Blueprint('visualization', __name__)


@visualization.route("/visualizations/<string:dataset_name>")
def visualize(dataset_name):
    analysis = Analysis(dataset_name)
    pca = analysis.pca()
    chisq = analysis.chi2()
    file_name_1, file_name_2 = facets(dataset_name)
    file_list = scatter(dataset_name)
    tooltip = get_tooltip()

    input_path = os.path.join(visualization.root_path, '../static/data/', dataset_name)

    df = pd.read_csv(input_path)
    sns_plot = sns.pairplot(df, size=2.5)

    output_path = os.path.join(visualization.root_path, '../static/data/', dataset_name+'.png')

    sns_plot.savefig(output_path)

    return render_template('visualizations.html', title='Visualization',
                           dataset_name=dataset_name, pca=pca, chisq=chisq, facet_dive=file_name_1,
                           facet_overview=file_name_2, plotly_scatter_list=file_list, tooltip=tooltip,
                           output_path='../static/data/'+dataset_name+'.png')

@visualization.route('/visulaization_regression/<string:dataset_name>')
def visualize_regression(dataset_name):
    analysis = Analysis(dataset_name)
    pca = analysis.pca()
    # chisq = analysis.chi2()
    file_name_1, file_name_2 = facets(dataset_name)
    file_list = scatter(dataset_name)
    tooltip = get_tooltip()

    input_path = os.path.join(visualization.root_path, '../static/data/', dataset_name)

    df = pd.read_csv(input_path)
    sns_plot = sns.pairplot(df.iloc[:, 0:5], size=2.5)

    output_path = os.path.join(visualization.root_path, '../static/data/', dataset_name + '.png')

    sns_plot.savefig(output_path)

    return render_template('visualizations_regression.html', title='Visualization',
                           dataset_name=dataset_name, pca=pca, facet_dive=file_name_1,
                           facet_overview=file_name_2, plotly_scatter_list=file_list, tooltip=tooltip,
                           output_path='../static/data/' + dataset_name + '.png')
