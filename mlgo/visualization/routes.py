from flask import Blueprint, render_template, request, redirect, url_for
from mlgo.datatraining.analysis import Analysis
from mlgo.visualization.graphs import facets, scatter
from mlgo.main.utils import get_tooltip
import pandas as pd
import seaborn as sns
import os
from mlgo.visualization.forms import OptionsForm

visualization = Blueprint('visualization', __name__)


@visualization.route("/visualizations/<string:dataset_name>")
def visualize(dataset_name):
    # analysis = Analysis(dataset_name)
    # pca = analysis.pca()
    # chisq = analysis.chi2()
    analysis = Analysis(dataset_name)
    _pca_result_dict = {
        'Test': 'Principal Component Analysis',
        'Score': "NA",
        'Covariance': ['NA'],
        'Precision': ['NA']
        # 'Log-Likelihood': log_likelihood
    }
    try:
        pca = analysis.pca()
    except:
        pca = _pca_result_dict
    _result_dict = {
        'Test': 'Chi-Squared Analysis',
        'Chi2Stat': 0,
        'P value': [""],
        'Degree of Freedom': [""]
    }

    try:
        chisq = analysis.chi2()
    except:
        chisq = _result_dict
    file_name_1, file_name_2 = facets(dataset_name)
    file_list, name_list = scatter(dataset_name)
    tooltip = get_tooltip()

    t_dict = dict(zip(file_list, name_list))
    print(t_dict)

    input_path = os.path.join(visualization.root_path, '../static/data/', dataset_name)

    df = pd.read_csv(input_path)
    sns_plot = sns.pairplot(df.iloc[:, 0:5], size=2.5)

    output_path = os.path.join(visualization.root_path, '../static/data/', dataset_name+'.png')

    sns_plot.savefig(output_path)

    return render_template('visualizations.html', title='Visualization',
                           dataset_name=dataset_name, pca=pca, chisq=chisq, facet_dive=file_name_1,
                           facet_overview=file_name_2, plotly_scatter_dict=t_dict, tooltip=tooltip,
                           output_path='../static/data/'+dataset_name+'.png')


@visualization.route('/graphs/<string:dataset_name>', methods=['GET', 'POST'])
def interactive_graphs(dataset_name):
    tooltip = get_tooltip()
    form = OptionsForm()
    plot=""
    if request.method == "GET":
        return render_template('interactive_graph.html', title='Graphs', form=form,
                               dataset_name=dataset_name, tooltip=tooltip,
                               output_path='../static/data/' + dataset_name + '.png')

    if request.method == "POST":
        graph = request.form['exampleRadios']
        feature_1 = form.feature_1
        feature_2 = form.feature_2
        feature_3 = form.feature_3
        print(graph)
        print(feature_1)
        print(feature_2)
        print(feature_3)
        if graph == "histogram":
            pass
        elif graph == "scatter":
            pass
        elif graph == "3D-Plot":
            pass
        pass





@visualization.route('/visulaization_regression/<string:dataset_name>')
def visualize_regression(dataset_name):
    # analysis = Analysis(dataset_name)
    # pca = analysis.pca()
    # chisq = analysis.chi2()
    analysis = Analysis(dataset_name)
    _pca_result_dict = {
        'Test': 'Principal Component Analysis',
        'Score': "NA",
        'Covariance': ['NA'],
        'Precision': ['NA']
        # 'Log-Likelihood': log_likelihood
    }
    try:
        pca = analysis.pca()
    except:
        pca = _pca_result_dict
    # _result_dict = {
    #     'Test': 'Chi-Squared Analysis',
    #     'Chi2Stat': 0,
    #     'P value': [""],
    #     'Degree of Freedom': [""]
    # }
    #
    # try:
    #     chisq = analysis.chi2()
    # except:
    #     chisq = _result_dict
    file_name_1, file_name_2 = facets(dataset_name)
    file_list, name_list = scatter(dataset_name)
    tooltip = get_tooltip()

    t_dict = dict(zip(file_list, name_list))
    print(t_dict)

    input_path = os.path.join(visualization.root_path, '../static/data/', dataset_name)

    df = pd.read_csv(input_path)
    sns_plot = sns.pairplot(df.iloc[:, 0:5], size=2.5)

    output_path = os.path.join(visualization.root_path, '../static/data/', dataset_name + '.png')

    sns_plot.savefig(output_path)

    print(file_name_1,"\n",file_name_2)

    return render_template('visualizations_regression.html', title='Visualization',
                           dataset_name=dataset_name, pca=pca, facet_dive=file_name_1,
                           facet_overview=file_name_2, plotly_scatter_dict=t_dict, tooltip=tooltip,
                           output_path='../static/data/' + dataset_name + '.png')
