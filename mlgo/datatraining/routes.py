from flask import Blueprint, render_template, request, redirect, url_for
from mlgo.datatraining.ml_models import ML_Models as ml_models
from mlgo.datatraining.ml_models_regression import RegressionModels as ml_models_regression
from mlgo.datatraining.forms import OptionsForm
from mlgo.main.utils import get_tooltip

data_training = Blueprint('datatraining', __name__)


@data_training.route("/dashboard_regression/<string:dataset_name>")
def dashboard_regression(dataset_name):
    obj = ml_models_regression(dataset_name)
    obj.clean_data()

    rs_r = [obj.lasso(), obj.sgd(), obj.linear()]
    tooltip = get_tooltip()
    return render_template('dashboard_regression.html', title='Dashboard', tooltip=tooltip, dataset_name=dataset_name,
                           mlmodels=rs_r)


@data_training.route("/dashboard/<string:dataset_name>")
def dashboard(dataset_name):
    obj = ml_models(dataset_name)
    obj.clean_data()

    rs = [obj.decision_tree(), obj.svm(), obj.naive_bayes(), obj.knn(), obj.cnn(), obj.random_forest(), obj.adaboost()]
    tooltip = get_tooltip()
    return render_template('dashboard.html', title='Dashboard', mlmodels=rs, dataset_name=dataset_name, tooltip=tooltip)


@data_training.route("/dashboard/<string:dataset_name>/<string:algo_name>/options", methods=['GET', 'POST'])
def options(dataset_name, algo_name):
    form = OptionsForm()

    current_algo = {'Decision Tree': False, 'Support Vector Machine': False, 'Naive Bayes': False, 'KNN': False, 'Random Forest': False, 'CNN': False, 'SGD': False, 'Adaboost':False}
    current_algo[algo_name] = True
    obj = ml_models(dataset_name)
    obj.clean_data()
    tooltip = get_tooltip()

    if request.method == 'POST':
        print("form.validate_on_submit(): ", form.validate_on_submit())
        print(request.form)
        if algo_name == 'Decision Tree':
            print(request.form['exampleRadios'], ", ", form.max_depth.data, ", ", form.min_samples_leaf.data,
                  ", ", form.min_samples_split.data,", ", request.form['aa'], request.form['feature_reduction'], form.p.data)

            if form.min_samples_split.data == '':
                form.min_samples_split.data = '2'
            elif form.min_samples_leaf.data == '':
                form.min_samples_leaf.data = '1'

            rs = obj.decision_tree(criterion=request.form['exampleRadios'], max_depth=form.max_depth.data,
                                   min_samples_split=int(form.min_samples_split.data),
                                   min_samples_leaf=int(form.min_samples_leaf.data),
                                   scaler=request.form['aa'], feature_selection=request.form['feature_reduction'],
                                   p=form.p.data)
        elif algo_name == 'Support Vector Machine':
            print("in post request svm")
            print(request.form['kernel'], " ", form.gamma.data, " ",request.form['scaler'])
            rs = obj.svm(c=form.c.data,kernel=request.form['kernel'], gamma=form.gamma.data, max_iter=form.max_iterations.data,
                         scaler=request.form['scaler'], feature_selection=request.form['feature_reduction'],
                         p=form.p.data)
        elif algo_name == 'Naive Bayes':
            rs = obj.naive_bayes(scaler=request.form['inlineRadioOptions'], feature_selection=request.form['feature_reduction'],
                         p=form.p.data)
        elif algo_name == 'KNN':
            rs = obj.knn(n_neighbors=form.n_neighbors.data, algorithm=request.form['algo'],
                         weights=request.form['weights'], leaf_size=form.leaf_size.data, scaler=request.form['inlineRadioOptions'],
                         feature_selection=request.form['feature_reduction'],
                         p=form.p.data)
        elif algo_name == 'Random Forest':
            rs = obj.random_forest(criterion=request.form['exampleRadios'], max_depth=form.max_depth.data,
                                   min_samples_split=int(form.min_samples_split.data),
                                   min_samples_leaf=int(form.min_samples_leaf.data),
                                   scaler=request.form['inlineRadioOptions'], feature_selection=request.form['feature_reduction'],
                                   p=form.p.data)
        elif algo_name == 'CNN':
            rs = obj.cnn(alpha=form.alpha.data, activation=request.form['activation'], max_iter=form.max_iter.data,
                         scaler=request.form['inlineRadioOptions'], feature_selection=request.form['feature_reduction'],
                         p=form.p.data)
        elif algo_name == 'Adaboost':
            rs = obj.adaboost(algorithm=request.form['algorithm'], n_estimators=form.n_estimators.data, learning_rate=form.learning_rate.data,
                              scaler=request.form['inlineRadioOptions'], feature_selection=request.form['feature_reduction'],
                              p=form.p.data)

        return render_template('options.html', title='Parameter Tuning', form=form, mlmodel=rs,
                               current_algo=current_algo, dataset_name=dataset_name, tooltip=tooltip)

    print(algo_name)

    if algo_name == 'Decision Tree':
        rs = obj.decision_tree()
    elif algo_name == 'Support Vector Machine':
        print("in get request svm")
        rs = obj.svm()
    elif algo_name == 'Naive Bayes':
        rs = obj.naive_bayes()
    elif algo_name == 'KNN':
        print("in get request KNN")
        rs = obj.knn()
    elif algo_name == 'Random Forest':
        print("in get request Random Forest")
        rs = obj.random_forest()
    elif algo_name == 'CNN':
        print("in get request CNN")
        rs = obj.cnn()
    elif algo_name == 'Adaboost':
        rs = obj.adaboost()

    return render_template('options.html', title='Parameter Tuning', form=form, mlmodel=rs, current_algo=current_algo,
                           dataset_name=dataset_name, tooltip=tooltip)


@data_training.route("/options_regression/<string:dataset_name>/<string:algo_name>/options", methods=['GET', 'POST'])
def options_regression(dataset_name, algo_name):
    print("***", " In option Regression", "********")
    form = OptionsForm()

    current_algo = {'SGD': False, 'Lasso Regression': False, 'Linear Regression': False}
    current_algo[algo_name] = True
    obj = ml_models_regression(dataset_name)
    obj.clean_data()
    tooltip = get_tooltip()

    if request.method == 'POST':
        print("form.validate_on_submit(): ", form.validate_on_submit())
        print(request.form)
        print(algo_name)
        if algo_name == 'SGD':
            rs = obj.sgd(loss=request.form['loss'], penalty=request.form['penalty'], alpha=form.alpha.data, max_iter=form.max_iter.data, scaler=request.form['inlineRadioOptions'], feature_selection=request.form['feature_reduction'], p=form.p.data)
        elif algo_name == 'Lasso Regression':
            rs = obj.lasso(alpha=form.alpha.data, scaler=request.form['inlineRadioOptions'], feature_selection=request.form['feature_reduction'], p=form.p.data)
        elif algo_name == 'Linear Regression':
            rs = obj.linear(scaler=request.form['inlineRadioOptions'], feature_selection=request.form['feature_reduction'], p=form.p.data)

        return render_template('options_regression.html', title='Parameter Tuning', form=form, mlmodel=rs,
                               current_algo=current_algo, dataset_name=dataset_name, tooltip=tooltip)

    print(algo_name, " ####################")

    if algo_name == 'SGD':
        print("In SGD")
        rs = obj.sgd()
    elif algo_name == 'Linear Regression':
        print("In Linear Regression")
        rs = obj.linear()
    elif algo_name == 'Lasso Regression':
        print("In Lasso Regression")
        rs = obj.lasso()
        print(": ", rs, " :")
    else:
        print("In else")

    return render_template('options_regression.html', title='Parameter Tuning', form=form, mlmodel=rs, current_algo=current_algo,
                           dataset_name=dataset_name, tooltip=tooltip)
