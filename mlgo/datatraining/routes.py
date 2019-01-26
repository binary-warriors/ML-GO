from flask import Blueprint, render_template, request, redirect, url_for
from mlgo.datatraining.ml_models import ML_Models as ml_models
from mlgo.models import ResultSet
from mlgo.datatraining.forms import OptionsForm
from mlgo.main.utils import get_tooltip

data_training = Blueprint('datatraining', __name__)


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

    if request.method :
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
        # elif algo_name == 'SGD':
        #     rs = obj.sdg()

        return render_template('options.html', title='Parameter Tuning', form=form, mlmodel=rs,
                               current_algo=current_algo, dataset_name=dataset_name, tooltip=tooltip)

    if algo_name == 'Decision Tree':
        rs = obj.decision_tree()
    elif algo_name == 'Support Vector Machine':
        print("in get request svm")
        rs = obj.svm()
    elif algo_name == 'Naive Bayes':
        rs = obj.naive_bayes()
    elif algo_name == 'KNN':
        rs = obj.knn()
    elif algo_name == 'Random Forest':
        rs = obj.random_forest()
    elif algo_name == 'CNN':
        rs = obj.cnn()
    elif algo_name == 'Adaboost':
        rs = obj.adaboost()
    # elif algo_name == 'SGD':
    #     rs = obj.sdg()

    return render_template('options.html', title='Parameter Tuning', form=form, mlmodel=rs, current_algo=current_algo,
                           dataset_name=dataset_name, tooltip=tooltip)
