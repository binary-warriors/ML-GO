from flask import Blueprint, render_template, request, redirect, url_for
from mlgo.datatraining.ml_models import ML_Models as ml_models
from mlgo.datatraining.forms import OptionsForm

data_training = Blueprint('datatraining', __name__)


@data_training.route("/dashboard/<string:dataset_name>")
def dashboard(dataset_name):
    obj = ml_models(dataset_name)
    obj.clean_data()

    rs = [obj.decision_tree(), obj.svm(), obj.naive_bayes(), obj.kn.now()]

    return render_template('dashboard.html', title='Dashboard', mlmodels=rs, dataset_name=dataset_name)


@data_training.route("/dashboard/<string:dataset_name>/<string:algo_name>/options", methods=['GET', 'POST'])
def options(dataset_name, algo_name):
    form = OptionsForm()

    current_algo = {'Decision Tree': False, 'Support Vector Machine': False, 'Naive Bayes': False, 'KNN': False, 'Random Forest': False, 'CNN': False, 'SGD': False}
    current_algo[algo_name] = True
    obj = ml_models(dataset_name)
    obj.clean_data()

    if request.method == 'POST':
        print("form.validate_on_submit(): ", form.validate_on_submit())
        if algo_name == 'Decision Tree':
            print(request.form['exampleRadios'], ", ", form.max_depth.data, ", ", form.min_samples_leaf.data,
                  ", ", form.min_samples_split.data)

            if form.min_samples_split.data == '':
                pass
        # elif algo_name == 'Random Forest':
        #     rs = obj.random_forest()
        # elif algo_name == 'CNN':
        #     rs = obj.cnn()
        # elif algo_name == 'SGD':
        #     rs = obj.sdg()
        return render_template('options.html', title='Parameter Tuning', form=form, mlmodel=rs,
                               current_algo=current_algo, dataset_name=dataset_name)

    if algo_name == 'Decision Tree':
        rs = obj.decision_tree()
    elif algo_name == 'Support Vector Machine':
        print("in get request svm")
        rs = obj.svm()
    elif algo_name == 'Naive Bayes':
        rs = obj.naive_bayes()
    elif algo_name == 'KNN':
        rs = obj.knn()
    # elif algo_name == 'Random Forest':
    #     rs = obj.random_forest()
    # elif algo_name == 'CNN':
    #     rs = obj.cnn()
    # elif algo_name == 'SGD':
    #     rs = obj.sdg()

    return render_template('options.html', title='Parameter Tuning', form=form, mlmodel=rs, current_algo=current_algo,
                           dataset_name=dataset_name)
