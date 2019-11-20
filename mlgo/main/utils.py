import os
from flask import current_app
import random
import urllib.request
from newsapi import NewsApiClient
import datetime
from dateutil.relativedelta import relativedelta


def save_file(form_file):
    file_name, file_extension = os.path.splitext(form_file.filename)
    f_name = file_name + file_extension
    file_path = os.path.join(current_app.root_path, 'static/data', f_name)

    form_file.save(file_path)

    return f_name


def download(url):
    name = random.randrange(1, 1000)
    full_name = str(name) + ".csv"
    file_path = os.path.join(current_app.root_path, 'static/data', full_name)
    urllib.request.urlretrieve(url, file_path)
    return full_name


def get_news():
    newsapi = NewsApiClient(api_key='4dbc17e007ab436fb66416009dfb59a8')

    today = datetime.datetime.now()
    one_month_before = datetime.datetime.now() - relativedelta(months=1)
    date_formated_today = today.strftime("%Y-%m-%d")
    date_formated_one_month_before = one_month_before.strftime("%Y-%m-%d")

    print(date_formated_today, date_formated_one_month_before)

    all_articles = newsapi.get_everything(q='machine learning',
                                          sources='bbc-news,google-news,google-news-in,time,cnn,the-verge,financial-times',

                                          from_param=date_formated_one_month_before,
                                          to=date_formated_today,
                                          language='en',
                                          sort_by='relevancy',
                                          page=2)

    articles = all_articles['articles']
    result = []
    for article in articles:
        t_dict = {}
        for k, v in article.items():
            if k == 'source':
                for key, val in v.items():
                    t_dict[key] = val
            else:
                t_dict[k] = v

        result.append(t_dict)

    return result


def get_tooltip():
    tooltips = {
        'Decision Tree':'Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.',
        'Support Vector Machine':'An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.',
        'Gaussian Naive Bayes':'Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes\' theorem with strong (naive) independence assumptions between the features. Gaussian Naive Bayes assumes that the continuous values associated with each class are distributed according to a Gaussian distribution',
        'KNN':'Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.',
        'CNN':'Multi-layer Perceptron',
        'Random Forest':'A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.',
        'PCA':'Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.',
        'Chi2':'The chi-squared test is used to determine whether there is a significant difference between the expected frequencies and the observed frequencies in one or more categories.',
        'Accuracy':'The percentage of predictions that matched with true values.',
        'TrainTest':'The ratio in which the provided data has been divided into training and test sets.',
        'FeatureSelection':'Number of features used to train the model and make predictions',
        'Normalisation':'The Scaling technique used for regularisation of data.',
        'AboutUs':'A breif description about the awesome developers who made this application.',
        'News':'The latest happenings in the world of Data Science',
        'rms':'Mean squared error regression loss',
        'SGD':'This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate)',
        'Lasso Regression':'Linear Model trained with L1 prior as regularizer (aka the Lasso)',
        'Linear Regression':'Ordinary least squares Linear Regression.'
    }

    return tooltips
