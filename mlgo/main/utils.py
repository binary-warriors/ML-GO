import os
from flask import current_app
import random



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
    newsapi 

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
        'Gaussian Naive Bayes':'Naive n the world of Data Science'
    }

    return tooltips
