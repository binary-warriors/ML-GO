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
