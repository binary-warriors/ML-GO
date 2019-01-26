from flask import Blueprint, render_template, request, redirect, url_for
from mlgo.main.forms import UploadForm
from mlgo.main.utils import save_file, download, get_news
from mlgo.main.utils import get_tooltip

main = Blueprint('main', __name__)


@main.route("/", methods=['GET', 'POST'])
@main.route("/home", methods=['GET', 'POST'])
def home():
    form = UploadForm()
    print("in home")
    tooltip = get_tooltip()
    if request.method == 'POST':
        print("in post", form.validate_on_submit())
        if form.validate_on_submit():
            if form.url.data:
                print("downloading...")
                print(form.url.data)
                data_file = download(form.url.data)
                print(data_file)
                if request.form['stream'] == 'classification':
                    return redirect(url_for('datatraining.dashboard', dataset_name=data_file, tooltip=tooltip))
                elif request.form['stream'] == 'regression':
                 tooltip)


@main.route("/about")
def about():
    tooltip = get_tooltip()
    return render_template('about.html', title='About', tooltip=tooltip)


@main.route('/news')
def news():
    tooltip = get_tooltip()
    news_1 = get_news()
    return render_template('news.html', title='News', news_all=news_1, tooltip=tooltip)
