from flask import Blueprint, render_template

main = Blueprint('main', __name__)


@main.route("/", methods=['GET', 'POST'])
@main.route("/home", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@main.route("/about")
def about():
    return render_template('about.html', title='About')
