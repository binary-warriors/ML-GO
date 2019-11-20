from flask import Blueprint, render_template
from mlgo.main.utils import get_tooltip

errors = Blueprint('errors', __name__)


@errors.app_errorhandler(404)
def error_404(error):
    tooltip = get_tooltip()
    return render_template('errors/404.html', tooltip=tooltip), 404


@errors.app_errorhandler(403)
def error_404(error):
    tooltip = get_tooltip()
    return render_template('errors/403.html', tooltip=tooltip), 403


@errors.app_errorhandler(500)
def error_500(error):
    tooltip = get_tooltip()
    return render_template('errors/404.html', tooltip=tooltip), 500
