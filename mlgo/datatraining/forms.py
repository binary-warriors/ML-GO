from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


class OptionsForm(FlaskForm):
    max_depth = StringField('Max Depth')
    min_samples_split = StringField('Min Samples Split')
    min_samples_leaf = StringField('Min Samples Leaf')
    p = StringField('Corresponding parameter value')
    c = StringField('C value')
    gamma = StringField('Gamma value')
    max_iterations = StringField('Max iterations')
    n_neighbors = StringField('N neighbors')
    leaf_size = StringField('Leaf Size')
    alpha = StringField('alpha')
    max_iter = StringField('max iteration')
    n_estimators = StringField('n estimators')
    learning_rate = StringField('Learning Rate')
    test_train = StringField('Test : Train Ratio')
    submit = SubmitField('Go')
