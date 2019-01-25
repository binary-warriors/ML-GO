from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


class OptionsForm(FlaskForm):
    max_depth = StringField('Max Depth')
    min_samples_split = StringField('Min Samples Split')
    min_samples_leaf = StringField('Min Samples Leaf')
    submit = SubmitField('Go')
