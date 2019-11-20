from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


class OptionsForm(FlaskForm):
    feature_1 = StringField('Feature 1')
    feature_2 = StringField('Feature 2')
    feature_3 = StringField('Feature 3')
    submit = SubmitField('Go')
