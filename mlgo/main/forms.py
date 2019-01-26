from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


class UploadForm(FlaskForm):
    data_file = FileField('Upload data file from local storage', validators=[FileAllowed(['csv'])])

    url = StringField('File from cloud, eg. Google Drive, Dropbox, OneDrive etc.')

    submit = SubmitField('Upload File')
    download = SubmitField('Start')
