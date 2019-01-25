from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


class UploadForm(FlaskForm):


    url = StringField('File from cloud, eg. Google Drive, Dropbox, OneDrive etc.')

    submit = SubmitField('Upload File')
    download = SubmitField('Start')

