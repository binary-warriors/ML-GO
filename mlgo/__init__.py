from mlgo.config import Config
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
import sys, os


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    from mlgo.main.routes import main
    
    app.register_blueprint(main)

    return app

