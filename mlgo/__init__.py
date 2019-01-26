from mlgo.config import Config
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
import sys, os

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    FACET_PATH = os.path.join(app.root_path, 'facets/facets_overview/python')
    print(FACET_PATH)
    sys.path.append(FACET_PATH)
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    from mlgo.visualization.routes import visualization
    from mlgo.datatraining.routes import data_training
    from mlgo.main.routes import main
    from mlgo.errors.handlers import errors

    app.register_blueprint(visualization)
    app.register_blueprint(data_training)
    app.register_blueprint(main)
    app.register_blueprint(errors)

    return app
