
from flask import Flask


def create_app(config_class=Config):
    app = Flask(__name__)

    from mlgo.visualization.routes import visualization
    from mlgo.main.routes import main

    app.register_blueprint(visualization)
    app.register_blueprint(main)

    return app
