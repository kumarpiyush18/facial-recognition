from flask_sqlalchemy import SQLAlchemy

from backend.models import Base
from server import flask_app

db = SQLAlchemy(model_class=Base)

# initialize the app with the extension
db.init_app(flask_app)
with flask_app.app_context():
    db.create_all()
