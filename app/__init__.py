from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask import Flask
    
app = Flask(__name__)
app.config['SECRET_KEY'] = 'top secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
#app.config.from_object('config')
app.config.from_object('secrets')

db = SQLAlchemy(app)
lm = LoginManager(app)
lm.login_view = 'index'

from app import views, models