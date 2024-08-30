from flask import Flask
from flask_mysqldb import MySQL

def create_app():
    app = Flask(__name__) 
    app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'msql1234'
    app.config['MYSQL_DB'] = 'sakila'
    # app.config['MYSQL_DB'] = 'OpenAYEye'
    mysql = MySQL(app)

    return app, mysql