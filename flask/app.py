import os
from flask import Flask, request, Response, flash, jsonify, session, render_template, send_from_directory, redirect, url_for
from flask_mysqldb import MySQL
from flask_cors import CORS

def create_app():
    app = Flask(__name__, static_url_path='',
                  static_folder='../dashboard/dist',
                  template_folder='../dashboard/dist') 
    app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'msql1234'
    app.config['MYSQL_DB'] = 'mock'
    mysql = MySQL(app)  
    CORS(app)
    return app, mysql

app, mysql = create_app()

@app.get("/health")
def health():
    return {"message": "health ok"}

@app.route('/')
def home():
    return app.send_static_file('index.html')

def show_tables():
    curs = mysql.connection.cursor()
    curs.execute('''SHOW TABLES''')
    fetched = curs.fetchall()
    curs.close()
    return fetched 


def select_All_Tables():
    tables = [i[0] for i in show_tables()]
    fetched = {}
    curs = mysql.connection.cursor()
    for table in tables:
        curs.execute('''SELECT * FROM {} LIMIT 10'''.format(table))
        fetched['{}'.format(table)] = curs.fetchall()
    curs.close()
    return fetched['users']
    # [print(k) for k in fetched.keys()]
    # return {'message': 'ok'}
    # return {'Entry': fetched['country']}
    # return {"{}".format(table) : fetched}

@app.get("/get_All")
def get_All():
    return {"Entries": select_All_Tables()}
    # return {'message': 'ok'}

@app.get("/get_Table")
def get_Table(table="users"):
    curs = mysql.connection.cursor()
    curs.execute('''SELECT * FROM {} LIMIT 10'''.format(table))
    fetched = curs.fetchall()
    curs.close()
    # print(fetched)
    return {"{}".format(table): fetched}

@app.get("/get_Tables")
def get_Tables():
    return {"Tables": show_tables()}
    # return {'message': 'ok'}

if __name__ == '__main__':  
    app.run(host='localhost', port=5000, debug=True) 

# flask --app app.py run