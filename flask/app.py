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
    cors = CORS(app)
    return app, mysql

app, mysql = create_app()

@app.get("/health")
def health():
    return {"message": "health ok"}

@app.route('/')
def home():
    return app.send_static_file('index.html')

# @app.get("/get_Table")
@app.route('/getTable', methods=['GET'])
def get_Entries(columns = ['id', 'name', 'email', 'age', 'phone', 'access'], table = 'users'):
    curs = mysql.connection.cursor()
    col_string = ", ".join(columns)
    curs.execute('''SELECT {} FROM {} LIMIT 10'''.format(col_string, table))
    fetched = curs.fetchall()
    curs.close()
    json_entry = [dict(zip(columns, i)) for i in fetched]
    return {table : json_entry}

@app.get("/showTables")
def show_Tables():
    curs = mysql.connection.cursor()
    curs.execute('''SHOW TABLES''')
    fetched = curs.fetchall()
    curs.close()
    return {'Tables': fetched}

if __name__ == '__main__':  
    app.run(host='localhost', port=5000, debug=True) 

# flask --app app.py run