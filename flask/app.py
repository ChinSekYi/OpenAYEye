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

with app.app_context():
    curs = mysql.connect.cursor()
    query_string = '''
            DROP TABLE IF EXISTS users;
            CREATE TABLE users (
            id SMALLINT UNSIGNED NOT NULL UNIQUE,
            name VARCHAR(45) NOT NULL,
            email VARCHAR(45) NOT NULL,
            age SMALLINT UNSIGNED NOT NULL,
            phone CHAR(20) NOT NULL,
            access VARCHAR(45) NOT NULL,
            PRIMARY KEY  (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            SET AUTOCOMMIT=0;
            INSERT INTO users (id, name, email, age, phone, access) VALUES 
            (1, "Jon Snow", "jonsnow@gmail.com", 35, "(665)121-5454", "admin"),
            (2, "Cersei Lannister", "cerseilannister@gmail.com", 42, "(421)314-2288", "manager"),
            (3, "Jaime Lannister", "jaimelannister@gmail.com", 45, "(422)982-6739", "user"),
            (4, "Anya Stark", "anyastark@gmail.com", 16, "(921)425-6742", "admin"),
            (5, "Daenerys Targaryen", "daenerystargaryen@gmail.com", 31, "(421)445-1189", "user"),
            (6, "Ever Melisandre", "evermelisandre@gmail.com", 150, "(232)545-6483", "manager"),
            (7, "Ferrara Clifford", "ferraraclifford@gmail.com", 44, "(543)124-0123", "user"),
            (8, "Rossini Frances", "rossinifrances@gmail.com", 36, "(222)444-5555", "user"),
            (9, "Harvey Roxie", "harveyroxie@gmail.com", 65, "(444)555-6239", "admin");

            COMMIT;
        '''
    curs.execute(query_string)
    mysql.connect.commit()
    curs.close()



@app.get("/health")
def health():
    return {"message": "health ok"}

@app.route('/')
def home():
    return app.send_static_file('index.html')

# @app.get("/get_Table")
@app.route('/getTable', methods=['GET'])
def get_Entries(columns = ['id', 'name', 'email', 'age', 'phone', 'access'], table = 'users'):
    curs = mysql.connect.cursor()
    col_string = ", ".join(columns)
    curs.execute('''SELECT {} FROM {} LIMIT 10'''.format(col_string, table))
    fetched = curs.fetchall()
    curs.close()
    json_entry = [dict(zip(columns, i)) for i in fetched]
    return {table : json_entry}

@app.get("/showTables")
def show_Tables():
    curs = mysql.connect.cursor()
    curs.execute('''SHOW TABLES''')
    fetched = curs.fetchall()
    curs.close()
    return {'Tables': fetched}

if __name__ == '__main__':  
    app.run(host='localhost', port=5000, debug=True) 

# flask --app app.py run