import sqlalchemy
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

def create_app(
    app=FastAPI(),
    origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:8000",
    ],
):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def create_db(user="root", password="msql1234", server="localhost", database="mock"):
    SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}/{}".format(
        user, password, server, database
    )
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine


app = create_app()
engine = create_db()


def getEntries(columns, table):
    db = engine.connect()
    col_string = ", ".join(columns)
    query_string = sqlalchemy.text(
        """SELECT {} FROM {} LIMIT 100""".format(col_string, table)
    )
    fetched = db.execute(query_string).fetchall()
    json_entry = [dict(zip(columns, i)) for i in fetched]
    return {table: json_entry}


@app.get("/health")
async def health():
    return {"message": "health ok"}


@app.get("/")
async def index():
    return {"message": "App Started"}


@app.get("/getTeams")
def getTeams(columns=["id", "name", "email", "age", "phone", "access"], table="users"):
    # Runs query "SELECT id, name, age, phone, email, access FROM users;"
    return getEntries(columns, table)


@app.get("/getContacts")
def getContacts(
    columns=[
        "id",
        "name",
        "email",
        "age",
        "phone",
        "address",
        "city",
        "zipCode",
        "registrarId",
    ],
    table="users",
):
    return getEntries(columns, table)


@app.get("/getInvoice")
def getInvoice(
    columns=["id", "name", "email", "phone", "cost", "date"], table="invoice"
):
    db = engine.connect()
    query_string = sqlalchemy.text(
        """
        SELECT invoice.invoice_id, users.name, users.phone, users.email, cost, date
        FROM users, invoice
        WHERE users.name = invoice.name
        AND users.email = invoice.email
        AND users.phone = invoice.phone
        ORDER BY invoice.invoice_id;
    """
    )
    fetched = db.execute(query_string).fetchall()
    json_entry = [dict(zip(columns, [str(j) for j in i])) for i in fetched]
    return {table: json_entry}
