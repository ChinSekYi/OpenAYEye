# %%
import re
import time
import datetime

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from dataset import engine, Churn, Engagement, RFM
from models import MyRFClassifier, MyRFRegressor
from configs import Config
from myutils import show_cm, show_feature_importance


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
        "http://localhost:8000",
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

app = create_app()

def getEntries(columns, table):
    db = engine.connect()
    col_string = ", ".join(columns)
    query_string = sqlalchemy.text(
        """SELECT {} FROM {};""".format(col_string, table)
    )
    fetched = db.execute(query_string).fetchall()
    json_entry = [dict(zip(columns, i)) for i in fetched]
    return fetched

@app.get("/health")
async def health():
    return {"message": "health ok"}


@app.get("/")
async def index():
    return {"message": "App Started"}

@app.get("/getChurn")
def getTeams(columns=["customerid", "surname", "creditscore", "age", "geography", "gender", "tenure " "exited"], table="churn"):
    # Runs query "SELECT id, name, age, phone, email, access FROM users;"
    fetched = getEntries(columns, table)
    columns = ['id'] + columns[1:]
    json_entry = [dict(zip(columns, i)) for i in fetched]
    return json_entry

