# %%
import numpy as np
import pandas as pd
from json import loads, dumps

from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore")

from dataset import engine

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

# @app.get("/getChurn")
# def getTeams(columns=["customerid", "surname", "creditscore", "age", "geography", "gender", "tenure " "exited"], table="churn"):
#     # Runs query "SELECT id, name, age, phone, email, access FROM users;"
#     fetched = getEntries(columns, table)
#     columns = ['id'] + columns[1:]
#     json_entry = [dict(zip(columns, i)) for i in fetched]
#     return json_entry

@app.get("/totalTraffic")
async def getTraffic():
    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT COUNT(*) FROM engagement;
            ''')
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {'status': 'ok', 'data': "{:,}".format(data)}

@app.get("/convertedClients")
async def getConvertedClients():
    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT COUNT(DISTINCT e.customer_id)
            FROM engagement e
            WHERE e.action_type = 'converted';
            ''')
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {'status': 'ok', 'data': "{:,}".format(data)}
    
    

@app.get("/campaignReach")
async def getReach():
    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT * FROM engagement;
            ''')
        df = pd.DataFrame(db.execute(query).fetchall())
        db.close()
    data = df[df['engagement_date'] >= df['engagement_date'].max() - relativedelta(months=11)] \
        .groupby([df['engagement_date'].dt.to_period("M"), 'action_type']) \
        .agg(['count'])['customer_id'] \
        .reset_index() \
        .pivot(index="engagement_date", columns="action_type", values="count") \
        .reset_index() \
        .loc[:, ['engagement_date', 'converted', 'credentials', 'clicked', 'scrolled']] \
        .rename(columns={'engagement_date': 'date'}) \
        .astype({'date': str})

    data['convertedColor'] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data['credentialsColor'] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data['clickedColor'] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data['scrolledColor'] = ["hsl(229, 70%, 50%)" for i in range(len(data))]


    data = data.to_dict(orient='records')

    return {'status': 'ok', 'data': data}

