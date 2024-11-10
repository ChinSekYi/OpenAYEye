# %%
import os
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
            WHERE e.action_type = 'converted'
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);
            ''')
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {'status': 'ok', 'data': "{:,}".format(data)}


@app.get("/potentialCustomers")
async def getConvertedClients():
    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT COUNT(DISTINCT e.customer_id)
            FROM engagement e
            WHERE e.action_type IN ('credentials', 'clicked')
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);
            ''')
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {'status': 'ok', 'data': "{:,}".format(data)}


@app.get("/conversionRate")
async def getConvertedClients():
    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT t1.converted / t2.impressions FROM 
            (SELECT COUNT(DISTINCT e.customer_id) AS converted
            FROM engagement e
            WHERE e.action_type = 'converted'
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS t1,
            (SELECT COUNT(DISTINCT e.customer_id) AS impressions
            FROM engagement e
            WHERE e.action_type IN ('converted', 'credentials', 'clicked')
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS t2;
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

@app.get("/latestEngage")
async def getLatest():
    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT *
            FROM engagement e
            ORDER BY e.engagement_date DESC
            LIMIT 10;
            ''')
        df = pd.DataFrame(db.execute(query).fetchall())
        db.close()

    df['engagement_date'] = df['engagement_date'].astype(str)
    df = df.loc[:, ['customer_id', 'engagement_date', 'action_type', 'feedback_score']]
    df = df.rename(columns={'customer_id': 'id', 'engagement_date': 'date', 'action_type': 'action', 'feedback_score': 'score'})
    data = df.to_dict(orient='records')

    return {'status': 'ok', 'data': data}


@app.get("/adSpend")
async def getReach():
    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT * 
            FROM campaign c;
            ''')
        df = pd.DataFrame(db.execute(query).fetchall())
        db.close()
    data = df[df['start_date'] >= df['start_date'].max() - relativedelta(months=11)] \
        .groupby([df['start_date'].dt.to_period("M")]) \
        .agg({"budget": ['sum']})['budget'].reset_index().rename(columns={'start_date':'date', 'sum':'spending'})
    data['date'] = data['date'].astype(str)
    data['spending'] = data['spending'].astype(int)
    data['spendingColor'] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient='records')

    return {'status': 'ok', 'data': data}


@app.get("/predROI")
async def getROI():
    from collections import OrderedDict
    from src.exception import CustomException
    from src.utils import load_object
    import joblib

    def get_CLO(features):
        model_path = os.path.join("artifacts", "roi_trained_model.pkl")
        encoder_path = os.path.join("artifacts", "roi_onehot_encoder.pkl")

        model = load_object(file_path=model_path)
        encoder = joblib.load(encoder_path) # Load pre-fitted OneHotEncoder
        # Apply the pre-fitted encoder to the new data
        X_encoded = encoder.transform(features[['category']])  # Use transform, not fit_transform

        # Get the expected encoded feature names from the encoder
        encoded_columns = encoder.get_feature_names_out(['category'])

        # Concatenate the encoded category columns with the 'cost' column
        X_transformed = np.concatenate([X_encoded, features[['cost']].values], axis=1)

        pred = model.predict(X_transformed)
        return pd.DataFrame(pred, columns=['clicks', 'leads', 'orders'])

    with engine.connect() as db:
        query = sqlalchemy.text(
            '''
            SELECT * FROM campaign;
            ''')
        df = pd.DataFrame(db.execute(query).fetchall())
        db.close()

    features = df.rename(columns={'channel': 'category', 'budget':'cost'}).loc[:, ['category', 'cost']]
    features

    CLO = get_CLO(features)

    df = pd.concat([df, CLO], axis=1)
    df = df[df['start_date'] >= df['start_date'].max() - relativedelta(months=11)] \
        .groupby([df['start_date'].dt.to_period("M")]) \
        .agg({"clicks": ['sum'], "leads": ['sum'], "orders": ['sum']})[['clicks', 'leads', 'orders']].reset_index()

    df.columns = df.columns.get_level_values(0)
    df['start_date'] = df['start_date'].astype(str)
    data = df
    data['clicksColor'] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data['leadsColor'] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data['ordersColor'] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient='records')
    return {'status': 'ok', 'data': data}