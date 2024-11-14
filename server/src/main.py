# %%
import os
import warnings
from json import dumps, loads

import numpy as np
import pandas as pd
from dataset import RFM, ROI, Churn, Engagement, Reco, RFM_churn, RFM_engage, engine
from dateutil.relativedelta import relativedelta
from train import explained_dct, reco_df, roi_est

warnings.filterwarnings("ignore")

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

# print(explained_dct)


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
    query_string = sqlalchemy.text("""SELECT {} FROM {};""".format(col_string, table))
    fetched = db.execute(query_string).fetchall()
    json_entry = [dict(zip(columns, i)) for i in fetched]
    return fetched


@app.get("/health")
async def health():
    print(engine)
    return {"message": "health ok"}


@app.get("/")
async def index():
    return {"message": "App Started"}


@app.get("/totalTraffic")
async def getTraffic():
    with engine.connect() as db:
        query = sqlalchemy.text(
            """
            SELECT COUNT(*) FROM engagement;
            """
        )
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {"status": "ok", "data": "{:,}".format(data)}


@app.get("/convertedClients")
async def getConvertedClients():
    with engine.connect() as db:
        query = sqlalchemy.text(
            """
            SELECT COUNT(DISTINCT e.customer_id)
            FROM engagement e
            WHERE e.action_type = 'converted'
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);
            """
        )
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {"status": "ok", "data": "{:,}".format(data)}


@app.get("/potentialCustomers")
async def getConvertedClients():
    with engine.connect() as db:
        query = sqlalchemy.text(
            """
            SELECT COUNT(DISTINCT e.customer_id)
            FROM engagement e
            WHERE e.action_type IN ('credentials', 'clicked')
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);
            """
        )
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {"status": "ok", "data": "{:,}".format(data)}


@app.get("/conversionRate")
async def getConvertedClients():
    with engine.connect() as db:
        query = sqlalchemy.text(
            """
            SELECT t1.converted / t2.impressions FROM 
            (SELECT COUNT(DISTINCT e.customer_id) AS converted
            FROM engagement e
            WHERE e.action_type = 'converted'
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS t1,
            (SELECT COUNT(DISTINCT e.customer_id) AS impressions
            FROM engagement e
            WHERE e.action_type IN ('converted', 'credentials', 'clicked')
            AND e.engagement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS t2;
            """
        )
        df = db.execute(query).fetchone()
        # print(df)
        db.close()
    data = df[0]
    return {"status": "ok", "data": "{:,}".format(data)}


@app.get("/campaignReach")
async def getReach():
    with engine.connect() as db:
        query = sqlalchemy.text(
            """
            SELECT * FROM engagement;
            """
        )
        df = pd.DataFrame(db.execute(query).fetchall())
        db.close()
    data = (
        df[
            df["engagement_date"]
            >= df["engagement_date"].max() - relativedelta(months=11)
        ]
        .groupby([df["engagement_date"].dt.to_period("M"), "action_type"])
        .agg(["count"])["customer_id"]
        .reset_index()
        .pivot(index="engagement_date", columns="action_type", values="count")
        .reset_index()
        .loc[:, ["engagement_date", "converted", "credentials", "clicked", "scrolled"]]
        .rename(columns={"engagement_date": "date"})
        .astype({"date": str})
    )

    data["convertedColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data["credentialsColor"] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data["clickedColor"] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data["scrolledColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]

    data = data.to_dict(orient="records")

    return {"status": "ok", "data": data}


@app.get("/latestEngage")
async def getLatest():
    with engine.connect() as db:
        query = sqlalchemy.text(
            """
            SELECT *
            FROM engagement e
            ORDER BY e.engagement_date DESC
            LIMIT 10;
            """
        )
        df = pd.DataFrame(db.execute(query).fetchall())
        db.close()

    df["engagement_date"] = df["engagement_date"].astype(str)
    df = df.loc[:, ["customer_id", "engagement_date", "action_type", "feedback_score"]]
    df = df.rename(
        columns={
            "customer_id": "id",
            "engagement_date": "date",
            "action_type": "action",
            "feedback_score": "score",
        }
    )
    data = df.to_dict(orient="records")

    return {"status": "ok", "data": data}


@app.get("/adSpend")
async def getReach():
    with engine.connect() as db:
        query = sqlalchemy.text(
            """
            SELECT * 
            FROM campaign c;
            """
        )
        df = pd.DataFrame(db.execute(query).fetchall())
        db.close()
    data = (
        df[df["start_date"] >= df["start_date"].max() - relativedelta(months=11)]
        .groupby([df["start_date"].dt.to_period("M")])
        .agg({"budget": ["sum"]})["budget"]
        .reset_index()
        .rename(columns={"start_date": "date", "sum": "spending"})
    )
    data["date"] = data["date"].astype(str)
    data["spending"] = data["spending"].astype(int)
    data["spendingColor"] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient="records")

    return {"status": "ok", "data": data}


@app.get("/predROI")
async def getROI():
    roi = ROI(engine)
    X = roi.get_X()
    pred = roi_est.predict(X)

    data = roi.df
    data[["clicks", "leads", "orders"]] = pred
    data[["clicks", "leads", "orders"]] = data[["clicks", "leads", "orders"]].astype(
        int
    )
    # data['c_date'] = data['c_date'].dt.to_period("M").astype(str)
    data = (
        data[data["c_date"] >= data["c_date"].max() - relativedelta(months=11)]
        .groupby([data["c_date"].dt.to_period("M")])
        .agg({"clicks": ["sum"], "leads": ["sum"], "orders": ["sum"]})
        .reset_index()
    )
    data.columns = data.columns.get_level_values(0)
    data["c_date"] = data["c_date"].astype(str)
    data = data.rename(columns={"c_date": "Campaign Month"})

    data["clicksColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data["leadsColor"] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data["ordersColor"] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient="records")
    return {"status": "ok", "data": data}


@app.get("/segByIncome")
async def getReach():
    rfm = RFM(engine)
    df = rfm.df
    rfm_seg = rfm.get_RFM()
    data = df.merge(rfm_seg, how="left", on="customer_id")
    data["income_cat"] = pd.qcut(
        df["yearly_income"].astype(np.float64),
        [0, 0.5, 0.75, 0.90, 1.0],
        labels=["Very Low", "Low", "Middle", "High"],
    )
    data = (
        data.groupby(["income_cat", "segment"])
        .agg("count")["customer_id"]
        .reset_index()
        .rename(columns={"customer_id": "count"})
        .pivot(index="income_cat", columns="segment", values="count")
        .reset_index()
    )

    data["At RiskColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data["HibernatingColor"] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data["Loyal CustomersColor"] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data["New CustomersColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient="records")

    return {"status": "ok", "data": data}


@app.get("/segByAge")
async def getReach():
    rfm = RFM(engine)
    df = rfm.df
    rfm_data = rfm.get_RFM()
    data = df.merge(rfm_data, how="left", on="customer_id")
    # data
    data["age_cat"] = pd.qcut(
        data["age"].astype(np.float64),
        [0.25, 0.5, 0.75, 1.0],
        labels=["Young", "Middle", "Elderly"],
    )
    data = (
        data.groupby(["age_cat", "segment"])
        .agg("count")["customer_id"]
        .reset_index()
        .rename(columns={"customer_id": "count"})
        .pivot(index="age_cat", columns="segment", values="count")
        .reset_index()
    )

    data["At RiskColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data["HibernatingColor"] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data["Loyal CustomersColor"] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data["New CustomersColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient="records")

    return {"status": "ok", "data": data}


@app.get("/churnBySeg")
async def getReach():
    churn = Churn(engine)
    rfm = RFM(engine)

    churn = churn.preprocess()[["customer_id", "churn"]]
    data = churn.merge(rfm.get_RFM(), how="left", on="customer_id")
    data = (
        data.groupby(["segment", "churn"])
        .agg("count")["customer_id"]
        .reset_index()
        .rename(columns={"customer_id": "count"})
        .pivot(index="segment", columns="churn", values="count")
        .reset_index()
        .rename(columns={"segment": "Segment", 0: "Remain", 1: "Exited"})
    )
    data["noColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data["yesColor"] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient="records")

    return {"status": "ok", "data": data}


@app.get("/recoBySeg")
async def getReco():
    rfm = RFM(engine)
    rfm = rfm.get_RFM()[["customer_id", "segment"]]

    data = (
        reco_df.merge(rfm, how="left", on="customer_id")
        .drop(["customer_id", "deposits", "cards", "account", "loan"], axis=1)
        .loc[:, ["segment", "deposits_reco", "cards_reco", "account_reco", "loan_reco"]]
        .groupby(["segment"])
        .agg("mean")
        .reset_index()
    )
    data["deposits_recoColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data["cards_recoColor"] = ["hsl(296, 70%, 50%)" for i in range(len(data))]
    data["account_recoColor"] = ["hsl(97, 70%, 50%)" for i in range(len(data))]
    data["loan_recoColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    data = data.to_dict(orient="records")

    return {"status": "ok", "data": data}

def get_relation(explained_dct, est = 'engage New Customers', X_col="budget",
        y_col="action_type", y_val="converted"
    ):
    data = explained_dct[est].get_shap(X_col, y_col, y_val)
    data = pd.concat([data.iloc[:, -1], data.iloc[:, 0]], axis=1).sort_values(["{}__{}".format(X_col, y_val)])
    data['shap'] = data['shap'].apply(lambda x:x*10000)
    data = data.rename(columns={"shap": "y", "{}__{}".format(X_col, y_val): "x"}) \
        .to_dict(orient='records')
    data = [{'id':X_col.capitalize(), "data": data}]
    return data

def get_relation_cat(explained_dct, est = 'engage New Customers', X_col="goal",
        y_col="action_type", y_val="converted"
    ):
    data = explained_dct[est].get_shap(X_col, y_col, y_val)
    data = pd.concat([data.iloc[:, -1], data.iloc[:, 0]], axis=1).sort_values(["{}__{}".format(X_col, y_val)])
    data['shap'] = data['shap'].apply(lambda x:x*10000)
    data = data.groupby(["{}__{}".format(X_col, y_val)]) \
        .agg(["mean"])['shap'].reset_index() \
        .rename(columns={"shap": "Weight", "{}__{}".format(X_col, y_val): X_col})
    data["meanColor"] = ["hsl(229, 70%, 50%)" for i in range(len(data))]
    # data['mean'] = data['mean'].astype(int)
    data = data.to_dict(orient="records")
    # data = [{'id':'1', "data": data}]
    return data

@app.get("/relateBudget")
async def getRelateBudget():
    data = get_relation(explained_dct, X_col="budget", y_col="action_type")
    return {"status": "ok", "data": data}

@app.get("/relateGoal")
async def getRelateBudget():
    data = get_relation_cat(explained_dct, X_col="goal", y_col="action_type")
    return {"status": "ok", "data": data}

@app.get("/relateChannel")
async def getRelateBudget():
    data = get_relation_cat(explained_dct, X_col="channel", y_col="action_type")
    return {"status": "ok", "data": data}

@app.get("/relateEngage")
async def getRelateBudget():
    data = get_relation(explained_dct, X_col="engage_month", y_col="action_type")
    return {"status": "ok", "data": data}
    engage_month