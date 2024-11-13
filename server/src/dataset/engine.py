# import pandas as pd
# import numpy as np

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


def create_db(user="root", password="msql1234", server="db", port="3306", database="transact"):
    SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}/{}".format(
        user, password, server, database
    )
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine, SessionLocal, Base

engine, SessionLocal, Base = create_db(server="db")