import os
from pathlib import Path

import sqlalchemy
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

root_dir = Path(__file__).resolve().parent
env_path = os.path.join(str(root_dir), "../../../.env")

load_dotenv(env_path)

MYSQL_ROOT_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_TCP_PORT = os.getenv("MYSQL_TCP_PORT")


def create_db(
    user=MYSQL_USER,
    password=MYSQL_ROOT_PASSWORD,
    host=MYSQL_HOST,
    port=MYSQL_TCP_PORT,
    database=MYSQL_DATABASE,
):
    SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}:{}/{}".format(
        user, password, host, port, database
    )
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine, SessionLocal, Base


engine, SessionLocal, Base = create_db()
