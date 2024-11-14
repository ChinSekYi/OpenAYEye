import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def kfold_KNNReg(X, y):
    kf = KFold(n_splits=5)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor(algorithm="kd_tree")),
        ]
    )
    best = None
    best_score = 1e99
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # print(train_index)
        X_train = X.iloc[train_index, :]
        X_test = X.iloc[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        if score < best_score:
            best = pipeline
    return best


random.seed(10)
# Define helper functions and parameters
num_campaigns = 300  # Set the desired number of campaigns


def seasonal_campaign_date():
    month_weights = {
        1: 1.2,
        2: 1.0,
        3: 0.8,
        4: 0.7,
        5: 0.7,
        6: 0.6,
        7: 0.5,
        8: 0.5,
        9: 0.7,
        10: 0.9,
        11: 1.3,
        12: 1.4,
    }
    month = random.choices(range(1, 13), weights=month_weights.values(), k=1)[0]
    start_day = random.randint(1, 15)
    start_date = datetime(random.choices([2023, 2024], k=1)[0], month, start_day)
    end_date = start_date + timedelta(days=random.randint(7, 30))  # Campaign length
    return start_date, end_date


# Seasonal budgets and goals
def seasonal_budget(month):
    if month in [1, 6, 7, 11, 12]:  # Peak months
        return round(random.uniform(100, 600), 2)
    return round(random.uniform(50, 200), 2)


campaign_goals = ["awareness", "consideration", "conversion", "retention"]
campaign_channels = ["social", "search", "influencer", "jkl"]  # "app","email", "sms",


# Generate data
data = {
    "campaign_id": [f"{str(i).zfill(4)}" for i in range(1, num_campaigns + 1)],
    "campaign_name": [f"Campaign_{i}" for i in range(1, num_campaigns + 1)],
    "start_date": [],
    "end_date": [],
    "target_segment": [
        random.choice(["High-income", "Young Adults", "Families", "Retirees"])
        for _ in range(num_campaigns)
    ],
    "budget": [],
    "channel": [random.choice(campaign_channels) for _ in range(num_campaigns)],
    "goal": [random.choice(campaign_goals) for _ in range(num_campaigns)],
    "displays": [],
}

for i in range(num_campaigns):
    start, end = seasonal_campaign_date()
    data["start_date"].append(start)
    data["end_date"].append(end)
    data["budget"].append(seasonal_budget(start.month))

df = pd.read_csv("data/ad_performance.csv")[["displays", "cost"]]
X = df[["cost"]]
y = df["displays"]

best = kfold_KNNReg(X, y)

data["displays"] = best.predict(np.array(data["budget"]).reshape(-1, 1)).astype(int)
campaign = pd.DataFrame(data)
# print(campaign)
# campaign.to_csv("campaign.csv", index=False)
