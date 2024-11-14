# DSA3101-2410 Group 8 OpenAYEye
[![Python application test with Github Actions](https://github.com/ChinSekYi/OpenAYEye/actions/workflows/main.yml/badge.svg)](https://github.com/ChinSekYi/OpenAYEye/actions/workflows/main.yml)

## Project overview
In today’s competitive financial sector, banks face challenges in effectively targeting customers with personalized marketing efforts. This often results in low engagement rates, inefficient use of resources, and missed opportunities to connect with customers in meaningful ways. The core challenge is the lack of personalization due to underutilization of vast customer data, which often leaves marketing campaigns generic and misaligned with individual preferences.

Our project addresses this challenge by developing an AI-driven, personalized marketing system that leverages machine learning to analyze customer data in depth. By integrating customer insights, predictive analytics, and real-time optimization, our system aims to create highly tailored marketing campaigns that enhance customer engagement, improve conversion rates, and support data-driven decision-making for marketing teams.

<br>

## Instructions for setting up the environment and running the code
1. Clone the Repository
```
git clone https://github.com/ChinSekYi/OpenAYEye.git
cd OpenAYEye
```

... continued

<br>

## Description of the repository structure
This is the top 2 levels of our repository structure. Unimportant files and folders are omitted below. 
```
OpenAYEye/
├── docker-run.bash
├── requirements.txt
├── README.md
├── .dockerignore
├── docker-compose.yml
├── docker.env
├── engine.env
├── server/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── logs/
├── notebook/
├── data/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── client/
│   ├── README copy.md
│   ├── index.html
│   ├── Dockerfile
│   ├── vite.config.js
│   ├── README.md
│   ├── .gitignore
│   ├── package-lock.json
│   └── package.json
├── src
└── sql/
    ├── Dockerfile
    └── init.sql
```

### Usage

Jupyter notebooks for EDA and model building are in:
```
notebook/
├── segment/
├── roi-model/
├── reco_system/
└── src/
```

<br>

## Data sources and any necessary data preparation steps

<br>

## Instructions for building and running the Docker container(s)

<br>

## API documentation (endpoints, request/response formats)

<br>

## [Optional] API documentation using Swagger/OpenAPI specification

<br>

### Web App
- Run
```{bash}
bash docker-run.bash
```

- or 

```{bash}
docker compose up
```

Wait for the docker logs to stop runnning or until you see
```{docker}
data    | Engine(mysql+pymysql://root:***@db:3306/transact)
data    | users Ok
data    | transactions Ok
data    | churn Ok
data    | campaign Ok
data    | engagement Ok
```

- Webapp served on: [localhost:5173](http://localhost:5173)
