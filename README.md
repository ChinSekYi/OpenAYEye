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
├── docker-run.bash                # Bash script to run Docker
├── requirements.txt               # Python dependencies
├── Makefile                       # Automates tasks for project setup
├── test.py                        # General test script
├── README.md                      # Project README
├── docker-compose.yml             # Docker Compose file for container orchestration
├── server/                        # Server-side code and ML models
│   ├── requirements.txt           # Server dependencies
│   ├── Dockerfile                 # Docker setup for server
│   ├── README.md                  # Server README
│   ├── artifacts/                 # Pretrained ML models
│   ├── app/                       # Application files
├── logs/                          # Root level logs
├── notebook/                      # Jupyter notebooks for data processing and analysis
│   ├── segment/                   # Segment model notebooks and dependencies
│   ├── roi-model/                 # ROI model notebooks and data
│   └── reco_system/               # Recommendation system notebooks and data
├── data/                          # Data processing and database code
│   ├── requirements.txt           # Data dependencies
│   ├── Dockerfile                 # Docker setup for data service
│   ├── README.md                  # Data README
│   └── src/                       # Data source files
├── client/                        # Client-side code and setup
│   ├── index.html                 # Client HTML file
│   ├── Dockerfile                 # Docker setup for client
│   ├── README.md                  # Client README
│   ├── public/                    # Public assets
│   └── src/                       # Client source files
├── src/                           # Core source files
│   ├── pipeline/                  # Data pipeline components
│   ├── components/                # UI components (if applicable)
├── sql/                           # SQL setup and initialization scripts
    ├── Dockerfile                 # Docker setup for SQL
    └── init.sql                   # SQL initialization script
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

### Production-ready Python code
**Main Python File**
```
server/src/main.py  #The main entry point for the server-side application.

notebook/segment/main.py #Script for FastAPI app with endpoints for data queries and ML predictions.
notebook/src/main.py  #Script for running data pipelines for recommendation system and ROI model. Refer to pipeline Readme.md for more info.
```

<br>

## Data sources and any necessary data preparation steps
For data sources, refer to the Wiki [3. Overall Data Understanding Database Schema](https://github.com/ChinSekYi/OpenAYEye/wiki/3.-Overall-Data-Understanding-Database-Schema).

For data preparation steps, refer to the wiki for relevant models such as RFM, Customer Behaviour EDA, Recommendation System and ROI model.  
Link to wiki: [OpenAYEye Wiki](https://github.com/ChinSekYi/OpenAYEye/wiki)

For detailed coding steps for data preparation, please refer to the relevant Jupyter notebooks, as they include more detailed explanations and methods. 

To access these notebooks, go to:
```
notebook/
├── segment/
├── roi-model/
├── reco_system/
└── src/
```


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
