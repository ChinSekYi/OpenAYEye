# DSA3101-2410 Group 8 OpenAYEye
[![Python application test with Github Actions](https://github.com/ChinSekYi/OpenAYEye/actions/workflows/main.yml/badge.svg)](https://github.com/ChinSekYi/OpenAYEye/actions/workflows/main.yml)

# Table of Contents
1. [Project Overview](#project-overview)
2. [Instructions for Setting Up the Environment and Running the Code](#instructions-for-setting-up-the-environment-and-running-the-code)
3. [Description of the Repository Structure](#description-of-the-repository-structure)
    - [Jupyter Notebooks for EDA and Model Development](#jupyter-notebooks-for-eda-and-model-development)
    - [Production-Ready Python Code](#production-ready-python-code)
6. [Data Sources and Data Preparation Steps](#data-sources-and-data-preparation-steps)
7. [Instructions for Building and Running the Docker Containers](#instructions-for-building-and-running-the-docker-containers)
8. [API Documentation (Endpoints, Request/Response Formats)](#api-documentation-endpoints-requestresponse-formats)
9. [API Documentation Using Swagger/OpenAPI Specification](#api-documentation-using-swaggeropenapi-specification)
    - [Web App](#web-app)
10. [Data Pipelining for MLOps](#data-pipelining-for-mlops)
    - [Recommendation System Pipeline](#recommendation-system-pipeline)
    - [ROI Model Pipeline](#roi-model-pipeline)

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
├── test.py                        # Test script for CI/CD Github Actions
├── README.md                      # Project README
├── docker-compose.yml             # Docker Compose file for container orchestration
├── server/                        # Server-side code and ML models
│   ├── requirements.txt           # Server dependencies
│   ├── Dockerfile                 # Docker setup for server
│   ├── README.md                  # Server README
│   ├── artifacts/                 # Pretrained ML models
│   ├── app/                       # Application files
├── notebook/                      # Jupyter notebooks for EDA, model training etc
│   ├── segment/                   # Segment model notebooks
│   ├── roi-model/                 # ROI model notebooks 
│   └── reco_system/               # Recommendation system notebooks 
    └── src/                           # Data pipelining
        ├── exception.py               # Custom exception handling classes
        ├── pipeline/                  # Pipeline for prediction
        ├── components/                # Pipeline for data ingestion and model training
        ├── logger.py                  # Logging utility 
        ├── utils.py                   # Utility functions
        ├── main.py                    # To run Pipeline
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
├── sql/                           # SQL setup and initialization scripts
    ├── Dockerfile                 # Docker setup for SQL
    └── init.sql                   # SQL initialization script
```

<br>


### Jupyter notebooks for EDA and Model development
Can be found under `notebook/`  
Subgroup A worked on: `segment`  
Subgroup B worked on: `roi-model` and `reco_system`  
```
notebook/
├── segment/                 
├── roi-model/    
    └── roi-model-with-wiki.ipynb        # EDA & Model development for ROI model
├── reco_system/  
    └── logistic_regression              # Model development
    └── New_data_creation_covid.ipynb    # EDA & data manipulation Notebook for Covid event
    └── New_data_creation_meteor.ipynb   # EDA & data manipulation Notebook for Meteor event 
    └── santender_cleaning.ipynb         # EDA Notebook 
```

<br>

### Production-ready Python code
**Main Python File**
```
server/src/main.py          # The main entry point for the server-side application.
notebook/segment/main.py    # Script for FastAPI app with endpoints for data queries and ML predictions.
notebook/src/main.py        # Script for running data pipelines for recommendation system and ROI model. 
                            # Refer to "Data Pipelining for MLOps Overview" section at the end of this README.md
```

Other relevant files such as `utils.py`, `logger.py` etc can also be found in the repository.

<br>

## Data sources and data preparation steps
- For data sources, refer to the Wiki [3. Overall Data Understanding Database Schema](https://github.com/ChinSekYi/OpenAYEye/wiki/3.-Overall-Data-Understanding-Database-Schema).

- For data preparation steps, refer to the wiki for relevant models such as RFM, Customer Behaviour EDA, Recommendation System and ROI model.  
Link to wiki: [OpenAYEye Wiki](https://github.com/ChinSekYi/OpenAYEye/wiki)

- For detailed coding steps for data preparation, please refer to the relevant Jupyter notebooks, as they include more detailed explanations and methods. Refer to "Jupyter notebooks for EDA and Model development" sectiion above.
 

<br>

## Instructions for building and running the Docker container(s)

<br>

## API documentation (endpoints, request/response formats)

<br>

## API documentation using Swagger/OpenAPI specification

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


<br>



### Data Pipelining for MLOps

This repository includes two main data pipeline systems: the **Recommendation System (Reco)** and the **ROI Model**. These pipelines automate the steps of data ingestion, transformation, model training, and prediction, crucial for machine learning workflows in business applications such as customer recommendation and return on investment (ROI) analysis.

One example:
<p align="center">
    <img src="notebook/src/image/ROI pipeline flowchart.png" alt="ROI Pipeline flowchart" width="180"/>
</p>


#### **Recommendation System Pipeline**
- **Purpose**: To ingest, clean, transform, and generate personalized recommendations based on historical data.
- **Pipelines**:
  - Data ingestion and transformation pipeline.
  - Model training pipeline to generate insights.
  - Prediction pipeline for generating ranked recommendations.
- **Location**: `/src/components/reco_sys/` for the components, `/src/pipeline/reco_sys/` for the prediction pipeline.

#### **ROI Model Pipeline**
- **Purpose**: To process and analyze data to predict the ROI for marketing campaigns, based on features such as cost and category.
- **Pipelines**:
  - Data ingestion and transformation pipeline.
  - Model training pipeline for predicting ROI-related metrics.
  - Prediction pipeline for new input data.
- **Location**: `/src/components/roi-model/` for the components, `/src/pipeline/roi-model/` for the prediction pipeline.

<br>

For detailed instructions on running each pipeline, please refer to `src/pipeline README.md`.
