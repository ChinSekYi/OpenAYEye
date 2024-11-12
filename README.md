# DSA3101-2410 Group 8 OpenAYEye
[![Python application test with Github Actions](https://github.com/ChinSekYi/OpenAYEye/actions/workflows/main.yml/badge.svg)](https://github.com/ChinSekYi/OpenAYEye/actions/workflows/main.yml)

## Project overview

## Instructions for setting up the environment and running the code

## Description of the repository structure

## Data sources and any necessary data preparation steps

## Instructions for building and running the Docker container(s)

## API documentation (endpoints, request/response formats)

## [Optional] API documentation using Swagger/OpenAPI specification

# Web App
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