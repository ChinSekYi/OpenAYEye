FROM python:3.9

WORKDIR /code
COPY ./app /code/app
COPY ./data code/data
COPY ./data code/artifacts

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# RUN python3 -m app.src.components.roi.ROI_data_pipeline
# RUN python3 -m app.src.components.roi.ROI_model_trainer
# RUN python3 -m app.src.pipeline.roi.ROI_predict_pipeline

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]

# docker build -t server .
# docker run -d --name servercontainer -p 8000:8000 server