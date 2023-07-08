FROM python:3.7-slim
COPY  front.py .
COPY  predict.py . 
COPY  model model
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 80

RUN mkdir ~/.streamlit
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

WORKDIR /app

ENTRYPOINT ["streamlit", "run", "front.py"]
