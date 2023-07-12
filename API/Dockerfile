FROM python:3.8-slim


WORKDIR /app

COPY  front.py .
COPY  predict.py . 
COPY  model model
COPY  requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

RUN mkdir ~/.streamlit
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

CMD ["streamlit", "run", "front.py", "--server.port=8501", "--server.address=0.0.0.0"] 
