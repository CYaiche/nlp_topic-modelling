FROM python:3.7-slim
RUN mkdir ~/app
WORKDIR /app
COPY  front.py .
COPY  predict.py . 
COPY  model model
COPY  requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 80

RUN mkdir ~/.streamlit
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

CMD ["streamlit", "run", "front.py"] 
