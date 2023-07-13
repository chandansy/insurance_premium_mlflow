FROM python:3.8
WORKDIR /app
COPY . /app


RUN apt-get update && pip install -r requirements.txt

CMD ["python3","app.py"]