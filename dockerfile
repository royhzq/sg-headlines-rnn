FROM python:3.6-stretch
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-pip nginx
COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD ["gunicorn", "--bind",  ":5001", "wsgi:app"]