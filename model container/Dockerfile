FROM python:3.10

WORKDIR /notogo-model

COPY requirements.txt .

RUN apt-get update && apt-get -y upgrade 
RUN apt-get install -yq python

RUN pip install -r requirements.txt

COPY ./app/. ./
RUN ls -la ./*

# CMD ['python','./app/deploy_test.py']

ENTRYPOINT ["python","./deploy_test.py"]

# G:\Other computers\My Laptop\Bangkit\Capstone\Recommender system\Model Container\app\deploy_test.py
#docker build -t notogo .
#docker run notogo