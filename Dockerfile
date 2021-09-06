FROM ubuntu:16.04

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y lsb-release && apt-get clean all && \
    apt-get install -y python3 && apt-get install -y python-pip

WORKDIR /usr/src/app

COPY requirements.txt ../

#RUN pip install --upgrade pip && \

RUN pip install --upgrade pip
    
#RUN pip install --no-cache-dir -r requirements.txt

COPY . .