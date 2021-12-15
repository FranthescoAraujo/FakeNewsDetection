FROM tensorflow/tensorflow:latest-gpu

RUN python3 -m pip install --upgrade pip

WORKDIR /application

COPY requirements.txt .

RUN pip install -r requirements.txt