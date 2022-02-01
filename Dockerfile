FROM python:3.8.12

RUN python3 -m pip install --upgrade pip

WORKDIR /FakeNewsDetection

COPY requirements.txt .

RUN pip install -r requirements.txt