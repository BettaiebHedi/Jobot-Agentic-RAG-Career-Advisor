FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Chainlit will be launched via docker-compose command
