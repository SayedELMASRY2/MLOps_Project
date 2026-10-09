.PHONY: help install train serve mlflow-ui test docker-build docker-run clean

PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

help:
	@echo "Available commands:"
	@echo "  make install       Install all project dependencies"
	@echo "  make train         Run MLflow training pipeline"
	@echo "  make test          Run full pytest test suite"
	@echo "  make serve         Start FastAPI serving server"
	@echo "  make mlflow-ui     Start MLflow Tracking UI"
	@echo "  make docker-build  Build Docker production container"
	@echo "  make docker-run    Run Docker container on port 8000"
	@echo "  make clean         Clean cache and temporary files"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

train:
	$(BIN)/python train.py

test:
	$(BIN)/pytest -v tests/

serve:
	$(BIN)/uvicorn app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	$(BIN)/mlflow ui --host 0.0.0.0 --port 5000

docker-build:
	docker build -t california-housing-api:v1 .

docker-run:
	docker run -d --name housing-api -p 8000:8000 california-housing-api:v1

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
