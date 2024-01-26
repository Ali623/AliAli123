# German Text Classification using Machine Learning

This repository contains the code for a machine learning model designed to classify German search queries into predefined categories. The model is trained on a dataset in a CSV format with two columns: "text" representing the search queries and "label" representing their associated classes.

## Project Structure

- `data`: This directory is not included in the repository, but it's assumed to contain the CSV file with the training data.

- `src`: Source code for the machine learning model and the REST API (FAST API).

- `Dockerfile`: Dockerfile for containerizing the model and REST API (FAST API).

- `requirements.txt`: List of Python dependencies.

## Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/ali623/AliAli123.git
cd AliAli123
```

## Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Model and REST API Locally

```bash
python train_model.py  # Train the model
uvicorn main:app --reload  # Run the FastAPI server locally
```
## Docker Deployment
Build and run the Docker container

```bash
docker build -t AliAli123 .
docker run -p 8000:8000 AliAli123
```


