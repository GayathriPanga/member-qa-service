## Member QA Service

This project is a simple question-answering API that infers answers about members based on their public message history.

It supports natural-language questions such as:

“When is Layla planning her trip to London?”

“How many cars does Vikram Desai have?”

“What are Amira’s favorite restaurants?”

The system fetches member messages from the provided public API, analyzes them, and generates an inferred answer.


## 📡 Provided API (Used by This Service)

Your service pulls data from the following public endpoint:

https://november7-730026606190.europe-west1.run.app/messages

Swagger documentation:

https://november7-730026606190.europe-west1.run.app/docs#/default/get_messages_messages__get

This endpoint returns member messages including user ID, name, text, and timestamp.


## 🚀 Live API (Google Cloud Run)

Your deployed service is publicly available at:

https://member-qa-service-438933417494.us-east4.run.app/docs

Use the /ask endpoint to test natural-language queries.


## 🛠 Tech Stack

Python

FastAPI

Docker

Google Cloud Run

Scikit-learn

NLTK

Dateparser


## 📌 Features

Detects which member the question refers to

Supports:

When questions

How many ownership questions

List-type questions

Handles contradictory statements using timestamps

Provides a safe fallback answer when inference is not possible


## 📝 Bonus 1: Design Notes (Summary)

Several approaches were considered for building this system:

Fine-tuned LLM

Vector embeddings + semantic search

Rule-based NLP + TF-IDF (chosen approach) — simple, deterministic, and explainable.


## 📊 Bonus 2: Data Insights

Insights observed during analysis of the dataset:

Some users mention numbers unrelated to ownership (e.g., durations).

Temporal expressions such as “next Monday” rely heavily on timestamps for correct interpretation.

Multiple conflicting statements exist for some users.

Message styles and formats vary significantly.


## 📦 Running Locally
pip install -r requirements.txt
uvicorn app:app --reload


Swagger UI will be available at:
http://localhost:8000/docs


## 📁 Repository

GitHub Repository:
https://github.com/GayathriPanga/member-qa-service
