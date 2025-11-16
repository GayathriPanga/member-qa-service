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

While building this question-answering system, I explored several possible approaches and refined the design based on what worked best for the dataset.

# 1. Rule-Based Question Classification  
I first designed the system to detect the type of question:
- **“When” questions** → require temporal parsing  
- **“How many” questions** → require ownership + numeric inference  
- **List / “What are” questions** → require finding plural entities  

This rule-based classification kept the logic simple, predictable, and easy to debug.

# 2. Fetching Messages with Pagination  
The `/messages` endpoint provides paginated data, so I implemented:
- Automatic pagination (`skip` + `limit`)
- Retry logic for occasional 404s
- Caps to avoid over-fetching

This ensured robustness even with thousands of messages.

# 3. User Detection Using Token Overlap  
To identify which member the question is about, I used:
- Token normalization  
- Stemming  
- Overlap scoring between the question and user names  

This avoided ambiguity when multiple users appear in the dataset.

# 4. Temporal Resolution for “When” Questions  
We used:
- `dateparser.search_dates`  
- The message timestamp as the reference point  
- TF-IDF + keyword scoring to choose the most relevant message  

This allowed the system to interpret phrases like “next Friday” correctly based on when the member sent the message.

### 5. Ownership & Count Logic  
For “How many” questions, we tested:
- Detecting ownership patterns (“my”, “I have”, “I bought”, etc.)
- Extracting numeric values  
- Handling contradictions by using the most recent message  
- Auto-learning synonyms from the user’s own message patterns  

This made the count inference more accurate.

### 6. List-Style Question Extraction  
We added logic to:
- Detect plural nouns from the question  
- Pull all messages mentioning those plural entities  
- Return a simple list of relevant statements  

### 7. Deployment Approaches Explored  
We tried multiple deployment paths:
- **Render** (initial attempt)
- **Final deployment** via **Google Cloud Run** with a Dockerfile

Google Cloud Run was chosen because it supports container-based deployment, scales automatically, and integrates cleanly with GitHub.

---

These approaches were selected to keep the system simple but reliable, while handling natural-language variability in the member messages.


## 📊 Bonus 2: Data Insights

While exploring the `/messages` API, I noticed a few patterns in the dataset:

1. **Members have many messages across different categories**, such as travel plans, hotel requests, restaurant bookings, billing issues, and profile updates. Each member is identified consistently by a `user_id` and `user_name`.

2. **Messages contain a lot of natural-language variability**—including dates (“next Friday”, “tomorrow”), numbers (party sizes, dates, phone numbers), and personal preferences. This required careful handling so my system doesn’t misinterpret phone numbers or dates as counts.

3. **Temporal expressions are often relative**, so for “when” questions I needed to resolve phrases like “next Monday” using the message’s timestamp.

4. **Ownership is not always explicit**, so for “how many” questions I only infer counts when a message clearly contains ownership words like “my”, “I have”, or “I own”.

5. **Plural entities appear frequently**, which I use to answer list-style questions by collecting relevant statements directly from the member’s message history.

These patterns shaped my QA logic and helped ensure that the system answers accurately and safely when information is available, and responds with a fallback message when it isn’t.



## 📦 Running Locally
`pip install -r requirements.txt`

`uvicorn app:app --reload`


Swagger UI will be available at:
`http://localhost:8000/docs`


## 📁 Repository

GitHub Repository:
https://github.com/GayathriPanga/member-qa-service
