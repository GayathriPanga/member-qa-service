FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for sklearn & nltk
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by PorterStemmer
RUN python3 -m nltk.downloader punkt

# Copy application files
COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
