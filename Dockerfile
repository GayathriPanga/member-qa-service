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

# Copy application files
COPY . .

# Cloud Run injects PORT environment variable
ENV PORT=8080

# Expose port (not required but good practice)
EXPOSE 8080

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
