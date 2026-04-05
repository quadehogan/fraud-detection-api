FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and pre-trained model
COPY . .

# Railway sets $PORT at runtime — app MUST listen on it
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
