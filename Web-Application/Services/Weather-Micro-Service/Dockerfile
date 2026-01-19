FROM python:3.13-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY weather_neo4j_service.py .

# Run the service
CMD ["python", "weather_neo4j_service.py"]