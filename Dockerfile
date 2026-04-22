# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY static/ ./static/

# Expose port 7680 for Flask
EXPOSE 7680

# Set environment variables
ENV FLASK_APP=app/app_flask.py
ENV FLASK_ENV=production

# Run Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
