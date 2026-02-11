FROM python:3.11-slim

WORKDIR /app

# Install only serving dependencies (CPU-only torch for smaller image)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application code
COPY config/ config/
COPY src/ src/
COPY webapp/ webapp/

# Create directories
RUN mkdir -p checkpoints webapp/uploads

EXPOSE 10000

CMD ["gunicorn", "webapp.app:create_app()", "--bind", "0.0.0.0:10000", "--timeout", "120", "--workers", "1"]
