# Use official Python image
FROM python:3.10

# Set working directory inside the app folder
WORKDIR /app

# Copy only the requirements file first to cache dependencies
COPY rag_app/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install uvicorn

# Copy the entire application
COPY rag_app /app/rag_app

# Change working directory to inside rag_app
WORKDIR /app/rag_app

# Expose the FastAPI port
EXPOSE 8080

# Start FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

