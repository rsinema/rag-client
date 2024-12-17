# Use Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY llm.py .

# Copy the .env file if it exists
COPY .env* ./ 

# Run the script
CMD ["python", "llm.py"]