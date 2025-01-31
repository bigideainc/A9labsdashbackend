FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (including OpenSSH client)
RUN apt-get update && apt-get install -y openssh-client && rm -rf /var/lib/apt/lists/*

# Install Python dependencies, including PyTorch for model loading
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Set Hugging Face cache directory
ENV HF_HOME=/app/huggingface_cache
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME

# Ensure user has a valid home directory
ENV HOME=/app

# Copy application code
COPY . .

COPY creds.json .

# Create non-root user
RUN adduser --system --group app_user
RUN chown -R app_user:app_user /app

# Switch to non-root user
USER app_user

# Use environment variable for port binding
ENV PORT=8890
# Command to run the application
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8890"]
