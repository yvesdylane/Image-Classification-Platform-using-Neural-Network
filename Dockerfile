# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory in the container(Normally you should know that is the directory of the container but we never know)
WORKDIR /app

# Install system dependencies and clean up to minimize image size(all it because of of liberies like hdbscan)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now understand this by yourself
COPY . .

# Expose the application port to your  own preference
EXPOSE 8080

# Set the command to run the application
CMD ["python", "app.py"]