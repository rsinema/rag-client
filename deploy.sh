#!/bin/bash

# Configuration
REPO_PATH="/home/rsinema/App"  
DOCKER_IMAGE_NAME="rag-client"     
CONTAINER_NAME="rag-client-web"
APP_PORT=8001                    

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Navigate to repository
cd $REPO_PATH || exit 1

# Pull latest changes
log_message "Pulling latest changes from main branch..."
git pull origin main

# Check if pull was successful
if [ $? -ne 0 ]; then
    log_message "Failed to pull changes from repository"
    exit 1
fi

# Build new Docker image
log_message "Building Docker image..."
docker build -t $DOCKER_IMAGE_NAME .

# Check if build was successful
if [ $? -ne 0 ]; then
    log_message "Failed to build Docker image"
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a | grep -q $CONTAINER_NAME; then
    log_message "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run new container
log_message "Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $APP_PORT:80 \
    --restart unless-stopped \
    $DOCKER_IMAGE_NAME

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    log_message "Container successfully started"
else
    log_message "Failed to start container"
    exit 1
fi

log_message "Deployment completed successfully"