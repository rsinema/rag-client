#!/bin/bash

# Configuration
REPO_PATH="/home/rsinema/App/rag-client"  
DOCKER_IMAGE_NAME="rag-client"     
CONTAINER_NAME="rag-client-web"
APP_PORT=8001                    

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Navigate to repository
cd $REPO_PATH || exit 1

# Pull latest changes and capture output
log_message "Pulling latest changes from main branch..."
PULL_OUTPUT=$(git pull origin main 2>&1)
echo "$PULL_OUTPUT"

# Check if pull was successful
if [ $? -ne 0 ]; then
    log_message "Failed to pull changes from repository"
    exit 1
fi

# Check if there were actually any changes
if echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
    log_message "No new changes detected, skipping rebuild"
    
    # Check if container is already running
    if docker ps | grep -q $CONTAINER_NAME; then
        log_message "Container is already running, no action needed"
        exit 0
    else
        log_message "Container not running, starting it with existing image..."
        docker run -d \
            --name $CONTAINER_NAME \
            -p $APP_PORT:80 \
            --restart unless-stopped \
            $DOCKER_IMAGE_NAME
        exit 0
    fi
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