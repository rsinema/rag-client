#!/bin/bash

# Configuration
REPO_PATH="/home/rsinema/App/rag-client"  # Adjust this to your repository root where docker-compose.yml is located
COMPOSE_FILE="docker-compose.yml"

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
    log_message "No new changes detected, checking if services are running..."
    
    # Check if containers are running
    if docker-compose ps | grep -q "Up"; then
        log_message "Services are already running, no action needed"
        exit 0
    else
        log_message "Services not running, starting them..."
        docker-compose up -d
        exit 0
    fi
fi

# Build and start services
log_message "Building and starting services..."
docker-compose up -d --build

# Check if services are running
if [ $? -ne 0 ]; then
    log_message "Failed to start services"
    exit 1
fi

# List running services
log_message "Current running services:"
docker-compose ps

log_message "Deployment completed successfully"

# Optional: Prune unused images to free up space
log_message "Cleaning up unused images..."
docker image prune -f