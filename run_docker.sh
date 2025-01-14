#!/bin/bash

# Get your host's UID and GID
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

IMAGE_NAME="steps:latest"

# Build the Docker image if it doesn't exist
if [[ "$(docker images -q "${IMAGE_NAME}" 2> /dev/null)" == "" ]]; then
    echo "Image '${IMAGE_NAME}' does not exist. Building..."
    docker build --build-arg UID=$HOST_UID --build-arg GID=$HOST_GID -t steps . -f Dockerfile
else
    echo "Image '${IMAGE_NAME}' already exists. Skipping build."
fi

# Run the Docker container
#docker run -v ${HOME}/github/StePS:/home/steps/StePS -it --rm steps /bin/bash