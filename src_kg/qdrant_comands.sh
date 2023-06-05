# Download image from DockerHub:

docker pull qdrant/qdrant

# And run the service inside the docker:


docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
