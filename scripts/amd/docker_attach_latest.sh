LATEST_CONTAINER=$(docker ps --latest --format "{{.Names}}")
docker attach $LATEST_CONTAINER
