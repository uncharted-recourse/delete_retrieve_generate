api:
	@echo "Building Image"
	docker build -t docker.ased.uncharted.software/delete_and_retrieve-api:$(version) -f flask.dockerfile .
	@echo "Starting a web API at http://localhost:5000"
	docker run --rm -p 5000:5000 docker.ased.uncharted.software/delete_and_retrieve-api:$(version)

