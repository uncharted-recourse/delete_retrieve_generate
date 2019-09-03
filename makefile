api:
	@echo "Building Image"
	docker build -t docker.ased.uncharted.software/delete_and_retrieve-api:$(version) -f flask.dockerfile .
	@echo "Starting a web API at http://localhost:5000"
	docker run --rm -p 5000:5000 docker.ased.uncharted.software/delete_and_retrieve-api:$(version)

train:
	@echo "Cleaning up containers"
	docker system prune
	@echo "Building image"
	docker build -t dar_train -f train.dockerfile .
	@echo "training using nvidia runtime"
	docker run --rm -v /Users/jgleason/Documents/NewKnowledge/ASED/delete_retrieve_generate:/root dar_train \
		python train.py --config config.json --bleu

