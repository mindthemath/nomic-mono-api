build: requirements.api.txt
	docker build -t nomic-vision-1.5-api:latest .

snowman.png:
	curl -fsSL https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png -o snowman.png

test: snowman.png
	curl -X POST -F "content=@snowman.png" http://127.0.0.1:8030/embed | jq .output

ptest: snowman.png
	seq 1 23 | parallel --jobs 24 "curl -X POST -F 'content=@snowman.png' http://127.0.0.1:8030/embed 2>&1 || echo 'Request failed'"

lint:
	uvx black .
	uvx isort --profile black .

tag: build
	docker tag nomic-vision-1.5-api:latest mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2.2
	docker tag nomic-vision-1.5-api:latest mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)
	docker tag nomic-vision-1.5-api:latest mindthemath/nomic-vision-1.5-api:latest
	docker images | grep mindthemath/nomic-vision-1.5-api

run: build
	docker run --rm -ti \
	--name embed-image-v1.5 \
	--gpus all \
	-p 8030:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),1) \
	-e MAX_BATCH_SIZE=$(or $(MAX_BATCH_SIZE),1) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	nomic-vision-1.5-api:latest
up: build
	docker run --restart unless-stopped -d \
	--name embed-image-v1.5 \
	--gpus all \
	-p 8030:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),1) \
	-e MAX_BATCH_SIZE=$(or $(MAX_BATCH_SIZE),1) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	nomic-vision-1.5-api:latest

requirements.api.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api -o requirements.api.txt
