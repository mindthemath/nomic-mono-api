build: lint requirements.cpu.txt
	docker build -f docker/Dockerfile.cpu -t nomic-mono-1.5-api:cpu -t nomic-mono-1.5-api:latest .
	# docker build -f docker/Dockerfile.prebaked -t nomic-mono-1.5-api:cpu-prebaked .

snowman.png:
	curl -fsSL https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png -o snowman.png

test-embed-image: snowman.png
	curl -X POST -F "content=@snowman.png" http://127.0.0.1:8000/img/embed | jq .embeddings

test-embed-text:
	curl -X POST -F "input=hello" http://127.0.0.1:8000/txt/embed | jq .embeddings

test-image-stats:
	curl -X POST -F "content=@snowman.png" http://127.0.0.1:8000/img/stats | jq

test: test-embed-image test-embed-text test-image-stats

ptest: snowman.png
	seq 1 64 | parallel --jobs 24 "curl -X POST -F 'content=@snowman.png' http://127.0.0.1:8000/img/embed 2>&1 || echo 'Request failed'"

lint:
	uvx black .
	uvx isort --profile black .
	uvx ruff check . --fix

tag: build-126
	docker tag mindthemath/nomic-mono-1.5-api:cu12.6 mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-gpu
	docker tag mindthemath/nomic-mono-1.5-api:cu12.6 mindthemath/nomic-mono-1.5-api:gpu
	docker images | grep mindthemath/nomic-mono-1.5-api

build-118: requirements.cu118.txt
	docker build -f docker/Dockerfile.cu118 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8.0 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8 \
		-t mindthemath/nomic-mono-1.5-api:cu11.8.0 \
		-t mindthemath/nomic-mono-1.5-api:cu11.8 \
		.

build-122: requirements.cu122.txt
	docker build -f docker/Dockerfile.cu122 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2.2 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2 \
		-t mindthemath/nomic-mono-1.5-api:cu12.2.2 \
		-t mindthemath/nomic-mono-1.5-api:cu12.2 \
		.

build-126: requirements.cu126.txt
	docker build -f docker/Dockerfile.cu126 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.6.1 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.6 \
		-t mindthemath/nomic-mono-1.5-api:cu12.6.1 \
		-t mindthemath/nomic-mono-1.5-api:cu12.6 \
		.

push-cu: lint build-118 build-122 build-126 tag
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8.0
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2.2
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.6.1
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.6
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-gpu
	docker push mindthemath/nomic-mono-1.5-api:cu12.6.1
	docker push mindthemath/nomic-mono-1.5-api:cu12.2.2
	docker push mindthemath/nomic-mono-1.5-api:cu11.8.0
	docker push mindthemath/nomic-mono-1.5-api:cu12.6
	docker push mindthemath/nomic-mono-1.5-api:cu12.2
	docker push mindthemath/nomic-mono-1.5-api:cu11.8
	docker push mindthemath/nomic-mono-1.5-api:gpu

push-pre: lint
	docker buildx build --builder multiarch-builder -f docker/Dockerfile.prebaked \
		--platform linux/amd64,linux/arm64 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cpu-prebaked \
		-t mindthemath/nomic-mono-1.5-api:cpu-prebaked \
		--push \
		.

push-cpu: lint
	docker buildx build --builder multiarch-builder -f docker/Dockerfile.cpu \
		--platform linux/amd64,linux/arm64 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cpu \
		-t mindthemath/nomic-mono-1.5-api:cpu \
		-t mindthemath/nomic-mono-1.5-api:latest \
		--push \
		.
	docker images | grep mindthemath/nomic-mono-1.5-api

push: push-cu push-cpu push-pre

run-gpu: tag
	docker run --rm -ti \
	--name nomic-mono-api-gpu \
	--gpus all \
	-p 8000:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),1) \
	-e WORKERS_PER_DEVICE=$(or $(WORKERS_PER_DEVICE),4) \
	-e IMAGE_MAX_BATCH_SIZE=$(or $(IMAGE_MAX_BATCH_SIZE),64) \
	-e TEXT_MAX_BATCH_SIZE=$(or $(TEXT_MAX_BATCH_SIZE),64) \
	-e THUMBNAIL_SIZE=$(or $(THUMBNAIL_SIZE),512) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	mindthemath/nomic-mono-1.5-api:gpu

run-cpu: build
	docker run --rm -ti \
	--name nomic-mono-api-cpu \
	-p 8000:8000 \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	nomic-mono-1.5-api:latest

setup-buildx:
	docker buildx create --name multiarch-builder
	docker buildx inspect --bootstrap

requirements: requirements.cu118.txt requirements.cu122.txt requirements.cu126.txt requirements.cpu.txt

requirements.cu118.txt: pyproject.toml uv.lock
	uv pip compile pyproject.toml --extra cu118 --upgrade -o requirements.cu118.txt

requirements.cu122.txt: pyproject.toml uv.lock
	uv pip compile pyproject.toml --extra cu122 --upgrade -o requirements.cu122.txt

requirements.cu126.txt: pyproject.toml uv.lock
	uv pip compile pyproject.toml --extra cu126 --upgrade -o requirements.cu126.txt

requirements.cpu.txt: pyproject.toml uv.lock
	uv pip compile pyproject.toml --extra cpu --upgrade -o requirements.cpu.txt
