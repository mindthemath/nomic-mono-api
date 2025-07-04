build: requirements.cpu.txt
	docker build -f Dockerfile.cpu -t nomic-mono-1.5-api:cpu .
	docker build -f Dockerfile.prebaked -t nomic-mono-1.5-api:cpu-prebaked .

snowman.png:
	curl -fsSL https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png -o snowman.png

test: snowman.png
	curl -X POST -F "content=@snowman.png" http://127.0.0.1:8030/embed | jq .embedding

ptest: snowman.png
	seq 1 24 | parallel --jobs 24 "curl -X POST -F 'content=@snowman.png' http://127.0.0.1:8030/embed 2>&1 || echo 'Request failed'"

lint:
	uvx black .
	uvx isort --profile black .

tag: build
	docker tag nomic-mono-1.5-api:latest mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-gpu
	docker tag nomic-mono-1.5-api:latest mindthemath/nomic-mono-1.5-api:gpu
	docker images | grep mindthemath/nomic-mono-1.5-api

build-118: requirements.cu118.txt
	docker build -t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8.0 -f Dockerfile.cu118 .
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8.0 mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8.0 mindthemath/nomic-mono-1.5-api:cu11.8.0
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8.0 mindthemath/nomic-mono-1.5-api:cu11.8

build-122: requirements.cu122.txt
	docker build -t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2.2 -f Dockerfile.cu122 .
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2.2 mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2.2 mindthemath/nomic-mono-1.5-api:cu12.2.2
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2.2 mindthemath/nomic-mono-1.5-api:cu12.2

build-124: requirements.cu124.txt
	docker build -t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.4.1 -f Dockerfile.cu124 .
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.4.1 mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.4
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.4.1 mindthemath/nomic-mono-1.5-api:cu12.4.1
	docker tag mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.4.1 mindthemath/nomic-mono-1.5-api:cu12.4

push-cu: build-118 build-122 build-124 tag
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8.0
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2.2
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.4.1
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu11.8
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.2
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cu12.4
	docker push mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-gpu
	docker push mindthemath/nomic-mono-1.5-api:cu12.4.1
	docker push mindthemath/nomic-mono-1.5-api:cu12.2.2
	docker push mindthemath/nomic-mono-1.5-api:cu11.8.0
	docker push mindthemath/nomic-mono-1.5-api:cu12.4
	docker push mindthemath/nomic-mono-1.5-api:cu12.2
	docker push mindthemath/nomic-mono-1.5-api:cu11.8
	docker push mindthemath/nomic-mono-1.5-api:gpu

push: push-cpu
	docker buildx build --builder multiarch-builder -f Dockerfile.prebaked \
		--platform linux/amd64,linux/arm64 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cpu-prebaked \
		-t mindthemath/nomic-mono-1.5-api:cpu-prebaked \
		--push \
		.

push-cpu: build
	docker buildx build --builder multiarch-builder -f Dockerfile.cpu \
		--platform linux/amd64,linux/arm64 \
		-t mindthemath/nomic-mono-1.5-api:$$(date +%Y%m%d)-cpu \
		-t mindthemath/nomic-mono-1.5-api:cpu \
		-t mindthemath/nomic-mono-1.5-api:latest \
		--push \
		.
	docker images | grep mindthemath/nomic-mono-1.5-api

run: build
	docker run --rm -ti \
	--name embed-image-v1.5 \
	--gpus all \
	-p 8030:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),1) \
	-e WORKERS_PER_DEVICE=$(or $(WORKERS_PER_DEVICE),4) \
	-e MAX_BATCH_SIZE=$(or $(MAX_BATCH_SIZE),32) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	nomic-mono-1.5-api:latest

up: build
	docker run --restart unless-stopped -d \
	--name embed-image-v1.5 \
	--gpus all \
	-p 8030:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),1) \
	-e WORKERS_PER_DEVICE=$(or $(WORKERS_PER_DEVICE),4) \
	-e MAX_BATCH_SIZE=$(or $(MAX_BATCH_SIZE),32) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	nomic-mono-1.5-api:latest

setup-buildx:
	docker buildx create --name multiarch-builder
	docker buildx inspect --bootstrap

requirements: requirements.api.txt requirements.cu118.txt requirements.cu122.txt requirements.cu124.txt requirements.cpu.txt

requirements.api.txt: pyproject.toml
	uv pip compile pyproject.toml --extra cu122 --upgrade -o requirements.api.txt

requirements.cu118.txt: pyproject.toml
	uv pip compile pyproject.toml --extra cu118 --upgrade -o requirements.cu118.txt

requirements.cu122.txt: pyproject.toml
	uv pip compile pyproject.toml --extra cu122 --upgrade -o requirements.cu122.txt

requirements.cu124.txt: pyproject.toml
	uv pip compile pyproject.toml --extra cu126 --upgrade -o requirements.cu124.txt

requirements.cpu.txt: pyproject.toml
	uv pip compile pyproject.toml --extra cpu --upgrade -o requirements.cpu.txt
