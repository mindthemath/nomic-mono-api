build: requirements.api.txt server.py
	docker build -f Dockerfile -t nomic-vision-1.5-api:latest .

snowman.png:
	curl -fsSL https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png -o snowman.png

test: snowman.png
	curl -X POST -F "content=@snowman.png" http://127.0.0.1:8030/embed | jq .embedding

ptest: snowman.png
	seq 1 23 | parallel --jobs 24 "curl -X POST -F 'content=@snowman.png' http://127.0.0.1:8030/embed 2>&1 || echo 'Request failed'"

lint:
	uvx black .
	uvx isort --profile black .

tag: build
	docker tag nomic-vision-1.5-api:latest mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-gpu
	docker tag nomic-vision-1.5-api:latest mindthemath/nomic-vision-1.5-api:gpu
	docker images | grep mindthemath/nomic-vision-1.5-api

build-122:
	docker build -t mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2.2 -f Dockerfile.cu122 .
	docker tag mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2.2 mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2
	docker tag mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2.2 mindthemath/nomic-vision-1.5-api:cu12.2.2
	docker tag mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2.2 mindthemath/nomic-vision-1.5-api:cu12.2

build-121: requirements.cu121.txt
	docker build -t mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.1.1 -f Dockerfile.cu121 .
	docker tag mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.1.1 mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.1
	docker tag mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.1.1 mindthemath/nomic-vision-1.5-api:cu12.1.1
	docker tag mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.1.1 mindthemath/nomic-vision-1.5-api:cu12.1

push-cu: build-121 build-122 tag
	docker push mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.1.1
	docker push mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2.2
	docker push mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.1
	docker push mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cu12.2
	docker push mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-gpu
	docker push mindthemath/nomic-vision-1.5-api:cu12.2.2
	docker push mindthemath/nomic-vision-1.5-api:cu12.1.1
	docker push mindthemath/nomic-vision-1.5-api:cu12.2
	docker push mindthemath/nomic-vision-1.5-api:cu12.1
	docker push mindthemath/nomic-vision-1.5-api:gpu

push: push-cpu
	docker buildx build -f Dockerfile.prebaked \
		--platform linux/amd64,linux/arm64 \
		-t mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cpu-prebaked \
		-t mindthemath/nomic-vision-1.5-api:cpu-prebaked \
		--push \
		.

push-cpu: build
	docker buildx build -f Dockerfile.cpu \
		--platform linux/amd64,linux/arm64 \
		-t mindthemath/nomic-vision-1.5-api:$$(date +%Y%m%d)-cpu \
		-t mindthemath/nomic-vision-1.5-api:cpu \
		-t mindthemath/nomic-vision-1.5-api:latest \
		--push \
		.
	docker images | grep mindthemath/nomic-vision-1.5-api

run: build
	docker run --rm -ti \
	--name embed-image-v1.5 \
	--gpus all \
	-p 8030:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),1) \
	-e MAX_BATCH_SIZE=$(or $(MAX_BATCH_SIZE),32) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	nomic-vision-1.5-api:latest

up: build
	docker run --restart unless-stopped -d \
	--name embed-image-v1.5 \
	--gpus all \
	-p 8030:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),4) \
	-e MAX_BATCH_SIZE=$(or $(MAX_BATCH_SIZE),32) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	nomic-vision-1.5-api:latest

requirements.api.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api --extra cu122 -o requirements.api.txt

requirements.cu122.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api --extra cu122 -o requirements.cu122.txt

requirements.cu121.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api --extra cu121 -o requirements.cu121.txt

requirements.cpu.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api --extra cpu -o requirements.cpu.txt
