#!/bin/bash

curl --request POST \
  --url http://localhost:8000/img/embed \
  -H "Content-Type: application/json" \
  -d '{
  "content": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
}' | jq '.embeddings[0:10]'
echo -e "\n-----------------"


curl --request POST \
  --url http://localhost:8000/txt/embed \
  -H "Content-Type: application/json" \
  -d '{
  "input": "something to embed"
}' | jq '.embeddings[0:10]'
echo -e "\n-----------------"


curl --request POST \
  --url http://localhost:8000/img/stats \
  -H "Content-Type: application/json" \
  -d '{
  "content": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
}' | jq
echo -e "\n-----------------"
