#!/bin/bash

curl --request POST \
  --url http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
  "content": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
}' | jq '.embedding[0:10]'

echo -e "\n-----------------"

curl --request POST \
  --url http://localhost:8000/stats \
  -H "Content-Type: application/json" \
  -d '{
  "content": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
}' | jq
echo -e "\n-----------------"
