import logging
import os
from io import BytesIO

import litserve as ls
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Environment configurations
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
NUM_API_SERVERS = int(os.environ.get("NUM_API_SERVERS", "1"))
WORKERS_PER_DEVICE = int(os.environ.get("WORKERS_PER_DEVICE ", "1"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
NORMALIZE = bool(os.environ.get("NORMALIZE", "0"))
DIMENSION = int(os.environ.get("DIMENSION", "256"))

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NomicVisionAPI(ls.LitAPI):
    def setup(self, device):
        logger.info("Setting up Nomic vision model.")
        self.processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5",
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.normalize = NORMALIZE
        self.dimension = DIMENSION

    def decode_request(self, request):
        file_obj = request["content"]

        if isinstance(file_obj, str) and "http" in file_obj:
            file_obj = file_obj.replace("localhost:3210", "backend:3210")  # HACK
            image = Image.open(requests.get(file_obj, stream=True).raw)
            logger.info("Processing URL input.")
            return image
        try:
            file_bytes = file_obj.file.read()
            image = Image.open(BytesIO(file_bytes))
            logger.info("Processing file input.")
            return image
        except AttributeError:
            logger.warning("Faild to process request")
        finally:
            if not isinstance(file_obj, str):
                file_obj.file.close()

    def predict(self, images):
        if isinstance(images, list):
            logger.info(f"Generating {len(images)} embeddings.")
        else:
            logger.info("Generating 1 embedding.")

        inputs = self.processor(images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            img_emb = self.model(**inputs).last_hidden_state
            img_embeddings = img_emb[:, 0]

        # Truncate to Matryoshka embedding dimension
        embedding = img_embeddings[:, : self.dimension]
        # Apply normalization if requested
        logger.debug(f"Embedding shape: {embedding.shape}")

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.cpu().numpy()

    def encode_response(self, output):
        return {"embedding": output.tolist()}


if __name__ == "__main__":
    api = NomicVisionAPI(
        max_batch_size=MAX_BATCH_SIZE,
        batch_timeout=1,
    )
    server = ls.LitServer(
        api,
        accelerator="auto",
        track_requests=True,
        api_path="/embed",
        workers_per_device=WORKERS_PER_DEVICE,
    )
    server.run(
        port=PORT,
        host="0.0.0.0",
        log_level=LOG_LEVEL.lower(),
        num_api_servers=NUM_API_SERVERS,
        generate_client_file=False,
    )
