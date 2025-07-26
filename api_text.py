import logging
import os

import litserve as ls
import numpy as np
from sentence_transformers import SentenceTransformer

# Environment configurations
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
NUM_API_SERVERS = int(os.environ.get("NUM_API_SERVERS", "1"))
WORKERS_PER_DEVICE = int(os.environ.get("WORKERS_PER_DEVICE", "1"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
BATCH_TIMEOUT = float(os.environ.get("BATCH_TIMEOUT", "0.1"))
NORMALIZE = bool(os.environ.get("NORMALIZE", "0"))
DIMENSION = int(os.environ.get("DIMENSION", "768"))
assert MAX_BATCH_SIZE > 1, "This implementation presumes MAX_BATCH_SIZE > 1"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NomicTextAPI(ls.LitAPI):
    def setup(self, device: str):
        logger.info("Setting up Nomic text model.")
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device
        )
        self.prefix = "search_query: "
        logger.info("Text model setup complete.")

    def decode_request(self, request):
        return self.prefix + np.asarray(request["input"])

    def predict(self, inputs):
        embeddings = self.model.encode(inputs)
        # normalize the embeddings
        if NORMALIZE:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings[:, :DIMENSION]

    def encode_response(self, output):
        return {"embeddings": output.tolist()}


if __name__ == "__main__":
    api = NomicTextAPI(
        max_batch_size=MAX_BATCH_SIZE,
        batch_timeout=BATCH_TIMEOUT,
        api_path="/embed",
    )
    server = ls.LitServer(
        api,
        accelerator="auto",
        track_requests=True,
        workers_per_device=WORKERS_PER_DEVICE,
    )
    server.run(
        port=PORT,
        host="0.0.0.0",
        log_level=LOG_LEVEL.lower(),
        num_api_servers=NUM_API_SERVERS,
        generate_client_file=False,
    )
