import os

import litserve as ls

from api_embed import NomicVisionAPI
from api_stats import ImageStatsAPI
from api_text import NomicTextAPI

PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
NUM_API_SERVERS = int(os.environ.get("NUM_API_SERVERS", "1"))
WORKERS_PER_DEVICE = int(os.environ.get("WORKERS_PER_DEVICE ", "1"))
IMAGE_MAX_BATCH_SIZE = int(os.environ.get("IMAGE_MAX_BATCH_SIZE", "16"))
IMAGE_BATCH_TIMEOUT = float(os.environ.get("IMAGE_BATCH_TIMEOUT", "0.2"))
TEXT_MAX_BATCH_SIZE = int(os.environ.get("TEXT_MAX_BATCH_SIZE", "32"))
TEXT_BATCH_TIMEOUT = float(os.environ.get("TEXT_BATCH_TIMEOUT", "0.05"))

if __name__ == "__main__":
    stats_api = ImageStatsAPI(
        max_batch_size=1,
        api_path="/img/stats",
    )
    img_api = NomicVisionAPI(
        max_batch_size=IMAGE_MAX_BATCH_SIZE,
        batch_timeout=IMAGE_BATCH_TIMEOUT,
        api_path="/img/embed",
    )
    txt_api = NomicTextAPI(
        max_batch_size=TEXT_MAX_BATCH_SIZE,
        batch_timeout=TEXT_BATCH_TIMEOUT,
        api_path="/txt/embed",
    )
    server = ls.LitServer(
        [img_api, stats_api, txt_api],
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
