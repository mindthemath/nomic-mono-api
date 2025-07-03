import os

import litserve as ls

from api_embed import NomicVisionAPI
from api_stats import ImageStatsAPI

PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
NUM_API_SERVERS = int(os.environ.get("NUM_API_SERVERS", "1"))
WORKERS_PER_DEVICE = int(os.environ.get("WORKERS_PER_DEVICE ", "1"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))

if __name__ == "__main__":
    stats_api = ImageStatsAPI(max_batch_size=1, api_path="/stats")
    embed_api = NomicVisionAPI(
        max_batch_size=MAX_BATCH_SIZE,
        batch_timeout=0.2,
        api_path="/embed",
    )
    server = ls.LitServer(
        [embed_api, stats_api],
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
