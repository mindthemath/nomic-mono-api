import logging
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def decode_request(request) -> Image.Image | None:
    file_obj = request["content"]

    if isinstance(file_obj, str) and "http" in file_obj:
        file_obj = file_obj.replace("localhost:3210", "backend:3210")  # HACK
        image = Image.open(
            requests.get(file_obj, stream=True).raw
        )  # TODO: handle errors?
        logger.info("Processing URL input.")
    try:
        file_bytes = file_obj.file.read()
        image = Image.open(BytesIO(file_bytes))
        logger.info("Processing file input.")
    except AttributeError:
        logger.warning("Faild to process request")
    finally:
        if not isinstance(file_obj, str):
            file_obj.file.close()

    return image
