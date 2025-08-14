import os
import wget
import logging

logger = logging.getLogger(__name__)

def dataset_download(data_url: str, filename: str) -> None:    
    """
    Helper function to download files from link

    Args:
    data_url - link to files
    filename - save name for files downloaded from URL

    Returns:
    None
    """
    logger.debug("******")
    if not os.path.exists(f"{DATA_DIR}/{filename}"):
        data_path = wget.download(data_url, DATA_DIR)
        logger.info(f"Dataset downloaded at: {data_path}")
    else:
        logger.debug("Tarfile already exists.")
        data_path = f'{DATA_DIR}/{filename}'