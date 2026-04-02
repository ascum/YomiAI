import os
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        logger.info(f"Starting download of {model_id}...")
        snapshot_download(
            repo_id=model_id,
            local_dir="venv/qwen-1.5b-cache",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info("Download complete!")
    except Exception as e:
        logger.error(f"Download failed: {e}")

if __name__ == "__main__":
    main()
