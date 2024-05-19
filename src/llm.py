"""
===========================================
        Module: Open-source LLM Setup
===========================================
"""

from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml
import os

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open("config/config.yml", "r", encoding="utf8") as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_llm():
    this_cuda_path = os.environ["CUDA_PATH"]

    # if cuda_path ends with bin, remove bin in the os environment variable
    if this_cuda_path.endswith("bin"):
        # set the new cuda path
        os.environ["CUDA_PATH"] = this_cuda_path[:-3]
        print("New CUDA PATH:", os.environ["CUDA_PATH"])

    # Local CTransformers model
    llm = CTransformers(
        model=cfg.MODEL_BIN_PATH,
        model_type=cfg.MODEL_TYPE,
        config={"max_new_tokens": cfg.MAX_NEW_TOKENS, "temperature": cfg.TEMPERATURE},
    )

    return llm
