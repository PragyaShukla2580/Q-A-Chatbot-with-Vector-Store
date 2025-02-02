import os
import json
from langchain.chat_models import ChatOpenAI

CONFIG_PATH = "configs/latest_config.json"

def load_config(collection_name: str):
    config_file_path = f"configs/{collection_name}_config.json"
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"No config file found for {collection_name}. Please run the store data API first.")
    with open(config_file_path, "r") as config_file:
        return json.load(config_file)

# try:
#       # Change this to dynamic input if needed
#     config = load_config(collection_name)
#     VECTOR_STORE_NAME = config["vector_store_name"]
#     CHAT_HISTORY_NAME = config["chat_history_name"]
# except FileNotFoundError as e:
#     logger.error(str(e))
#     VECTOR_STORE_NAME = None
#     CHAT_HISTORY_NAME = None