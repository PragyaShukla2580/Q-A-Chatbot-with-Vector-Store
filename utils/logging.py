import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with current date
LOG_FILE = os.path.join(LOG_DIR, f"api_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        TimedRotatingFileHandler(LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8"),  # Rotate daily, keep 7 days
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
