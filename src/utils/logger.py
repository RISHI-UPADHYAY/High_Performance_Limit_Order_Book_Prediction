from loguru import logger

logger.remove()

logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time}</green> | <level>{level}</level> | {message}"
)

def get_logger():
    return logger