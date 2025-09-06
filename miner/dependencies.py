from miner.config import WorkerConfig
from miner.config import factory_worker_config


def get_worker_config() -> WorkerConfig:
    return factory_worker_config()
