import random


class RandomConfigSampler:
    def __init__(self, config: dict):
        self._rng = random.Random(config["seed"])
        self._randomization = config["randomization"]

    def sample(self) -> dict:
        return {
            key: self._rng.randint(params["min"], params["max"])
            for key, params in self._randomization.items()
        }
