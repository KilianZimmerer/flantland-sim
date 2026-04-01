import argparse
import yaml

from flatland_sim.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run flatland-sim scenario generation.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    Pipeline(config).run()


if __name__ == "__main__":
    main()
