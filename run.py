import argparse
from flatland_sim import generate_scenarios


def main():
    parser = argparse.ArgumentParser(description="Run flatland-sim scenario generation.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file.")
    parser.add_argument("--preview", action="store_true", help="Enable preview mode.")
    args = parser.parse_args()
    generate_scenarios(config=args.config, preview=args.preview)


if __name__ == "__main__":
    main()
