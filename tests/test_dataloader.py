from fronts.train import open_config_yaml_as_dataclass
from fronts.data import config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help=("Path to the configuration yaml."),
    )

    args = parser.parse_args()
    config = open_config_yaml_as_dataclass(
        path=args.config_path, config_class=config.DataConfig
    )

    data_config = config.build()
