import argparse


class PredictEventParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Predict events from configuration database file")
        self._add_arguments()

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            'configuration_db_path',
            type=str,
            help="Path of configuration database file"
        )

    def parse(self) :
        args = self.parser.parse_args()
        return args.configuration_db_path
