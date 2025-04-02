import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import src.constants as CONST
from src.conf.events_conf.events_conf import EventsConf
from src.conf.testing.testing_conf import TestingConf
from src.conf.training.model.multivariate_hawkes_training_conf import (
    MultivariateHawkesTrainingConf,
)
from src.events_extractor.multivariate_lob_events_extractor import (
    MultivariateLOBEventsExtractor,
)
from src.lob_data_loader.loading_info_getter import LoadingInfoGetter
from src.lob_data_loader.lob_data_loader import LOBDataLoader
from src.lob_period.lob_period_extractor import LOBPeriodExtractor
from src.multivariate_hawkes_training.lob_event_combinator import LOBEventCombinator
from src.parser.predict_events_parser import PredictEventParser
from src.parser.predict_events_run_info import PredictEventsRunInfo
from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_model.time_prediction_model_factory.time_prediction_model_factory import (
    TimePredictionModelFactory,
)
from time_prediction_tester.every_time_prediction_tester import (
    EveryTimePredictionTester,
)


def get_conf(path: str) -> MultivariateHawkesTrainingConf:
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    return conf


def get_event_type_times_maps_with_combined_types(
    event_type_times_map: List[Dict[str, np.ndarray]],
    combined_name_events_to_combine_map: Dict[str, List[str]],
) -> List[Dict[str, np.ndarray]]:

    lob_event_combinator = LOBEventCombinator([event_type_times_map])

    for (
        combination_name,
        lob_events_to_combine,
    ) in combined_name_events_to_combine_map.items():
        event_type_times_maps = (
            lob_event_combinator.get_event_type_times_maps_with_new_combination(
                lob_events_to_combine,
                combination_name=combination_name,
            )
        )
        lob_event_combinator.event_type_times_maps = event_type_times_maps

    return event_type_times_maps


def get_event_type_times_maps_filtered(
    event_type_times_map: List[Dict[str, np.ndarray]], events_to_compute: List[str]
) -> List[Dict[str, np.ndarray]]:
    return [
        {
            key: value
            for key, value in event_type_times.items()
            if key in events_to_compute
        }
        for event_type_times in event_type_times_map
    ]


if __name__ == "__main__":
    parser = PredictEventParser()
    run_info_db_path = parser.parse()
    run_info_db = pd.read_csv(run_info_db_path, sep="\t")

    cpu_times = []
    for row in run_info_db.itertuples(index=False):
        print(row)
        run_info = PredictEventsRunInfo.from_namedtuple(row)

        testing_conf_map = get_conf(
            os.path.join(CONST.CONF_TESTING_FOLDER, run_info.testing_conf_filename)
        )
        testing_conf = TestingConf.from_dict(testing_conf_map)

        events_conf_map = get_conf(
            os.path.join(CONST.CONF_EVENTS_FOLDER, run_info.events_conf_filename)
        )
        events_conf = EventsConf.from_dict(events_conf_map)

        pair_orderbook_changes_path = os.path.join(
            CONST.ORDERBOOK_CHANGES_FOLDER, testing_conf.pair
        )
        periods_df = pd.read_csv(
            os.path.join(
                pair_orderbook_changes_path, CONST.SIMULATION_START_TIMESTAMPS_FILE
            )
        )

        loading_info_for_all_dfs = LoadingInfoGetter(periods_df).get_loading_info(
            lob_df_folder_path=pair_orderbook_changes_path,
            lob_df_prefix=CONST.ORDERBOOK_CHANGES_FILE_PREFIX,
        )

        for loading_info in loading_info_for_all_dfs:
            print(loading_info.path)
            lob_df_loader = LOBDataLoader()
            lob_df = lob_df_loader.get_lob_dataframe(loading_info.path, 10)

            lob_period_extractor = LOBPeriodExtractor(lob_df)

            for start_simulation_time in loading_info.start_times:
                start_warmup_time = (
                    start_simulation_time - testing_conf.seconds_warm_up_period
                )
                end_simulation_time = (
                    start_simulation_time + testing_conf.seconds_simulation_period
                )

                lob_period = lob_period_extractor.get_lob_period(
                    start_warmup_time, end_simulation_time
                )
                lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()

                lob_df_for_events["Timestamp"] = lob_df_for_events["Timestamp"] * 1000

                lob_events_extractor = MultivariateLOBEventsExtractor(
                    lob_df_for_events,
                    events_conf.num_levels_in_a_side,
                    events_conf.num_levels_for_which_save_events,
                )

                event_type_times_map = lob_events_extractor.get_events()
                event_type_times_map = {
                    key.name: value for key, value in event_type_times_map.items()
                }

                event_type_times_maps = get_event_type_times_maps_with_combined_types(
                    event_type_times_map, events_conf.combined_event_types_map
                )

                event_type_times_maps = get_event_type_times_maps_filtered(
                    event_type_times_maps, events_conf.events_to_compute
                )

                event_type_times_maps_formatted_in_seconds = [
                    {
                        event_type: (times / 1000)
                        for event_type, times in event_type_times_map.items()
                    }
                    for event_type_times_map in event_type_times_maps
                ]

                simulated_params_dir = os.path.join(
                    CONST.TRAINED_PARAMS_FOLDER,
                    run_info.model_name,
                    testing_conf.pair,
                )

                simulated_params_subdirs = [
                    os.path.join(simulated_params_dir, d)
                    for d in os.listdir(simulated_params_dir)
                    if os.path.isdir(os.path.join(simulated_params_dir, d))
                    # and "lshade_training" in d
                ]

                for simulated_params_subdir in simulated_params_subdirs:
                    time_prediction_model_factory = TimePredictionModelFactory(
                        run_info.model_name,
                        30,
                        simulated_params_subdir,
                        loading_info.start_registration_time,
                        start_simulation_time,
                    )

                    time_prediction_model = time_prediction_model_factory.get_model()

                    period_for_simulation = PeriodForSimulation(
                        event_type_event_times_map=event_type_times_maps_formatted_in_seconds[
                            0
                        ],
                        event_types_to_predict=["MID_PRICE_CHANGE"],
                        event_types_order=events_conf.events_to_compute,
                    )

                    time_prediction_tester = EveryTimePredictionTester(
                        time_prediction_model,
                        period_for_simulation,
                        testing_conf.seconds_warm_up_period,
                    )

                    try:
                        start_cpu_time = time.process_time()
                        event_type_predicted_events_map = (
                            time_prediction_tester.get_predicted_event_times()
                        )
                        end_cpu_time = time.process_time()

                        cpu_times.append((end_cpu_time - start_cpu_time) / len(event_type_predicted_events_map["MID_PRICE_CHANGE"]))
                        event_type_real_events_map = (
                            time_prediction_tester.get_event_type_real_event_times_map()
                        )

                        predicted_array = event_type_predicted_events_map[
                            "MID_PRICE_CHANGE"
                        ]
                        real_array = event_type_real_events_map["MID_PRICE_CHANGE"]

                        df = pd.DataFrame(
                            {"real": real_array, "predicted": predicted_array}
                        )

                        simulations_dir = os.path.join(
                            CONST.SIMULATIONS_FOLDER,
                            run_info.model_name,
                            testing_conf.pair,
                            os.path.basename(os.path.normpath(simulated_params_subdir)),
                            f"simulation_seconds_{testing_conf.seconds_simulation_period}",
                        )

                        if not os.path.exists(simulations_dir):
                            os.makedirs(simulations_dir)

                        prefix = os.path.basename(loading_info.path)
                        prefix = os.path.splitext(prefix)[0]
                        prefix = os.path.join(simulations_dir, prefix)

                        df.to_csv(
                            os.path.join(f"{prefix}_{start_simulation_time}.tsv"),
                            index=False,
                            sep="\t",
                        )

                    except Exception as e:
                        print(e)
                        print(
                            f"{simulated_params_subdir} start_simulation_time {start_simulation_time} failed"
                        )

    with open(
        os.path.join(simulations_dir, "times_simulation.txt"), "w"
    ) as file:
        file.writelines(f"{item}\n" for item in cpu_times)