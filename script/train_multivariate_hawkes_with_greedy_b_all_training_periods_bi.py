import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import src.constants as CONST
from src.conf.events_conf.events_conf import EventsConf
from src.conf.training.model.multivariate_hawkes_training_conf import (
    MultivariateHawkesTrainingConf,
)
from src.conf.training.training_conf import TrainingConf
from src.lob_data_loader.loading_info_getter import LoadingInfoGetter
from src.lob_data_loader.lob_data_loader import LOBDataLoader
from src.lob_period.lob_period_extractor import LOBPeriodExtractor
from src.multivariate_hawkes_training.event_type_times_maps_formatter import (
    EventTypeTimesMapsFormatter,
)
from src.multivariate_hawkes_training.lob_event_combinator import LOBEventCombinator
from src.multivariate_hawkes_training.multivariate_hawkes_trainer_with_greedy_beta_search import (
    MultivariateHawkesTrainerWithGreedyBetaSearch,
)

CONF_EVENTS_FILENAME = "mid_price_increase_and_decrease_events_conf.yml"
CONF_TRAINING_FILENAME = "training_conf.yml"
CONF_MULTIVARIATE_HAWKES_TRAINING_FILENAME = "multivariate_hawkes_training_conf.yml"


def get_conf(path: str) -> Dict:
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


def extract_lob_events(df: pd.DataFrame, bi_level: int = 10) -> Dict[str, np.ndarray]:
    """
    Extract timestamps for four LOB events:
    1. Return > 0
    2. Return < 0
    3. AskSize1 increases OR BidSize1 decreases (without consuming all liquidity)
    4. AskSize1 decreases OR BidSize1 increases

    Parameters:
        df (pd.DataFrame): LOB DataFrame with columns ['Timestamp', 'AskSize1', 'BidSize1', 'Return']

    Returns:
        dict: A dictionary containing lists of timestamps for each event.
    """

    # Ensure the dataframe is sorted by Timestamp
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    pbid = df["BidPrice1"] - df[f"BidPrice{bi_level}"]
    pask = df[f"AskPrice{bi_level}"] - df["AskPrice1"]
    df["BaseImbalance"] = (pbid-pask)/(pbid+pask)


    # Shift values to compare with the previous row
    df["AskSize1_prev"] = df["AskSize1"].shift(1)
    df["BidSize1_prev"] = df["BidSize1"].shift(1)
    df["AskPrice1_prev"] = df["AskPrice1"].shift(1)
    df["BidPrice1_prev"] = df["BidPrice1"].shift(1)

    mask_event_1 = (((df["AskPrice1"] + df["BidPrice1"])/2) > ((df["AskPrice1_prev"] + df["BidPrice1_prev"])/2)) & (df["BaseImbalance"] > 0)
    timestamps_event_1 = df[mask_event_1]["Timestamp"].to_numpy()

    mask_event_2 = (((df["AskPrice1"] + df["BidPrice1"])/2) > ((df["AskPrice1_prev"] + df["BidPrice1_prev"])/2)) & (df["BaseImbalance"] < 0)
    timestamps_event_2 = df[mask_event_2]["Timestamp"].to_numpy()

    mask_event_3 = (((df["AskPrice1"] + df["BidPrice1"])/2) < ((df["AskPrice1_prev"] + df["BidPrice1_prev"])/2)) & (df["BaseImbalance"] > 0)
    timestamps_event_3 = df[mask_event_3]["Timestamp"].to_numpy()

    mask_event_4 = (((df["AskPrice1"] + df["BidPrice1"])/2) < ((df["AskPrice1_prev"] + df["BidPrice1_prev"])/2)) & (df["BaseImbalance"] < 0)
    timestamps_event_4 = df[mask_event_4]["Timestamp"].to_numpy()

    # sort the timestamps in ascending order
    timestamps_event_1.sort()
    timestamps_event_2.sort()
    timestamps_event_3.sort()
    timestamps_event_4.sort()

    return {
        "MP_UP_BI_POS": timestamps_event_1,
        "MP_UP_BI_NEG": timestamps_event_2,
        "MP_DOWN_BI_POS": timestamps_event_4,
        "MP_DOWN_BI_NEG": timestamps_event_3,
    }


if __name__ == "__main__":
    multivariate_hawkes_training_conf_map = get_conf(
        os.path.join(
            CONST.CONF_TRAINING_MODEL_FOLDER, CONF_MULTIVARIATE_HAWKES_TRAINING_FILENAME
        )
    )
    multivariate_hawkes_training_conf = MultivariateHawkesTrainingConf.from_dict(
        multivariate_hawkes_training_conf_map
    )

    training_conf_map = get_conf(
        os.path.join(CONST.CONF_TRAINING_FOLDER, CONF_TRAINING_FILENAME)
    )
    training_conf = TrainingConf.from_dict(training_conf_map)

    events_conf_map = get_conf(
        os.path.join(CONST.CONF_EVENTS_FOLDER, CONF_EVENTS_FILENAME)
    )
    events_conf = EventsConf.from_dict(events_conf_map)

    pair_orderbook_changes_path = os.path.join(
        CONST.ORDERBOOK_CHANGES_FOLDER, training_conf.pair
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

    training_time_file_likelihood_map = {
        training_time: {"file": [], "score": []}
        for training_time in training_conf.seconds_in_a_period
    }

    training_function_cpu_times = []

    for loading_info in loading_info_for_all_dfs:
        lob_df_loader = LOBDataLoader()
        lob_df = lob_df_loader.get_lob_dataframe(loading_info.path, 10)

        lob_period_extractor = LOBPeriodExtractor(lob_df)

        for start_simulation_time in loading_info.start_times:
            for training_time_seconds in training_conf.seconds_in_a_period:
                start_time = start_simulation_time - training_time_seconds

                end_time = start_simulation_time

                lob_period = lob_period_extractor.get_lob_period(start_time, end_time)
                lob_df_for_events = lob_period.get_lob_df_with_timestamp_column()

                event_type_times_maps = [extract_lob_events(lob_df_for_events)]

                # removing events considered as the same event if they are too close to each other
                event_type_times_maps = [
                    {
                        key: np.array(
                            [
                                timestamps[i]
                                for i in range(len(timestamps))
                                if i == 0 or (timestamps[i] - timestamps[i - 1]) > 0.01
                            ]
                        )
                        for key, timestamps in event_dict.items()
                    }
                    for event_dict in event_type_times_maps
                ]

                event_type_times_map_formatter = EventTypeTimesMapsFormatter()

                event_type_times_formatted_in_seconds = (
                    event_type_times_map_formatter.get_events_types_periods(
                        event_type_times_maps, [   "MP_UP_BI_POS",
                                "MP_UP_BI_NEG",
                                "MP_DOWN_BI_POS",
                                "MP_DOWN_BI_NEG"
                            ],
                    )
                )

                trainer = MultivariateHawkesTrainerWithGreedyBetaSearch(
                    event_type_times_formatted_in_seconds,
                    multivariate_hawkes_training_conf.betas_to_train,
                )

                start_time_cpu = time.process_time()
                hawkes_kernel = trainer.get_trained_kernel(
                    multivariate_hawkes_training_conf.beta_values_to_test
                )
                end_time_cpu = time.process_time()
                training_function_cpu_times.append(end_time_cpu - start_time_cpu)

                print(hawkes_kernel.score())

                params_dir = os.path.join(
                    CONST.TRAINED_PARAMS_FOLDER,
                    CONST.MULTIVARIATE_HAWKES + "_bid_ask_bi",
                    training_conf.pair,
                    "training_time_" + str(training_time_seconds),
                )

                if not os.path.exists(params_dir):
                    os.makedirs(params_dir, exist_ok=True)

                prefix = os.path.basename(loading_info.path)
                prefix = os.path.splitext(prefix)[0]
                prefix = os.path.join(params_dir, prefix)

                training_time_file_likelihood_map[training_time_seconds]["file"].append(
                    f"{prefix}_{start_simulation_time}"
                )
                training_time_file_likelihood_map[training_time_seconds][
                    "score"
                ].append(hawkes_kernel.score())

                np.savetxt(
                    f"{prefix}_{start_simulation_time}_mu.txt", hawkes_kernel.baseline
                )
                np.savetxt(
                    f"{prefix}_{start_simulation_time}_rho.txt",
                    hawkes_kernel.adjacency,
                )
                np.savetxt(
                    f"{prefix}_{start_simulation_time}_beta.txt", hawkes_kernel.decays
                )

                with open(
                    os.path.join(params_dir, "times.txt"), "w"
                ) as file:
                    file.writelines(f"{num}\n" for num in training_function_cpu_times)

    for training_time_seconds in training_conf.seconds_in_a_period:
        params_dir = os.path.join(
            CONST.TRAINED_PARAMS_FOLDER,
            CONST.MULTIVARIATE_HAWKES + "_bid_ask_bi",
            training_conf.pair,
            "training_time_" + str(training_time_seconds),
        )
        with open(
            os.path.join(params_dir, CONST.ORDER_OF_EVENT_TYPES_FILE), "w"
        ) as file:
            file.writelines(f"{item}\n" for item in events_conf.events_to_compute)

        df = pd.DataFrame(training_time_file_likelihood_map[training_time_seconds])
        df.to_csv(
            os.path.join(params_dir, CONST.LIKELIHOODS_FILE), sep="\t", index=False
        )
