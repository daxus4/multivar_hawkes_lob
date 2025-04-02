# multivariate_hawkes_lob

This repository contains the code used to collect Limit Order Book data from Bitfinex, convert them into a dataframe and using them to train Multivariate Hawkes Processes.

## Usage

You have to create two conda enviroment, one for collecting LOB data (using the file requirements_bitfinex.txt) and one for training and simulate Multivariate Hawkes Processes (using environment_hawkes.yml).

To collect LOB data use the script collect_lob_data.py and then convert_lob_data.py to have them in a dataframe format.

With the script train_multivariate_hawkes_with_greedy_b_all_training_periods_bi.py you can train Multivariate Hawkes Processes with greedy algorithm.

Finally, use predict_mid_price_events.py to simulate events.