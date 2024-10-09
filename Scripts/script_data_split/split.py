import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
import json


def max_pooling(row, spectrum, interval=5):
    value = row[spectrum]
    arr_wave = np.array(value)
    arr_pool = block_reduce(arr_wave, interval, np.max)

    return arr_pool


def run_split(full_df, config, int_inter):

    train_df, test_df = train_test_split(full_df, test_size=config['split']['test_size'],
                                         random_state=42)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    train_df['IR_SPECTRUM_POOLED'] = train_df.apply(max_pooling, spectrum="IR_SPECTRUM",
                                                       interval=int_inter, axis=1)

    print("Train ended")

    test_df['IR_SPECTRUM_POOLED'] = test_df.apply(max_pooling, spectrum="IR_SPECTRUM",
                                                     interval=int_inter, axis=1)

    print("Test ended")
    return train_df, test_df


def add_peak_number(dtf_in, type_pred, **kwargs):
    dtf_out = dtf_in.copy()
    dtf_out['IR_PEAK_NUM'] = dtf_out['IR_SPECTRUM'].apply(lambda x: len([c for c in x if c > 1e-6]))
    return dtf_out


if __name__ == '__main__':
    with open("Config/config_no_conv_IR_inter_6.json") as file:
        config = json.load(file)

    full_df = pd.read_pickle(fr"data\raw\{config['starting_dtf']}.pickle")

    int_inter = config.get('split_f', 1)

    train_df, test_df = run_split(full_df, config, int_inter)
    train_df = add_peak_number(train_df, config['type_pred'], **config.get('source_peak_data', {}))
    test_df = add_peak_number(test_df, config['type_pred'], **config.get('source_peak_data', {}))

    validation_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    pd.to_pickle(train_df, rf'data\raw\train_{config["starting_dtf"]}.pickle')
    pd.to_pickle(validation_df, rf'data\raw\validation_{config["starting_dtf"]}.pickle')
    pd.to_pickle(test_df, rf'data\raw\test_{config["starting_dtf"]}.pickle')

    print('Dataset Split Saved!')