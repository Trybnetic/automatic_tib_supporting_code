"""
This script trains the keras models
"""
import random
import time 
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import keras
from keras import Sequential
from keras.layers import Embedding, Masking, LSTM, Dense, Input
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping

from actiaug import Crop, Flip, Reverse, AddNoise


def normalize(x, m, s):
    return (x - m) / s

def calculate_enmo(data):
    enmo = np.sqrt(np.sum(np.square(data[["Y", "X", "Z"]].values), axis=1)) - 1
    enmo = enmo.clip(min=0)
  
    return enmo


def horizontal_angle(data):
    return np.arctan(np.sqrt(data["X"]**2 + data["Z"]**2) / data["Y"])


def load(files, n_samples=5, train=False, return_full_ts=False):
    res = []
    for file in tqdm(files, desc="Load data"):  
        data = pd.read_csv(file, index_col=0)
        data["ENMO"] = calculate_enmo(data[["Y", "X", "Z"]])
        data["HorAngle"] = horizontal_angle(data[["Y", "X", "Z"]])

        # x_all = np.expand_dims(data[["Y", "X", "Z"]].values, axis=0)
        x_all = np.expand_dims(data[["ENMO", "HorAngle"]].values, axis=0)
        y_all = np.expand_dims(data[["is_in_bed"]].values, axis=0)

        if return_full_ts:
            x, y = x_all, y_all
            length = y.shape[1]
            res += [(x, y, length)]
        else:
            slices = []
            for _ in range(n_samples):
                if train:
                    # Crop random slice between 5 and 300min
                    #
                    # Change augmentation here if you want a different augmentation strategy
                    ts_len = random.sample(range(5, 300), 1)[0]
                    augmenter = (
                        Crop(size=ts_len) + 
                        Flip() @ 0.5 + # Simulate wrongly worn sensor
                        Reverse() @ 0.5 + # with 50% probability, reverse the sequence
                        AddNoise(scale=0.01) @ 0.1 # Add noise to prevent overfitting
                    )
                else:
                    # Crop random slice between 5min and all time
                    ts_len = random.sample(range(5, x_all.shape[1]), 1)[0]
                    augmenter = Crop(size=ts_len)

                # Augment data
                x, y = augmenter.augment(x_all, y_all)
                length = y.shape[1]
                res += [(x, y, length)]

    x, y, lens = list(zip(*res))

    n_samples = len(x)
    seq_len = max(lens)
    x_dims = x_all.shape[2]

    x_out = np.zeros((n_samples, seq_len, x_dims))
    mask = np.zeros((n_samples, seq_len), dtype=bool)
    y_out = np.zeros((n_samples, seq_len))            

    for jj in range(len(x)):
        x_out[jj,:lens[jj], :] = x[jj].squeeze()
        mask[jj,:lens[jj]] = True
        y_out[jj,:lens[jj]] = y[jj].squeeze()

    return x_out, np.expand_dims(y_out, axis=2), mask

random.seed(1)
np.random.seed(1)
keras.utils.set_random_seed(1)

# Define conditions 
hidden_sizes = [1,2,4,8]
n_layers = [1,2,4]
datasets = [
    ("tu7_tib", "1min_data/tib_dataset"),
    ("tu7_accelerometer", "1min_data/accelerometer_dataset")
]

early_stopping = EarlyStopping(patience=2,verbose=1)


for dataset, base_path in datasets:
    files = np.array([os.path.join(base_path, file) for file in os.listdir(base_path)])
    
    # Calculate mean and std for normalization
    start = time.time()
    x, _, mask = load(files, return_full_ts=True)
    means, stds = x[mask].mean(axis=0), x[mask].std(axis=0)
    print(f"Calculated means {means} and stds {stds} in {(time.time() - start)/60:.2f}min.")
    del x, mask, start


    with open(f"results.csv", "w") as f:
        f.write(f'dataset,fold,n_layers,hidden_size,test_loss,test_accuracy,full_loss,full_accuracy\n')

    if not os.path.exists(f"models/{dataset}/"):
        os.makedirs(f"models/{dataset}/")

    kf = KFold(n_splits=10)
    for ii, (train_idx, test_idx) in enumerate(kf.split(files)):
        print(f"Start processing fold {ii+1}")
        train_files, valid_files = train_test_split(files[train_idx], train_size=.8)
        test_files = files[test_idx]

        start = time.time()
        # Train data
        x_train, y_train, mask_train = load(train_files, train=True, n_samples=100)
        x_train[mask_train] = normalize(x_train[mask_train], means, stds)

        # Validation data
        x_valid, y_valid, mask_valid = load(valid_files, train=False, n_samples=30)
        x_valid[mask_valid] = normalize(x_valid[mask_valid], means, stds)

        # Test data
        x_test, y_test, mask_test = load(test_files, train=False, n_samples=30)
        x_test_full, y_test_full, mask_test_full = load(test_files, return_full_ts=True)
        x_test[mask_test] = normalize(x_test[mask_test], means, stds)
        print(f"Loaded data in {time.time() - start:.2f}s.")

        for hidden_size in hidden_sizes:
            for n_layer in n_layers:

                # Prep model
                model = Sequential()
                model.add(Input(shape=(None, x_train.shape[2])))
                model.add(Masking(mask_value=0.))
                for _ in range(n_layer):
                    model.add(LSTM(hidden_size, dropout=.5, return_sequences=True))
                model.add(Dense(1, activation="sigmoid"))
                model.compile(
                    optimizer=Adam(1e-3), 
                    loss="binary_crossentropy",
                    metrics=["binary_accuracy"]
                    )

                history = model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=8,
                    epochs=100,
                    callbacks=[early_stopping],
                    validation_data=(x_valid, y_valid),
                    shuffle=True
                )

                # Evaluate model
                test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
                print(f"Results on sampled test set: loss = {test_loss:.2f}, accuracy = {test_accuracy:.2f}")
                full_loss, full_accuracy = model.evaluate(x=x_test_full, y=y_test_full, verbose=0)
                print(f"Results on full test set: loss = {full_loss:.2f}, accuracy = {full_accuracy:.2f}")

                with open(f"results.csv", "a") as f:
                    f.write(f'{dataset},{ii},{n_layer},{hidden_size},{test_loss},{test_accuracy},{full_loss},{full_accuracy}\n')

                model.save(f'models/{dataset}/lstm_{n_layer}_{hidden_size}.keras')

