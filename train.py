"""
This script trains the keras models
"""
import random
import time 
import datetime
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import keras
from keras import Sequential
from keras.layers import Embedding, Masking, LSTM, Dense, Input, Bidirectional
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping, TensorBoard

from actiaug import Crop, Flip, Reverse, AddNoise


def normalize(x, m, s):
    return (x - m) / s

def calculate_enmo(data):
    enmo = np.sqrt(np.sum(np.square(data[["Y", "X", "Z"]].values), axis=1)) - 1
    enmo = enmo.clip(min=0)
  
    return enmo


def horizontal_angle(data):
    return np.abs(np.arctan(np.sqrt(data["X"]**2 + data["Z"]**2) / data["Y"]))


def load(files, n_samples=5, train=False, return_full_ts=False, desc="Load data"):
    res = []

    for file in tqdm(files, desc=desc):  
        data = pd.read_csv(file, index_col=0)
        data["ENMO"] = calculate_enmo(data[["Y", "X", "Z"]])
        data["HorAngle"] = horizontal_angle(data[["Y", "X", "Z"]])

        x_all = np.expand_dims(data[["Y", "X", "Z", "ENMO", "HorAngle"]].values, axis=0)
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

# Remove data from older runs
os.system("rm -r logs/")
os.system("rm -r models/")

# Define conditions 
hidden_sizes = [1, 2, 4, 8, 16, 32]
# hidden_sizes = [32] 
n_layers = [1, 2, 4]
# n_layers = [4] 
datasets = [
    ("tu7_tib", "1min_data/tib_dataset"),
    # ("tu7_accelerometer", "1min_data/accelerometer_dataset")
]
result_path = "results.csv"

training_conditions = {
    "Y_X_Z": [0, 1, 2],
    "ENMO_HorAngle": [3, 4],
    "Y_X_Z_HorAngle": [0, 1, 2, 4],
    "Y_X_Z_ENMO_HorAngle": [0, 1, 2, 3, 4]
}

# Create new files and folders
with open(result_path, "w") as f:
    f.write(f'dataset,train_cols,fold,n_layers,hidden_size,test_loss,test_accuracy,test_precision,test_recall,full_loss,full_accuracy,full_precision,full_recall,model\n')

total_start = time.time()

for dataset, base_path in datasets:
    print(f"Start training with {dataset}")

    files = np.array([os.path.join(base_path, file) for file in os.listdir(base_path)])

    # Calculate mean and std for normalization
    start = time.time()
    x, y, mask = load(files, return_full_ts=True)
    means, stds = x[mask].mean(axis=0), x[mask].std(axis=0)
    label_balance = y[mask].mean() * 100
    print(f"Calculated means {means} and stds {stds} in {(time.time() - start)/60:.2f}min.")
    print(f"{label_balance:.0f}%  of all data is time in bed")
    del x, mask, start

    kf = KFold(n_splits=10)
    for ii, (train_idx, test_idx) in enumerate(kf.split(files)):
        
        print(f"Start processing fold {ii+1:02d}")
        print("========================")
        train_files, valid_files = train_test_split(files[train_idx], train_size=.8)
        test_files = files[test_idx]

        start = time.time()
        # Train data
        x_train, y_train, mask_train = load(train_files, train=True, n_samples=100, desc="Load train data")
        x_train[mask_train] = normalize(x_train[mask_train], means, stds)
        label_balance = y_train[mask_train].mean() * 100
        print(f"{label_balance:.0f}%  of the sampled training data is time in bed")

        # Validation data
        x_valid, y_valid, mask_valid = load(valid_files, train=False, n_samples=30, desc="Load validation data")
        x_valid[mask_valid] = normalize(x_valid[mask_valid], means, stds)
        label_balance = y_valid[mask_valid].mean() * 100
        print(f"{label_balance:.0f}%  of the sampled validation data is time in bed")

        # Test data
        x_test, y_test, mask_test = load(test_files, train=False, n_samples=30, desc="Load test data")
        x_test[mask_test] = normalize(x_test[mask_test], means, stds)
        label_balance = y_test[mask_test].mean() * 100
        print(f"{label_balance:.0f}%  of the sampled test data is time in bed")
        x_test_full, y_test_full, mask_test_full = load(test_files, train=False, return_full_ts=True, desc="Load full data")
        x_test_full[mask_test_full] = normalize(x_test_full[mask_test_full], means, stds)
        print(f"Loaded data in {time.time() - start:.2f}s.")

        for col_dir, train_cols in training_conditions.items():
            for hidden_size in hidden_sizes:
                for n_layer in n_layers:

                    print(f"{col_dir} columns: Start training with {n_layer} layer and a hidden size of {hidden_size}.")
                    
                    log_dir = f"logs/{dataset}/{col_dir}/lstm_{n_layer}_{hidden_size:02d}_{ii}"
                    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
                    early_stopping = EarlyStopping(patience=2, verbose=1, restore_best_weights=True)

                    # Prep model
                    model = Sequential()
                    model.add(Input(shape=(None, x_train.shape[2])))
                    model.add(Masking(mask_value=0.)) #, input_shape=(None, x_train.shape[2])))
                    for _ in range(n_layer):
                        model.add(Bidirectional(LSTM(
                            hidden_size, 
                            dropout=.5, 
                            return_sequences=True
                        )))
                    model.add(Dense(1, activation="sigmoid"))
                    model.compile(
                        # optimizer=Adam(1e-3), 
                        optimizer=SGD(), 
                        loss="binary_focal_crossentropy", #"binary_crossentropy",
                        metrics=["binary_accuracy", "precision", "recall"]
                    )

                    history = model.fit(
                        x=x_train[:,:,train_cols],
                        y=y_train,
                        batch_size=64,
                        epochs=100,
                        callbacks=[early_stopping, tensorboard_callback],
                        validation_data=(x_valid[:,:,train_cols], y_valid),
                        shuffle=True
                    )

                    # Evaluate model
                    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x=x_test[:,:,train_cols], y=y_test, verbose=0)
                    print(f"Results on sampled test set: loss = {test_loss:.2f}, accuracy = {test_accuracy:.2f}, precision = {test_precision:.2f}, recall = {test_recall:.2f}")
                    full_loss, full_accuracy, full_precision, full_recall= model.evaluate(x=x_test_full[:,:,train_cols], y=y_test_full, verbose=0)
                    print(f"Results on full test set: loss = {full_loss:.2f}, accuracy = {full_accuracy:.2f}, precision = {full_precision:.2f}, recall = {full_recall:.2f}")

                    model_dir = f"models/{dataset}/{col_dir}/"
                    os.makedirs(model_dir, exist_ok=True)

                    model_path = os.path.join(model_dir, f'lstm_{n_layer}_{hidden_size:02d}_{ii}.keras')
                    with open(result_path, "a") as f:
                        f.write(f'{dataset},"{col_dir}",{ii},{n_layer},{hidden_size},{test_loss},{test_accuracy},{test_precision},{test_recall},{full_loss},{full_accuracy},{full_precision},{full_recall},{model_path}\n')

                    model.save(model_path)

        ellapsed_time = (time.time() - start) / 60
        print(f"Processed fold {ii+1:02d} in {ellapsed_time:.2f}min.")
        print("")

ellapsed_time = (time.time() - total_start) / 3600
print(f"Whole training procedure took {ellapsed_time:.2f}h.")
print("")