import numpy as np
import os

DATA_DIR = "sign_dataset"

signs = ["hello", "yes", "no", "thank you", "welcome"]

X_list = []
y_list = []

for sign in signs:
    X = np.load(os.path.join(DATA_DIR, f"X_{sign}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{sign}.npy"))

    X_list.append(X)
    y_list.append(y)

X = np.vstack(X_list)
y = np.concatenate(y_list)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

np.save(os.path.join(DATA_DIR, "X_all.npy"), X)
np.save(os.path.join(DATA_DIR, "y_all.npy"), y)
