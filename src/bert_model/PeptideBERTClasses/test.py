# Available backend options are: "jax", "torch", "tensorflow".
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras

model = keras.saving.load_model("hf://GrimSqueaker/proteinBERT")

print(1)