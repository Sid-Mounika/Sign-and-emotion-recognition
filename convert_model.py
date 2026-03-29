import tensorflow as tf

# Load OLD model (trained with Keras 2.x)
old_model = tf.keras.models.load_model(
    "facialemotionmodel.h5",
    compile=False
)

# Save in NEW Keras 3 format
old_model.save("facialemotionmodel.keras")

print("✅ Model converted successfully to new format")
