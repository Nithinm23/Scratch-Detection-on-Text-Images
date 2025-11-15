import tensorflow as tf
from tensorflow import keras

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = 2.0 * tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - intersection / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def build_model(input_shape=(256,256,3)):
    base = keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet",
        input_shape=input_shape, pooling=None
    )

    pooled = keras.layers.GlobalAveragePooling2D()(base.output)
    class_out = keras.layers.Dense(1, activation="sigmoid", name="class_output")(pooled)
    scratch_out = keras.layers.Dense(1, activation="linear", name="scratch_output")(pooled)

    x = base.output
    for filters in [512,256,128,64,32,16]:
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.UpSampling2D()(x)

    mask_out = keras.layers.Conv2D(1, 1, activation="sigmoid", name="mask_output")(x)

    model = keras.Model(inputs=base.input,
                        outputs=[class_out, scratch_out, mask_out])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={
            "class_output": "binary_crossentropy",
            "scratch_output": "mse",
            "mask_output": bce_dice
        },
        loss_weights={
            "class_output": 1.0,
            "scratch_output": 0.3,
            "mask_output": 2.0
        },
        metrics={"class_output": "accuracy"}
    )

    return model
