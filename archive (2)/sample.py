import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Config
# -------------------------
BASE_DIR = r"C:\Users\HP\Desktop\dog breed identification project\archive (2)\images\Images"
MODEL_PATH = "dog_breed_model.keras"
LABELS_PATH = "dog_breed_labels.json"
TEST_IMAGE = r"C:\Users\HP\Desktop\dog breed identification project\test_dog.jpg"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
HEAD_EPOCHS = 6
FINE_TUNE_EPOCHS = 8
FORCE_RETRAIN = True  # set False to load saved model

model = None
class_names = None


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def check_paths():
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {BASE_DIR}")


def load_data():
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=VAL_SPLIT
    )
    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VAL_SPLIT
    )

    train_data = train_gen.flow_from_directory(
        BASE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", seed=SEED, shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        BASE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", seed=SEED, shuffle=False
    )
    return train_data, val_data


def save_labels(class_indices):
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)


def load_labels():
    if not os.path.exists(LABELS_PATH):
        return None
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def compute_class_weights(train_data):
    classes, counts = np.unique(train_data.classes, return_counts=True)
    total = counts.sum()
    n = len(classes)
    return {int(c): float(total / (n * count)) for c, count in zip(classes, counts)}


def build_model(num_classes):
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False

    inp = layers.Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    net = models.Model(inp, out)
    return net, base


def compile_model(net, lr):
    net.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )


def callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
    ]


def train_or_load():
    global model, class_names

    train_data, val_data = load_data()
    num_classes = len(train_data.class_indices)
    save_labels(train_data.class_indices)
    class_names = {v: k for k, v in train_data.class_indices.items()}

    if os.path.exists(MODEL_PATH) and not FORCE_RETRAIN:
        model = load_model(MODEL_PATH)
        labels = load_labels()
        if labels is not None:
            class_names = labels
        return

    class_weights = compute_class_weights(train_data)
    model, base = build_model(num_classes)

    compile_model(model, 1e-3)
    h1 = model.fit(
        train_data, validation_data=val_data, epochs=HEAD_EPOCHS,
        class_weight=class_weights, callbacks=callbacks(), verbose=1
    )

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, 1e-5)
    h2 = model.fit(
        train_data, validation_data=val_data,
        initial_epoch=h1.epoch[-1] + 1, epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
        class_weight=class_weights, callbacks=callbacks(), verbose=1
    )

    model = load_model(MODEL_PATH)
    loss, acc, top3 = model.evaluate(val_data, verbose=0)
    print(f"Val Accuracy: {acc*100:.2f}% | Top-3: {top3*100:.2f}%")

    acc_hist = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc_hist = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    plt.plot(acc_hist, label="train")
    plt.plot(val_acc_hist, label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.show()


def clean_name(raw):
    return " ".join(raw.split("-")[1:]).replace("_", " ").title()


def predict_dog_breed(img_path):
    if not os.path.exists(img_path):
        return "Image not found."
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    top = np.argsort(preds)[-3:][::-1]
    lines = [f"{i+1}. {clean_name(class_names[idx])}: {preds[idx]*100:.2f}%" for i, idx in enumerate(top)]
    return "\n".join(lines)


def predict_gradio(img):
    if img is None:
        return "Upload an image."
    arr = img_to_array(img)
    arr = tf.image.resize(arr, IMG_SIZE).numpy()
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    top = np.argsort(preds)[-3:][::-1]
    return "\n".join([f"{i+1}. {clean_name(class_names[idx])}: {preds[idx]*100:.2f}%" for i, idx in enumerate(top)])


def launch_web_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Dog Breed Classifier")
        with gr.Row():
            img_input = gr.Image(type="pil", label="Upload dog photo")
            output = gr.Textbox(label="Top 3 Predictions", lines=6)
        gr.Button("Identify").click(predict_gradio, inputs=img_input, outputs=output)
    demo.launch(share=True)


if __name__ == "__main__":
    set_seed(SEED)
    check_paths()
    train_or_load()

    print("\nQuick test:")
    print(predict_dog_breed(TEST_IMAGE))

    if input("\nLaunch web app? (y/n): ").strip().lower() == "y":
        launch_web_app()
