import pathlib
import tensorflow as tf
from tensorflow.keras import layers, Sequential


class FigureModel:
    def __init__(self, data_dir, model_save_path):



        self.data_dir = pathlib.Path(data_dir)
        self.model_save_path = model_save_path
        self.model = None
        self.class_names = None
        self.train_ds = None
        self.val_ds = None
        self.history = None

    def load_datasets(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(180,180),
            batch_size=32
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(180,180),
            batch_size=32
        )

        self.class_names = self.train_ds.class_names

    def create_model(self):
        num_classes = len(self.class_names)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(180, 180, 3),
            include_top=False,
            weights='imagenet')

        base_model.trainable = False
        self.model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])


    def train_model(self, epochs=10):
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs)
    def save_model(self):
        self.model.save(self.model_save_path)


if __name__ == "__main__":
    data_dir = r"D:\python\pythonProject2\train"
    model_save_path = "figure_model.h5"

    figure_model = FigureModel(data_dir, model_save_path)
    figure_model.load_datasets()
    figure_model.create_model()
    figure_model.train_model(epochs=10)
    figure_model.save_model()
    print("Model training complete and saved.")
