import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf




class FigurePredictorApp:
    def __init__(self,root, model_path):
        self.root = root
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['circle', 'square', 'triangle']

        self.root.title("Figure Classifier")
        self.root.geometry("500x500")

        self.label = tk.Label(self.root, text="Upload an image to classify", font=("Arial", 16))
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Classify", command=self.predict_image)
        self.predict_button.pack(pady=10)

        self.image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = img.resize((300,300))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(150,150,image=img_tk)
            self.canvas.image = img_tk
            self.result_label.config(text="")

    def preprocess_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0
        return img_array

    def predict_image(self):
        if not self.image_path:
            self.result_label.config(text="Please upload an image first!")
            return

        img_array = self.preprocess_image(self.image_path)
        predictions = self.model.predict(img_array)
        predicted_label = np.argmax([predictions[0]])
        predicted_class = self.class_names[predicted_label]

        self.result_label.config(text=f"Predictions: {predicted_class}")

if __name__ == "__main__":
    model_path = "figure_model.h5"
    root = tk.Tk()
    app = FigurePredictorApp(root, model_path)
    root.mainloop()
