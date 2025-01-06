import numpy as np
import os
import tensorflow as tf
from PIL import Image
import time

IMG_INP_SHAPE = (300, 300)
CHANNELS = 3
LABELS = ('average', 'bad', 'good')


def load_image(img, inp_shape=IMG_INP_SHAPE):
    with Image.open(img) as _img:
        image = _img.resize(inp_shape)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
    return image


def load_models(groceries):
    models = {}
    for grocery_type in groceries:
        model_path = os.path.join('models', grocery_type)
        model = tf.lite.Interpreter(model_path=model_path)
        models.update({grocery_type: model})
    return models


class ImageClassifier(object):
    """
    Classifier class to read the .tflite model and predict the classes
    """

    def __init__(self, model_path, image_size=IMG_INP_SHAPE[0]):
        """
        :param model_path: Path of the model file (string)
        :param image_size: input size of the image (tuple); Sample - (224, 224)
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = ("average", "bad", "good")
        self.image_size = image_size

    def run(self, image):
        """
        :param image: a (1, image_size, image_size, 3) np.array
        Returns list of [Label, Probability], of type List<str, float>
        """
        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(
            self._output_details[0]["index"])
        probabilities = np.float32(tflite_interpreter_output[0])
        probabilities = tf.nn.softmax(probabilities)
        # create list of ["label", probability]
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        return label_to_probabilities


def predict(model_path, image):
    """
    :param model_path: Path of the model file (string)
    :param image_size: input size of the image (tuple); Sample - (224, 224)
    """
    classifier = ImageClassifier(model_path, image_size=IMG_INP_SHAPE)
    start_time = time.time()
    prediction = classifier.run(image)
    end_time = time.time()
    print("Time taken to predict using tflite model is {} seconds".format(
        end_time - start_time))
    # Create a dictionary of labels and predictions
    prediction = dict(prediction)
    for key, value in prediction.items():
        # Convert values to list
        # Multiply by 100 to infer probability as accuracy
        prediction[key] = [value * 100]
    return prediction
