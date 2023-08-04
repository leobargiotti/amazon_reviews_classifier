from transformers import pipeline
import time

from .Classifier import Classifier
from configuration.ConfigFile import ConfigFile
from utils.utils import calculate_train_test_classes, remove_duplicates_and_nan_values, print_class_and_probability


class ClassifierDeberta(Classifier):

    def __init__(self):
        self.model = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        self.config_file = ConfigFile()
        self.train_original, self.train_cleaned, self.test_original, self.test_cleaned = remove_duplicates_and_nan_values(
            self.config_file)
        self.X_train, self.X_test, self.y_train, self.y_test, self.classes = calculate_train_test_classes(
            self.train_cleaned, self.test_cleaned, self.config_file)

    def calculate_class(self, text):
        t0 = time.time()
        result = self.model(text, self.config_file.name_classes)
        return print_class_and_probability(result["labels"][0], result["scores"][0], str(round(time.time() - t0, 3)))
