from configparser import ConfigParser
from pathlib import Path

from utils.utils import create_dictionary


class ConfigFile:
    config_file = "../config.ini"
    config_dataset = "DATASET"
    keys_dataset = ["path_training", "path_test", "column_text", "column_target", "language", "int_classes", "name_classes"]
    dataset_default = ["../dataset/Product_Reviews/new/train.csv", "../dataset/Product_Reviews/new&test.csv", 'Text_Complete',
                       'Class', "english", "1, 2", "Negative, Positive"]

    def __init__(self):
        self.config_object = ConfigParser()
        if not self.is_present_config_file(): self.create_default_config_file()
        self.path_training, self.path_test, self.column_text, self.column_target, self.language, \
            self.int_classes, self.name_classes = self.read_all_attributes_section(self.config_dataset, self.keys_dataset)

    def create_default_config_file(self):
        """
        Method to write default configuration file
        """
        self.config_object[self.config_dataset] = create_dictionary(self.keys_dataset, self.dataset_default)
        self.write_config_file()

    def is_present_config_file(self):
        """
        Method to compute if configuration file exists
        :return: boolean value
        """
        return True if Path(self.config_file).is_file() else False

    def read_all_attributes_section(self, section, attributes):
        """
        Method to read all attributes of the section
        :param section: String of the section
        :param attributes: Array of the attributes of the section
        :return: array
        """
        return [self.read_attribute(section, attribute) for attribute in attributes]

    def read_attribute(self, section, attribute):
        """
        Method to read one attribute
        :param section: String of the section
        :param attribute: String of the attribute
        :return: string of read value
        """
        self.config_object.read(self.config_file)
        return self.config_object.get(section, attribute)

    def write_config_file(self):
        """
        Method to write configuration file
        """
        self.config_object.write(open(self.config_file, 'w'))
