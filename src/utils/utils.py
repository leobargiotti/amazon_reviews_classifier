import pandas as pd
import nltk


def calculate_training_test(train, test, config_file):
    """
    Method to compute training and test set
    :param train: training dataframe
    :param test: test dataframe
    :param config_file: configuration file
    :return: list containing train-test split
    """
    return train[config_file.column_text], train[config_file.column_target], \
        test[config_file.column_text], test[config_file.column_target]


def calculate_train_test_classes(train, test, config_file):
    """
    Method to compute training, test set and dictionary of the classes
    :param train: training dataframe
    :param test: test dataframe
    :param config_file: configuration file
    :return: list containing train-test split and dictionary of the classes
    """
    X_train, y_train, X_test, y_test = calculate_training_test(train, test, config_file)
    return X_train, X_test, y_train, y_test, create_dictionary_classes(config_file.int_classes, config_file.name_classes)


def create_dictionary_classes(int_classes, name_classes):
    """
    Method to compute dictionary of the classes
    :param int_classes: list of integer value of classes
    :param name_classes: list of strings value of classes
    :return: dictionary that associates integer to string value of classes
    """
    return create_dictionary([int(x) for x in int_classes.split(",")], [x.strip() for x in name_classes.split(",")])


def create_dictionary(keys, values):
    """
    Method compute dictionary
    :param keys: array to of keys
    :param values: array to associate value at each key
    :return: dictionary
    """
    return dict(zip(keys, values))


def drop_duplicates_and_nan(data, column_text, column_target):
    """
    Method to remove duplicates and Nan values
    :param data: dataframe to remove duplicates and Nan values
    :param column_text: string of column name that contains text
    :param column_target: string of column name that contains classes
    :return: dataframe without duplicates and Nan values
    """
    return data.drop_duplicates(subset=[column_text, column_target]).dropna()


def remove_duplicates_and_nan_values(config_file):
    """
    Method to remove duplicates and Nan values
    :param config_file: configuration file
    :return: list containing train-test dataframe without duplicates and Nan values
    """
    train_original = pd.read_csv(config_file.path_training)
    test_original = pd.read_csv(config_file.path_test)
    return train_original, drop_duplicates_and_nan(train_original, config_file.column_text, config_file.column_target),\
        test_original, drop_duplicates_and_nan(test_original, config_file.column_text, config_file.column_target)


def download_if_non_existent(res_path, res_name):
    """
    Method to download nltk package only if it is not downloaded
    :param res_path: string local directory of the package
    :param res_name: string name of package
    """
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f'resource {res_path} not found. Downloading now...')
        nltk.download(res_name)


def print_class_and_probability(class_predicted, probability, time):
    """
    Method to print class predicted and its probability
    :param class_predicted: string of the class predicted
    :param probability: float value of the probability
    :return: string of predicted class and its probability
    """
    return "Class predicted is: " + class_predicted \
        + "\nand its probability is: " + str(round(probability * 100, 2)) + "%" + " in " + time + " seconds"
