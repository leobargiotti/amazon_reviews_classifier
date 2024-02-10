import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from wordcloud import WordCloud
from collections import Counter
import string
from yellowbrick.classifier import ClassPredictionError
from matplotlib.colors import LinearSegmentedColormap as lsg


class Statistics:

    def __init__(self, classifier, configFile):
        self.classifier = classifier
        self.classes_array = list(self.classifier.classes.values())
        self.column_text = configFile.column_text
        self.column_target = configFile.column_target

    # ---------DATASET---------

    def class_distribution(self, data_original, data_cleaned):
        """
        Method to display class distribution statistics
        :param data_original: dataframe before removing duplicates and Nan values
        :param data_cleaned: dataframe after removing duplicates and Nan values
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        for index in range(2):
            sns.countplot(x=self.column_target, data=data_original if index == 0 else data_cleaned, ax=axes[index])
            axes[index].set_xticklabels(self.classes_array)
            axes[index].tick_params(axis='x', labelrotation=90)
            axes[index].set_title(
                ('Before' if index == 0 else 'After') + ' removal of tuples with NaN and duplicated values')
        plt.show()

    def calculate_information(self):
        """
        Method to compute information about before and after preprocess:
        first rows, class distribution, duplicates and numbers of row
        :return: list of two strings (information before and after preprocess)
        """
        train_original = self.classifier.train_original
        test_original = self.classifier.test_original
        train_cleaned = self.classifier.train_cleaned
        test_cleaned = self.classifier.test_cleaned

        train_preprocessed = self.classifier.model[0].transform(self.classifier.X_train)
        test_preprocessed = self.classifier.model[0].transform(self.classifier.X_test)

        def print_error_dataset(dataset, error):
            return "There is no " + dataset + " set to calculate " + error + "\n \n"

        def print_first_rows(dataset, X, y=None):
            return "The first rows of " + dataset + f" set are: \n{X[self.column_text].head() if y is None else X.head()} \n \n" + \
                   "The first rows of " + dataset + f" set are: \n{X[self.column_target].head() if y is None else y.head()} \n \n" \
                if X is not None else print_error_dataset(dataset, "first rows")

        def print_class_distribution(dataset, data):
            if data is not None:
                class_distribution = data[self.column_target].value_counts()
                if type(class_distribution.index[0]) is not str:
                    class_distribution.set_axis([class_distribution.index.map(self.classifier.classes)], copy=True)
                return "The class distribution on " + dataset + f" set is \n{class_distribution} \n \n"
            else:
                return print_error_dataset(dataset, "class distribution")

        def print_nan(dataset, data):
            return "The numbers of Nan value on " + dataset + f" set are {data[self.column_text].isna().sum()} \n \n" \
                if data is not None else print_error_dataset(dataset, "Nan value")

        def print_duplicate(dataset, data):
            return "The numbers of duplicate elements on " + dataset \
                   + f" set are {sum(data.duplicated(subset=[self.column_text, self.column_target]))} \n \n" \
                if data is not None else print_error_dataset(dataset, "number of duplicates")

        def print_number_row(dataset, data):
            return f"There are {len(data)} rows in the " + dataset + " set \n \n" \
                if data is not None else print_error_dataset(dataset, "number of rows")

        column0 = "Before Preprocess: \n" + print_first_rows("training", train_original) + \
                  print_first_rows("test", test_original) + print_class_distribution("training", train_original) + \
                  print_class_distribution("test", test_original) + \
                  print_nan("training", train_original) + print_nan("test", test_original) + \
                  print_duplicate("training", train_original) + print_duplicate("test", test_original) + \
                  print_number_row("training", train_original) + print_number_row("test", test_original)

        column1 = "After Preprocess: \n" + print_first_rows("training", train_preprocessed, self.classifier.y_train) + \
                  print_first_rows("test", test_preprocessed, self.classifier.y_test) + \
                  print_class_distribution("training", train_cleaned) + print_class_distribution("test", test_cleaned) + \
                  print_nan("training", train_cleaned) + print_nan("test", test_cleaned) + \
                  print_duplicate("training", train_cleaned) + print_duplicate("test", test_cleaned) + \
                  print_number_row("training", self.classifier.X_train) + print_number_row("test",
                                                                                           self.classifier.X_test)
        return column0, column1

    def wordcloud(self, data, data_text):
        """
        Method to display wordcloud of a specific dataframe
        :param data: dataframe
        :param data_text: string (training or test)
        """
        wordcloud = WordCloud(
            width=800,
            height=800,
            max_words=200,
            mask=None,
            contour_width=0,
            contour_color="black",
            min_font_size=4,
            background_color="black",
            max_font_size=None,
            relative_scaling="auto",
            colormap="viridis").generate_from_frequencies(dict(Counter(
                self.classifier.model[0].transform(data).str.cat(sep=" ").split())))
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.title("Wordcloud on " + data_text + " set")
        plt.show()

    def show_top20(self, data, data_text):
        """
        Method to display the 20 most frequent words of a specific dataframe
        :param data: dataframe
        :param data_text: string (training or test)
        """
        NUM_TOP_WORDS = 20
        pattern = (
            rf"((\w)[{string.punctuation}](?:\B|$)|(?:^|\B)[{string.punctuation}](\w))"
        )
        top_20 = (self.classifier.model[0].transform(data)
                  .str.replace(pattern, r"\2 \3").str.split().explode().value_counts(False)).head(NUM_TOP_WORDS)
        top_20.plot.bar(rot=90, title="Top 20 words in " + data_text + " set", figsize=(12, 7))
        plt.show()

    # ----------CLASSIFIER------------

    def calculate_class_report(self):
        """
        Method to compute classification report
        :return: string about classification report, training and test accuracy
        """
        return classification_report(self.classifier.y_test, self.classifier.model.predict(self.classifier.X_test),
                                     target_names=self.classes_array) + "\n\n" + 'Final Training Accuracy: ' + \
            "%.2f" % (self.classifier.model.score(self.classifier.X_train, self.classifier.y_train) * 100) + '%\n' + \
            'Model Accuracy: ' + "%.2f" % (
                    self.classifier.model.score(self.classifier.X_test, self.classifier.y_test) * 100) + '%'

    def confusion_matrix(self, title):
        """
        Method to display confusion matrix
        :param title: string that is the title of the confusion matrix
        """
        matrix = ConfusionMatrixDisplay.from_estimator(self.classifier.model, self.classifier.X_test,
                                                       self.classifier.y_test, cmap=plt.cm.Blues,
                                                       display_labels=self.classes_array, xticks_rotation="vertical")
        plt.grid(False)
        matrix.ax_.set_title(title)
        plt.show()

    # ------------PRETRAINED CLASSIFIERS--------------

    def class_report(self, limit_row=None):
        """
        Method to compute classification report
        :param limit_row: integer to specify the number row to use to test, if it
        is not specify it is the entire test set
        :return: string about classification report, training and test accuracy
        """
        predicted_labels = []
        if limit_row is None:
            limit_row = len(self.classifier.y_testlimit_row)
        else:
            limit_row = limit_row
        for row in self.classifier.X_test.head(limit_row):
            predicted_labels.append(self.classifier.model(row, candidate_labels=self.classes_array)['labels'][0])
        return classification_report(self.classifier.y_test.head(limit_row),
                                     np.vectorize({val: key for (key, val) in self.classifier.classes.items()}.get)
                                     (predicted_labels), target_names=self.classes_array)

    def cm(self, title, limit_row=None):
        """
        Method to display confusion matrix
        :param title: string that is the title of the confusion matrix
        :param limit_row: integer to specify the number row to use to test, if it
        is not specify it is the entire test set
        """
        if limit_row is None:
            limit_row = len(self.classifier.y_testlimit_row)
        else:
            limit_row = limit_row
        predicted_labels = []
        for row in self.classifier.X_test.head(limit_row):
            predicted_labels.append(self.classifier.model(row, candidate_labels=self.classes_array)['labels'][0])
        predicted_label = np.vectorize({val: key for (key, val) in self.classifier.classes.items()}.get)(
            predicted_labels)
        true_label = self.classifier.y_test.head(limit_row)
        labels = sorted(set(true_label) | set(predicted_label))
        plt.figure(figsize=(10, 7))
        ax = plt.subplot()
        sns.heatmap(pd.DataFrame(confusion_matrix(true_label, predicted_label), index=labels, columns=labels),
                    annot=True, cmap="Blues", fmt='d')
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        ax.xaxis.set_ticklabels(self.classes_array)
        ax.yaxis.set_ticklabels(self.classes_array)
        ax.set_title(title)
        plt.show()
