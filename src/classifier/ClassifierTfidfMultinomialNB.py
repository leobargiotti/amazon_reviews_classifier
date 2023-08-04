from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .Classifier import Classifier


class ClassifierTfidfMultinomialNB(Classifier):

    def __init__(self):
        Classifier.__init__(self, "../models_saved/TfidfMultinomialNB.pk")

    def build_pipeline(self):
        """
        Method to instance Pipeline with classifier
        :return: Pipeline with TextPreprocessor, TfidfVectorizer and MultinomialNB
        """
        return Pipeline(super().build_pipeline().steps.__add__(
            [('multinomial', GridSearchCV(MultinomialNB(),
                                          param_grid={'alpha': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.05, 0.01,
                                                                0.001, 0.0001, 0.00001]},
                                          cv=StratifiedKFold(n_splits=10), scoring='accuracy', verbose=3))]))
