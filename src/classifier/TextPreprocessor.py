import contractions
from sklearn.base import BaseEstimator, TransformerMixin
from simplemma import lemmatize
import pycountry
import nltk
import re
import unicodedata
from nltk.stem import SnowballStemmer


class PreprocessingSteps:
    def __init__(self, X, language):
        self.X = X
        self.language = language
        self.stopwords = set(nltk.corpus.stopwords.words(self.language))
        self.stopwords.remove("not")

    def expanding_contractions(self):
        """
        Method to expand english contractions
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: contractions.fix(x)) if self.language == "english" else self.X
        return self

    def remove_html_tags(self):
        """
        Method to remove html tags
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: re.sub(r'<.*?>', '', x))
        return self

    def remove_urls(self):
        """
        Method to remove urls
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: re.sub(r'http\S+', '', x))
        return self

    def remove_punctuation(self):
        """
        Method to remove punctuation
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: re.sub(r'\W+', ' ', x))
        return self

    def remove_diacritics(self):
        """
        Method to remove diacritics
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))
        return self

    def lowercase(self):
        """
        Method to lowercase
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: x.lower())
        return self

    def remove_digits(self):
        """
        Method to remove digits
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
        return self

    def remove_extra_whitespace(self):
        """
        Method to remove extra whitespace
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: ' '.join(x.split()))
        return self

    def remove_stopwords(self):
        """
        Method to remove stopwords
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(
            lambda x: ' '.join([word for word in nltk.word_tokenize(str(x)) if word not in self.stopwords]))
        return self

    def stem(self):
        """
        Method to perform stemming
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: ' '.join([SnowballStemmer(self.language).stem(word) for word in nltk.word_tokenize(str(x))]))
        return self

    def lemma(self):
        """
        Method to perform lemmatization
        :return: itself (PreprocessingSteps)
        """
        self.X = self.X.apply(lambda x: ' '.join([lemmatize(str(word), pycountry.languages.get(name=self.language).alpha_2)
                                                  for word in nltk.word_tokenize(str(x))]))
        return self

    def get_processed_text(self):
        """
        Method to get preprocessed text
        :return: preprocessed text
        """
        return self.X


class TextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, language):
        self.language = language

    def fit(self, X, y=None):
        """
        Method to preprocess text
        """
        return self

    def transform(self, X, y=None):
        """
        Method to preprocess text
        """
        return PreprocessingSteps(X.copy(), self.language).expanding_contractions().remove_urls().remove_digits()\
            .remove_html_tags().remove_punctuation().remove_diacritics().lowercase().remove_stopwords()\
            .remove_extra_whitespace().stem().lemma().lowercase().get_processed_text()
