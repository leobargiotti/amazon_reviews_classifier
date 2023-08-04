import os

from gui.WindowHome import WindowHome
from utils.utils import download_if_non_existent
from classifier.ClassifierTfidfMultinomialNB import ClassifierTfidfMultinomialNB
from classifier.ClassifierTfidfLogReg import ClassifierTfidfLogReg
from classifier.ClassifierBART import ClassifierBART
from classifier.ClassifierDeberta import ClassifierDeberta

if __name__ == '__main__':
    """
    Main to open GUI application of News Classifier specifying classifiers and their names
    Application reads preferences from file 'config.ini'
    """

    download_if_non_existent('corpora/stopwords', 'stopwords')
    download_if_non_existent('tokenizers/punkt', 'punkt')

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = WindowHome([ClassifierTfidfMultinomialNB(),  ClassifierTfidfLogReg(), ClassifierBART(), ClassifierDeberta()],
                     ["TfidfVectorizer\n- MultinomialNB", "TfidfVectorizer\n- LogisticRegression", "Bart", "Deberta"])

    app.mainloop()
