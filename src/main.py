import os

from gui.WindowHome import WindowHome
from classifier.ClassifierTfidfLogReg import ClassifierTfidfLogReg
from classifier.ClassifierDeberta import ClassifierDeberta
from utils.utils import download_if_non_existent

if __name__ == '__main__':
    """
    Main to open GUI application of News Classifier specifying classifiers and their names
    Application reads preferences from file 'config.ini'
    """

    download_if_non_existent('corpora/stopwords', 'stopwords')
    download_if_non_existent('tokenizers/punkt', 'punkt')

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = WindowHome([ClassifierTfidfLogReg(), ClassifierDeberta()],
                     ["TfidfVectorizer\n- LogisticRegression", "Deberta"])
    app.mainloop()
