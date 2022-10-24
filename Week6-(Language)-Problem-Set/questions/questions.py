from math import log
import nltk
from nltk import word_tokenize as wtokenized
from string import punctuation
import sys
import os

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    txtfiles = os.listdir(directory)
    istextfile = lambda name : name[-4:] == '.txt'
    txtfiles = [filename for filename in txtfiles if istextfile(filename)]
    del istextfile
    files = dict.fromkeys(txtfiles)
    for txtfile in txtfiles:
        with open(os.path.join(directory, txtfile), encoding='utf-8') as content:
            files[txtfile] = content.read()
    return files

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopwords = nltk.corpus.stopwords.words("english")
    def clean_lowered(string):
        cleaned = [char for char in string if char not in punctuation]
        cleaned = ''.join(cleaned)
        return cleaned.lower()

    words = wtokenized(clean_lowered(document))
    words = [word for word in words if word not in stopwords]
    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    def freq_in_docs(word):
        freq = 0
        for doc in documents:
            if any(word == w for w in documents[doc]):
                freq += 1
        return freq

    words = set(word for doc in documents for word in documents[doc])
    IDFs = dict().fromkeys(words)
    N = len(documents)
    for word in words:
        IDFs[word] = log(N / freq_in_docs(word))
    return IDFs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    filesnames = list(files.keys())
    TFIDF_score = dict().fromkeys(filesnames, 0.0)
    for word in query:
        for filename in filesnames:
            TFIDF_score[filename] += files[filename].count(word) * idfs[word]
    filesnames.sort(key = lambda filename : TFIDF_score[filename], reverse=True)
    return filesnames[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # `mwm` = matching word measure
    scores = []
    for sentence in sentences:
        freq = 0
        mwm = 0.0
        for word in query:
            if word in sentences[sentence]:
                freq += 1
                mwm += idfs[word]
        scores.append((sentence, mwm, freq / len(sentences[sentence])))
    scores.sort(key = lambda score : (score[1], score[2]), reverse=True)
    rank = [score[0] for score in scores]
    return rank[:n]

if __name__ == "__main__":
    main()