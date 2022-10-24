from genericpath import exists
from re import sub
import nltk
from nltk import word_tokenize as tokenized
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""
NONTERMINALS = """
S -> NP VP | VP NP | S Conj S
NP -> N | Adj | Det N | Det Adj | NP Adv
NP -> NP Conj NP | Adj NP | Det Adj NP | PP
PP -> PP NP | NP PP | P NP
VP -> V | V NP | V PP 
VP -> V Adv PP | Adv VP | VP Adv 
"""
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    words = [word for word in tokenized(sentence) 
        if any(char.isalpha() for char in word)]
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    def NPsubtrees_of(tree):
        subtrees = list(tree.subtrees(lambda st : st.label() == 'NP'))
        return subtrees if subtrees[0] != tree else None 
        
    def np_chunks_from(NPtree):
        '''
        return tree if there's no NP subtrees from it 
        else return all NP subtrees from the NP subtrees of the main tree. 
        '''
        NPsubtrees = NPsubtrees_of(NPtree)
        if NPsubtrees is None:
            return NPtree
        return [np_chunks_from(subtree) for subtree in NPsubtrees]

    subtrees = np_chunks_from(tree)
    return subtrees

if __name__ == "__main__":
    main()


'''
DEBUGING
"Armchair on the sat Holmes."
We arrived the day before Thursday.
His Thursday chuckled in a paint
S -> NP VP
S -> 'we' VP (NP-->N)
S -> 'we' 'arrived' NP (VP-->V NP)
S -> 'we' 'arrived' 'the' NP (NP-->Det NP)
S -> 'we' 'arrived' 'the' 'day' PP (NP-->PP-->NP PP)
S -> 'we' 'arrived' 'the' 'day' 'before' NP (PP-->PP NP)
S -> 'we' 'arrived' 'the' 'day' 'before' 'Thursday' (NP-->N)
I had a little moist red paint in the palm of my hand.
S -> NP VP (initially)
S -> 'I' VP (NP -> N)
S -> 'I had' NP (VP -> V NP)
S -> 'I had a little' NP (NP -> Det Adj NP)
S -> 'I had a little moist' NP (NP -> Adj NP)
S -> 'I had a little moist red' NP (NP -> Adj NP)
S -> 'I had a little moist red paint' PP (NP -> NP PP)

'''