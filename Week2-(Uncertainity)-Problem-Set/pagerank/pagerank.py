import os
import random
import re
import sys
from math import fabs
from copy import deepcopy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    P = dict()
    p = 0
    if len(corpus[page]) > 0:
        p = (1 - damping_factor) / len(corpus.keys())
        P = P.fromkeys(corpus.keys(), p)
        p = damping_factor / len(corpus[page])
        for next in corpus[page]:
            P[next] += p
    else:
        p = 1 / len(corpus.keys())
        P = P.fromkeys(corpus.keys(), p)
    return P



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    PR = dict().fromkeys(corpus.keys(), 0)
    next_page = random.choice(list(corpus.keys()))
    next_sample = transition_model(corpus, next_page, damping_factor)
    PR[next_page] = 1
    for _ in range(n):
        next_page = random.choices(list(next_sample.keys()), 
            weights = list(next_sample.values()))[0]
        next_sample = transition_model(corpus, next_page, damping_factor)
        PR[next_page] += 1
    for page in corpus.keys():
        PR[page] /= n
    return PR



def converged(PR, prevPR, threshold = 0.001):
    for page in PR:
        if fabs(PR[page] - prevPR[page]) >= threshold:
            return False
    return True

def pages_link_to(page, corpus):
    pages = set()
    for i in corpus:
        if page in corpus[i]:
            pages.add(i)
    return pages

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = corpus.keys()
    N = len(pages)
    C = (1 - damping_factor) / N
    NumLinks = dict().fromkeys(pages)
    pages_linking_to = dict().fromkeys(pages)
    for page in pages:
        l = len(corpus[page])
        if l == 0:
            corpus[page] = pages
            l = N
        NumLinks[page] = l

    for page in pages:
        pages_linking_to[page] = pages_link_to(page, corpus)

    prevPR = dict().fromkeys(pages, 0)
    PR = dict().fromkeys(pages, 1 / N)
    while True:
        for page in pages:
            PR[page] = C + damping_factor * sum(
                [(PR[i] / NumLinks[i]) for i in pages_linking_to[page]])
        if converged(PR, prevPR) and (fabs(sum(PR.values()) - 1.0) <= 1e-4):
            break
        prevPR = deepcopy(PR)
    return PR
    


if __name__ == "__main__":
    main()
