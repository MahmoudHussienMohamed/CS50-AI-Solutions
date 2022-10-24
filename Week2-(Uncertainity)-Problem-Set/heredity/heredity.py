import csv
import itertools
import sys
from copy import deepcopy

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def set_vals(people, people_to_set, key, value):
    for person in people_to_set:
        people[person][key] = value
def setup(ps, one_gene, two_genes, have_trait):
    copy_ps = deepcopy(ps)
    names = set(copy_ps.keys())

    set_vals(copy_ps, two_genes, 'gene', 2)
    set_vals(copy_ps, one_gene, 'gene', 1)
    set_vals(copy_ps, names.difference(one_gene.union(two_genes)), 'gene', 0)

    set_vals(copy_ps, have_trait, 'trait', True)
    set_vals(copy_ps, names.difference(have_trait), 'trait', False)
    return copy_ps

def parent_prob(parent):
    if parent['gene'] == 2:
        return 1 - PROBS['mutation']
    elif parent['gene'] == 0:
        return PROBS['mutation']
    else:
        return 0.5

def probability(all_people, people_to_calc, gene_copies):
    prob = 1.0
    for person in people_to_calc:
        father, mother = all_people[person]['father'], all_people[person]['mother']
        if father is not None:
            father = parent_prob(all_people[father])
            mother = parent_prob(all_people[mother])
            if gene_copies == 0:
                prob *= (1 - father) * (1 - mother)
            elif gene_copies == 1:
                prob *= father * (1 - mother) + (1 - father) * mother
            else:
                prob *= father *mother          
        else:
            prob *= PROBS["gene"][gene_copies]
        have_trait = all_people[person]['trait']
        prob *= PROBS["trait"][gene_copies][have_trait]
    return prob


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    ppl = setup(people, one_gene, two_genes, have_trait)
    P = probability(ppl, two_genes, 2) * probability(ppl, one_gene, 1)
    P *= probability(ppl, set(people.keys()).difference(one_gene.union(two_genes)), 0)
    return P

def increment(probs, people, key, idx, val):
    for i in people:
        probs[i][key][idx] += val

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    people =  set(probabilities.keys())
    increment(probabilities, two_genes, 'gene', 2, p)
    increment(probabilities, one_gene, 'gene', 1, p)
    increment(probabilities, people.difference(two_genes.union(one_gene)), 'gene', 0, p)
    increment(probabilities, have_trait, 'trait', True, p)
    increment(probabilities, people.difference(have_trait), 'trait', False, p)

def norm(Ps, key):
    total = 0.0
    tmp = None
    for person in Ps:
        tmp = Ps[person][key]
        total = sum(tmp.values())
        if total > 0:
            for i in tmp:
                tmp[i] /= total
def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    norm(probabilities, 'gene')
    norm(probabilities, 'trait')
    
if __name__ == "__main__":
    main()