# NOTE: CUSTOM MODIFICATION OF BUILT-IN FUNCTION
# Below is the function spacy uses to generate noun chunks; I simply added NUM to
# also be recognized as noun chunk.
# See https://github.com/explosion/spaCy/blob/master/spacy/lang/en/syntax_iterators.py
from spacy.symbols import NOUN, PROPN, PRON, NUM
from nltk import Tree as nltkTree

def noun_chunks(obj):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    labels = [
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
    ]
    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    np_label = doc.vocab.strings.add("NP")
    seen = set()
    for i, word in enumerate(obj):
        if word.pos not in (NOUN, PROPN, PRON, NUM):
            continue
        # Prevent nested chunks from being produced
        if word.i in seen:
            continue
        if word.dep in np_deps:
            if any(w.i in seen for w in word.subtree):
                continue
            seen.update(j for j in range(word.left_edge.i, word.i + 1))
            yield obj[word.left_edge.i:word.i + 1]
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                if any(w.i in seen for w in word.subtree):
                    continue
                seen.update(j for j in range(word.left_edge.i, word.i + 1))
                yield obj[word.left_edge.i:word.i + 1]
SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}

def to_nltk_tree(node):
    # reference: https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
    if node.n_lefts + node.n_rights > 0:
        return nltkTree("(%s) %s" % (node.dep_, node.orth_),
                        [to_nltk_tree(child)
                         for child in node.children])
    else:
        return node.orth_

