import spacy

nlp = spacy.load("es_core_news_md")


def get_spans(sentence: str):
    spans, begun, start = [], False, None
    punct = '.,;:()-""'
    for i, c in enumerate(sentence):
        if not begun and c not in " " + punct:
            begun = True
            start = i
        if begun and c in " " + punct:
            begun = False
            spans.append((start, i))
            if c in punct:
                spans.append((i, i))
    return spans


def get_spacy_vector(word: str):
    return nlp.vocab.get_vector(word)
