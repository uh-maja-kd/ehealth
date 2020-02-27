def _get_spans(self, sentence: str):
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


def _get_spacy_vector(self, word: str, lang: str = "spanish"):
    pass
