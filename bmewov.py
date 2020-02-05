import re
from functools import reduce
import operator

def add(n, t):
    return [n+i for i in t]

class BMEWOV:
    tags = ["B", "M", "E", "W", "O", "V"]

    @staticmethod
    def _discontinuos_match_regex1(match):
        entities_spans = []

        start, end = match.span()
        string = match.string[start:end]

        reg_v = r"V+"
        reg_me = r"M*E"

        match_v = re.search(reg_v, string)

        for match_me in re.finditer(reg_me, string):
            entities_spans.append(add(start, list(range(*match_v.span())) + list(range(*match_me.span()))))

        return entities_spans

    @staticmethod
    def _discontinuos_match_regex2(match):
        entities_spans = []

        start, end = match.span()
        string = match.string[start:end]

        reg_b = r"B"
        reg_v = r"V+"

        match_v = re.search(reg_v, string)

        for match_b in re.finditer(reg_b, string):
            entities_spans.append(add(start, list(range(*match_b.span())) + list(range(*match_v.span()))))

        return entities_spans

    @staticmethod
    def _discontinuos_entities(sequence: list):
        sequence_str = "".join(sequence)

        cont_matches = []
        entities_spans = []

        reg1 = r"V+(M*EO*)+M*E"
        reg2 = r"(BO)+BV+"

        for match in re.finditer(reg1, sequence_str):
            entities_spans.extend(BMEWOV._discontinuos_match_regex1(match))
            for i in range(*match.span()):
                sequence[i] = "O"

        for match in re.finditer(reg2, sequence_str):
            entities_spans.extend(BMEWOV._discontinuos_match_regex2(match))
            for i in range(*match.span()):
                sequence[i] = "O"

        return entities_spans

    @staticmethod
    def _continuos_entities(sequence: list):
        return []

    @staticmethod
    def decode(sequence: list):
        return BMEWOV._discontinuos_entities(sequence) + BMEWOV._continuos_entities(sequence)

    @staticmethod
    def encode(sentence_spans: list, entities_spans: list):
        ret_tags = []
        for span in token_spans:
            tag = None
            ent_with_span = [entity for entity in entities_spans if span in entity]
            if len(ent_with_span) == 0:
                tag = "O"
            elif len(ent_with_span)>1:
                tag = "V"
            else:
                entity = ent_with_span[0]
                if len(entity) == 0:
                    tag = "W"
                elif span == entity[0]:
                    tag = "B"
                elif span == entity[-1]:
                    tag = "E"
                else:
                    tag = "M"
            ret_tags.append(tag)
        return ret_tags


print(BMEWOV.decode(["B","O","B","V","V","O","O","V", "V", "M", "M", "E", "E", "O"]))