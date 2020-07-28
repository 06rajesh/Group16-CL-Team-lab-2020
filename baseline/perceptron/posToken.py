
class PosToken:
    """
    Class Token to format and extract the features of a toke
    """
    def __init__(self,):
        pass

    @staticmethod
    def format_term(term):
        term = term.lower()
        term = term.strip()
        return  term

    @staticmethod
    def get_features(sentence_terms, index):
        """ Compute some very basic word features.
            :param sentence_terms: [w1, w2, ...]
            :type sentence_terms: list
            :param index: the index of the word
            :type index: int
            :return: dict containing features
            :rtype: dict
        """
        original = sentence_terms[index]
        term = PosToken.format_term(original)
        return {
            'nb_terms': len(sentence_terms),
            'term': term,
            'is_first': index == 0,
            'is_last': index == len(sentence_terms) - 1,
            'is_capitalized': original[0].upper() == original[0],
            'is_all_caps': original.upper() == original,
            'is_all_lower': original.lower() == original,
            'prefix-1': term[0],
            'prefix-2': term[:2],
            'prefix-3': term[:3],
            'suffix-1': term[-1],
            'suffix-2': term[-2:],
            'suffix-3': term[-3:],
            'prev_word': '' if index == 0 else PosToken.format_term(sentence_terms[index - 1]),
            'next_word': '' if index == len(sentence_terms) - 1 else PosToken.format_term(sentence_terms[index + 1])
        }
