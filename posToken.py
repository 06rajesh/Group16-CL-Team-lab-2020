import re


class PosToken:
    """
    Class Token to format and extract the features of a toke
    :param token string
    """
    def __init__(self, token):
        self._token = token.strip()

    def _search_pattern(self, pattern):
        """
        search pattern among the token
        :param pattern: string
        :return: bool
        """
        if re.search(pattern, self._token):
            return True
        else:
            return False

    @staticmethod
    def _check_ends_with(term, parts):
        part_len = len(parts)
        token_part = term[-part_len:]
        if token_part == parts:
            return True
        else:
            return False

    def _get_end_features(self):
        features = list()
        lower = self._token.lower()
        last_chars = lower[-5:]

        # if token ends with `ing`: eING
        if self._check_ends_with(last_chars, 'ing') or self._check_ends_with(last_chars, 'ings'):
            features.append("eING")

        # if token ends with `ed`: eED
        if self._check_ends_with(last_chars, 'ed'):
            features.append("eED")

        # if token ends with `s`
        if self._check_ends_with(last_chars, 's'):
            features.append("eS")

        # if token ends with `es`
        if self._check_ends_with(last_chars, 'es'):
            features.append("eES")

        # if token ends with `ly`
        if self._check_ends_with(last_chars, 'ly'):
            features.append("eLY")

        # if token ends with `ous`
        if self._check_ends_with(last_chars, 'ous'):
            features.append("eOUS")

        # if token ends with `ful`
        if self._check_ends_with(last_chars, 'ful'):
            features.append("eFUL")

        # if token ends with `er`
        if self._check_ends_with(last_chars, 'er'):
            features.append("eER")

        # if token ends with `ier`
        if self._check_ends_with(last_chars, 'ier'):
            features.append("eIER")

        # if token ends with `est`
        if self._check_ends_with(last_chars, 'est'):
            features.append("eEST")

        # if token ends with `ion` or `ions`
        if self._check_ends_with(last_chars, 'ion') or self._check_ends_with(last_chars, 'ions'):
            features.append("eION")

        # if token ends with `tion` or `tions`
        if self._check_ends_with(last_chars, 'tion') or self._check_ends_with(last_chars, 'tions'):
            features.append("eTION")

        # if token ends with `ness` or `nesses`
        if self._check_ends_with(last_chars, 'ness') or self._check_ends_with(last_chars, 'nesses'):
            features.append("eNESS")

        # if token ends with `ship` or `ships`
        if self._check_ends_with(last_chars, 'ship') or self._check_ends_with(last_chars, 'ships'):
            features.append("eSHIP")

        # if token ends with `sion` or `sions`
        if self._check_ends_with(last_chars, 'sion') or self._check_ends_with(last_chars, 'sions'):
            features.append("eSION")

        # if token ends with `ment` or `ments`
        if self._check_ends_with(last_chars, 'ment') or self._check_ends_with(last_chars, 'ments'):
            features.append("eMENT")

        return features

    def _get_is_exact_feature(self):
        token = self._token.lower()

        exact_list = ['of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
                      'about', 'as', 'after', 'into', 'before', 'under', 'around',
                      'the', 'an', 'a', 'i', 'you', 'they', 'he', 'she', 'it', 'is',
                      'was', 'am', 'were', 'who', 'what', 'when', 'then', 'where', 'there',
                      'why', 'how']

        if token in exact_list:
            return 'i' + token.upper()
        elif token == '.':
            return "iPERIOD"
        elif token == ',':
            return "iCOMMA"
        elif token == ':':
            return "iCOLON"
        elif token == ';':
            return "iSEMICOLON"
        elif token == '(' or token == '{':
            return "iLRB"
        elif token == ')' or token == '}':
            return "iRRB"
        elif token == "''" or token == "``":
            return "iQUOTE"
        elif token == "&" or token == "and":
            return "iAND"
        elif token == '$':
            return "iDOLLAR"
        elif token == '-':
            return "iHYPHEN"
        elif token == "'s" or token == "'":
            return "iAPOS"
        else:
            return None

    # feature idea inspired from
    # https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31
    def get_features(self):
        features = list()

        # If token starts with capital letter: bCAP
        if self._search_pattern(r'[A-Z]+\S+$'):
            features.append("bCAP")

        # If token is all capital letter: iCAP
        if self._search_pattern('[A-Z]+$'):
            features.append("iCAP")

        # if token starts with UN: bUN
        if self._search_pattern(r'^[Uu]n\S+'):
            features.append("bUN")

        # if token starts with non: bNON
        if self._search_pattern(r'(^[Nn]on)\S+'):
            features.append("bNON")

        # if token is a number
        if self._search_pattern('^[0-9]+'):
            features.append("iNUM")

        # if token length is two
        if len(self._token) == 2:
            features.append("iL2")

        # if token length is greater than 7
        if len(self._token) > 7:
            features.append("iG7")

        features = features + self._get_end_features()
        is_exact = self._get_is_exact_feature()
        if is_exact is not None:
            features.append(is_exact)

        return features

    @staticmethod
    def all_features_list():
        features = ["bCAP", "iCAP", "bUN", "bNON", "iNUM", "iG7", "iL2", "eING", "eED", "eS", "eES", "eLY", "eOUS",
                    "eFUL","eER", "eIER", "eEST", "eION", "eTION", "eNESS", "eSHIP", "eSION", "eMENT", "iPERIOD",
                    "iCOMMA", "iCOLON", "iSEMICOLON", "iLRB", "iRRB", "iQUOTE", "iAND", "iDOLLAR", "iHYPHEN", "iAPOS"]
        exact_list = ['of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
                      'about', 'as', 'after', 'into', 'before', 'under', 'around',
                      'the', 'an', 'a', 'i', 'you', 'they', 'he', 'she', 'it', 'is',
                      'was', 'am', 'were', 'who', 'what', 'when', 'then', 'where', 'there',
                      'why', 'how']
        for term in exact_list:
            features.append("i" + term.upper())

        return features
