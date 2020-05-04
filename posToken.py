import re


class PosToken:
    """
    Class Token to format and extract the features of a toke
    :param token string
    """
    def __init__(self, token):
        self._token = token

    def _search_pattern(self, pattern):
        """
        search pattern among the token
        :param pattern: string
        :return: 1 if true else 0
        """
        if re.search(pattern, self._token):
            return 1.
        else:
            return 0.

    def _check_ends_with(self, parts):
        pattern = r'\S+' + parts + '$'
        return self._search_pattern(pattern)

    # feature idea inspired from
    # https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31
    def get_features(self):
        features = list()
        features.append(1.)

        # If token starts with capital letter
        features.append(self._search_pattern('[A-Z]+[a-z]+$'))

        # if token ends with `ing`
        features.append(self._check_ends_with('ing'))

        # if token ends with `ed`
        features.append(self._check_ends_with('ed'))

        # if token ends with `s`
        features.append(self._check_ends_with('s'))

        # if token ends with `es`
        features.append(self._check_ends_with('es'))

        # if token ends with `ly`
        features.append(self._check_ends_with('ly'))

        # if token ends with `ous`
        features.append(self._check_ends_with('ous'))

        # if token is a number
        features.append(self._search_pattern('^[0-9]+'))

        return features
