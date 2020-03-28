import re
import string
from spacy.lang.en.stop_words import STOP_WORDS

class Tokenizer(object):
    def __init__(self):
        self.ignore_punc = "-_,.=;:/'()[]\"<>\\|\{\}"
        self.preserve_punc = "!"
        pass

    def _remove_number(self, text):
        return re.sub(r'\d+', '', text) 

    def _whitespace_split(self, text):
        out = text.strip().split()
        return out

    def _split_on_punc(self, text):
        text = text.strip()
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if char in string.punctuation:
                if char in self.preserve_punc:
                    output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]
    
    def _remove_stop_words(self, word_list):
        out = []
        for w in word_list:
            if w.lower() not in STOP_WORDS:
                out.append(w)

        return out

    def tokenize(self, text):
        # out = text.lower()
        out = self._remove_number(text)
        out = self._whitespace_split(out)
        new_word_list = []
        for word in out:
            if len(word) < 20:
                new_word_list.extend(self._split_on_punc(word))
        new_word_list = self._remove_stop_words(new_word_list)

        return new_word_list
