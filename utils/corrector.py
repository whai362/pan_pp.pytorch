import editdistance
import mmcv

class Corrector:
    def __init__(self,
                 data_type,
                 len_thres,
                 score_thres,
                 unalpha_score_thres,
                 ignore_score_thres,
                 editDist_thres,
                 voc_path=None):
        self.data_type = data_type

        self.len_thres = len_thres
        self.score_thres = score_thres
        self.unalpha_score_thres = unalpha_score_thres
        self.ignore_score_thres = ignore_score_thres
        self.editDist_thres = editDist_thres

        self.voc = self.load_voc(voc_path)

    def process(self, outputs):
        words = outputs['words']
        word_scores = outputs['word_scores']
        words = [self.correct(word, score, self.voc) for word, score in zip(words, word_scores)]
        outputs.update(dict(
            words=words
        ))
        return outputs

    def correct(self, word, score, voc=None):
        EPS = 1e-6
        if len(word) < self.len_thres:
            return None
        if score > self.score_thres:
            return word
        if not word.isalpha():
            if score > self.unalpha_score_thres:
                return word
            return None

        if score < self.ignore_score_thres:
            return None

        if voc is not None:
            min_d = 1e10
            matched = ''
            for voc_word in voc:
                d = editdistance.eval(word, voc_word)
                if d < min_d:
                    matched = voc_word
                    min_d = d
                if min_d == 0:
                    break
            if min_d < self.editDist_thres:
                return matched
            else:
                return None

        return word

    def load_voc(self, path):
        if path is None:
            return None
        if 'IC15' in self.data_type:
            return self._load_voc_ic15(path)
        elif 'TT' in self.data_type:
            return self._load_voc_tt(path)

    def _load_voc_ic15(self, path):
        lines = mmcv.list_from_file(path)
        voc = []
        for line in lines:
            if len(line) == 0:
                continue
            line = line.encode('utf-8').decode('utf-8-sig')
            line = line.replace('\xef\xbb\xbf', '')
            if line[0] == '#':
                continue
            voc.append(line.lower())
        return voc

    def _load_voc_tt(self, path):
        pass