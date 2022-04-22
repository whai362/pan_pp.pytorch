import os.path as osp
import editdistance
import mmcv


class Corrector:
    def __init__(self,
                 data_type,
                 len_thres,
                 score_thres,
                 unalpha_score_thres,
                 ignore_score_thres,
                 edit_dist_thres=0,
                 edit_dist_score_thres=0,
                 voc_type=None,
                 voc_path=None):

        self.len_thres = len_thres
        self.score_thres = score_thres
        self.unalpha_score_thres = unalpha_score_thres
        self.ignore_score_thres = ignore_score_thres
        self.edit_dist_thres = edit_dist_thres
        self.edit_dist_score_thres = edit_dist_score_thres
        self.voc_type = voc_type
        self.voc = self.load_voc(data_type, voc_type, voc_path)

    def process(self, img_metas, outputs):
        img_name = img_metas['img_name'][0]
        words = outputs['words']
        word_scores = outputs['word_scores']
        words = [
            self.correct(
                word, score,
                self.voc if self.voc_type != 's' \
                    else self.voc['voc_%s.txt' % img_name]
            )
            for word, score in zip(words, word_scores)
        ]
        outputs.update(dict(words=words))
        return outputs

    @staticmethod
    def _prefix_score(a, b):
        prefix_s = 0
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                prefix_s += 1.0 / (i + 1)
        return prefix_s

    def correct(self, word, score, voc=None):
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
                prefix_s = self._prefix_score(word, voc_word)
                if d < min_d:
                    matched = voc_word
                    min_d = d
                    max_prefix_s = prefix_s
                elif d == min_d and prefix_s > max_prefix_s:
                    matched = voc_word
                    max_prefix_s = prefix_s

                if min_d == 0:
                    break
            if min_d < self.edit_dist_thres or \
                    float(min_d) / len(word) < self.edit_dist_score_thres:
                return matched

            return None

        return word

    def load_voc(self, data_type, voc_type, voc_path):
        if voc_path is None:
            return None
        if 'IC15' in data_type:
            return self._load_voc_ic15(voc_type, voc_path)
        elif 'TT' in data_type:
            return self._load_voc_tt(voc_path)

    def _load_voc(self, voc_path):
        lines = mmcv.list_from_file(voc_path)
        voc = []
        for line in lines:
            line = line.encode('utf-8').decode('utf-8-sig')
            line = line.replace('\xef\xbb\xbf', '')
            line = line.replace('\r', '').replace('\n', '')
            if len(line) == 0 or line[0] == '#':
                continue
            voc.append(line.lower())

        return voc

    def _load_voc_ic15(self, voc_type, voc_path):
        if voc_type == 's' and osp.isdir(voc_path):
            voc_names = [voc_name for voc_name in
                         mmcv.utils.scandir(voc_path, '.txt')]
            voc = {}
            for voc_name in voc_names:
                voc[voc_name] = self._load_voc(osp.join(voc_path, voc_name))
        elif voc_type in ['g', 'w'] and osp.isfile(voc_path):
            voc = self._load_voc(voc_path)

        return voc

    def _load_voc_tt(self, voc_path):
        pass
