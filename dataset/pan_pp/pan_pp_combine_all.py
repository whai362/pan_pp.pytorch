import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import math
import string
import scipy.io as scio
import mmcv
from .coco_text import COCO_Text

EPS = 1e-6
synth_root_dir = './data/SynthText/'
synth_train_data_dir = synth_root_dir
synth_train_gt_path = synth_root_dir + 'gt.mat'

ic17_root_dir = './data/ICDAR2017MLT/'
ic17_train_data_dir = ic17_root_dir + 'ch8_training_images/'
ic17_train_gt_dir = ic17_root_dir + 'ch8_training_localization_transcription_gt_v2/'

ct_root_dir = './data/COCO-Text/'
ct_train_data_dir = ct_root_dir + 'train2014/'
ct_train_gt_path = ct_root_dir + 'COCO_Text.json'

ic15_root_dir = './data/ICDAR2015/Challenge4/'
ic15_train_data_dir = ic15_root_dir + 'ch4_training_images/'
ic15_train_gt_dir = ic15_root_dir + 'ch4_training_localization_transcription_gt/'

tt_root_dir = './data/total_text/'
tt_train_data_dir = tt_root_dir + 'Images/Train/'
tt_train_gt_dir = tt_root_dir + 'Groundtruth/Polygon/Train/'

def get_img(img_path, read_type='cv2'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def check(s):
    for c in s:
        if c in list(string.printable[:-6]):
            continue
        return False
    return True


def get_ann_synth(img, gts, texts, index):
    bboxes = np.array(gts[index])
    bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
    bboxes = bboxes.transpose(2, 1, 0)
    bboxes = np.reshape(bboxes, (bboxes.shape[0], -1)) / ([img.shape[1], img.shape[0]] * 4)

    words = []
    for text in texts[index]:
        text = text.replace('\n', ' ').replace('\r', ' ')
        words.extend([w for w in text.split(' ') if len(w) > 0])

    return bboxes, words


def get_ann_ic17(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word = gt[9].replace('\r', '').replace('\n', '')

        if len(word) == 0 or word[0] == '#':
            words.append('###')
        elif not check(word):
            words.append('???')
        else:
            words.append(word)

        bbox = [int(gt[i]) for i in range(8)]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words


def get_ann_ct(img, anns):
    h, w = img.shape[0:2]
    bboxes = []
    words = []
    for ann in anns:
        bbox = ann['polygon']
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * (len(bbox) // 2))
        bboxes.append(bbox)

        if 'utf8_string' not in ann:
            words.append('###')
        else:
            word = ann['utf8_string']
            if not check(word):
                words.append('???')
            else:
                words.append(word)

    return np.array(bboxes), words


def get_ann_ic15(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word = gt[8].replace('\r', '').replace('\n', '')
        if word[0] == '#':
            words.append('###')
        else:
            words.append(word)

        bbox = [int(gt[i]) for i in range(8)]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words


def get_ann_tt(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    words = []

    data = scio.loadmat(gt_path)
    data_polygt = data['polygt']
    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])

        point_num = len(X[0])
        word = lines[4]
        if len(word) == 0:
            word = '???'
        else:
            word = word[0]
            # word = word[0].encode("utf-8")

        if word == '#':
            word = '###'

        words.append(word)

        arr = np.concatenate([X, Y]).T
        bbox = []
        for i in range(point_num):
            bbox.append(arr[i][0])
            bbox.append(arr[i][1])
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * point_num)
        bboxes.append(bbox)

    return bboxes, words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, min_size, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)
    # print (h_scale, w_scale, h_scale / w_scale)

    img = scale_aligned(img, h_scale, w_scale)
    return img


def random_crop_padding(imgs, target_size):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char


class PAN_PP_CombineAll(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 with_rec=False,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.for_rec = with_rec
        self.read_type = read_type

        self.img_paths = {}
        self.gts = {}
        self.texts = {}

        self.img_num = 0
        # synth
        data = scio.loadmat(synth_train_gt_path)
        self.img_paths['synth'] = data['imnames'][0]
        self.gts['synth'] = data['wordBB'][0]
        self.texts['synth'] = data['txt'][0]
        self.img_num += len(self.img_paths['synth'])

        # ic17
        self.img_paths['ic17'] = []
        self.gts['ic17'] = []
        img_names = [img_name for img_name in mmcv.utils.scandir(ic17_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(ic17_train_data_dir, '.png')])
        for idx, img_name in enumerate(img_names):
            img_path = ic17_train_data_dir + img_name
            self.img_paths['ic17'].append(img_path)

            gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
            gt_path = ic17_train_gt_dir + gt_name
            self.gts['ic17'].append(gt_path)
        self.img_num += len(self.img_paths['ic17'])

        # coco_text
        self.ct = COCO_Text(ct_train_gt_path)
        self.img_paths['ct'] = self.ct.getImgIds(imgIds=self.ct.train, catIds=[('legibility', 'legible')])
        self.img_num += len(self.img_paths['ct'])

        # ic15
        self.img_paths['ic15'] = []
        self.gts['ic15'] = []
        img_names = [img_name for img_name in mmcv.utils.scandir(ic15_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(ic15_train_data_dir, '.png')])
        for idx, img_name in enumerate(img_names):
            img_path = ic15_train_data_dir + img_name
            self.img_paths['ic15'].append(img_path)

            gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
            gt_path = ic15_train_gt_dir + gt_name
            self.gts['ic15'].append(gt_path)
        self.img_num += len(self.img_paths['ic15'])

        # tt
        self.img_paths['tt'] = []
        self.gts['tt'] = []
        img_names = [img_name for img_name in mmcv.utils.scandir(tt_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(tt_train_data_dir, '.png')])

        for idx, img_name in enumerate(img_names):
            img_path = tt_train_data_dir + img_name
            self.img_paths['tt'].append(img_path)

            gt_name = 'poly_gt_' + img_name.split('.')[0] + '.mat'
            gt_path = tt_train_gt_dir + gt_name
            self.gts['tt'].append(gt_path)
        self.img_num += len(self.img_paths['tt'])

        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 200
        self.max_word_len = 32
        print('reading type: %s.' % self.read_type)

    def __len__(self):
        return self.img_num

    def load_synth_single(self, index):
        img_path = synth_train_data_dir + self.img_paths['synth'][index][0]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_synth(img, self.gts['synth'], self.texts['synth'], index)
        return img, bboxes, words

    def load_ic17_single(self, index):
        img_path = self.img_paths['ic17'][index]
        gt_path = self.gts['ic17'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic17(img, gt_path)
        return img, bboxes, words

    def load_ct_single(self, index):
        img_meta = self.ct.loadImgs(self.img_paths['ct'][index])[0]
        img_path = ct_train_data_dir + img_meta['file_name']
        img = get_img(img_path, self.read_type)

        annIds = self.ct.getAnnIds(imgIds=img_meta['id'])
        anns = self.ct.loadAnns(annIds)
        bboxes, words = get_ann_ct(img, anns)

        return img, bboxes, words

    def load_ic15_single(self, index):
        img_path = self.img_paths['ic15'][index]
        gt_path = self.gts['ic15'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words

    def load_tt_single(self, index):
        img_path = self.img_paths['tt'][index]
        gt_path = self.gts['tt'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_tt(img, gt_path)
        return img, bboxes, words

    def __getitem__(self, index):
        choice = random.random()
        if choice < 1.0 / 5.0:
            index = random.randint(0, len(self.img_paths['synth']) - 1)
            img, bboxes, words = self.load_synth_single(index)
        elif choice < 2.0 / 5.0:
            index = random.randint(0, len(self.img_paths['ic17']) - 1)
            img, bboxes, words = self.load_ic17_single(index)
        elif choice < 3.0 / 5.0:
            index = random.randint(0, len(self.img_paths['ct']) - 1)
            img, bboxes, words = self.load_ct_single(index)
        elif choice < 4.0 / 5.0:
            index = random.randint(0, len(self.img_paths['ic15']) - 1)
            img, bboxes, words = self.load_ic15_single(index)
        else:
            index = random.randint(0, len(self.img_paths['tt']) - 1)
            img, bboxes, words = self.load_tt_single(index)


        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        gt_words = np.full((self.max_word_num + 1, self.max_word_len), self.char2id['PAD'], dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1, ), dtype=np.int32)
        for i, word in enumerate(words):
            if word == '###':
                continue
            if word == '???':
                continue
            word = word.lower()
            gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        if self.is_transform:
            img = random_scale(img, self.img_size[0], self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            if type(bboxes) == list:
                for i in range(len(bboxes)):
                    bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                           (bboxes[i].shape[0] // 2, 2)).astype('int32')
            else:
                bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * (bboxes.shape[1] // 2)),
                                    (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            if not self.for_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop, word_mask)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )
        if self.for_rec:
            data.update(dict(
                gt_words=gt_words,
                word_masks=word_mask
            ))

        return data

if __name__=='__main__':
    data_loader = PANPP_CombineAll(
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_scale=0.5,
        read_type='pil',
        with_rec=True
    )
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    for data in train_loader:
        print('-' * 20)
        for k, v in data.items():
            print(f'k: {k}, v.shape: {v.shape}')

