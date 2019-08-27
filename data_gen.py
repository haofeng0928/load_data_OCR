import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from bucketdata import BucketData
import traceback


class DataGen(object):

    GO = 1
    EOS = 2

    def __init__(self,
                 data_root,
                 annotation_fn,
                 lexicon_file,
                 mean,
                 channel,
                 evaluate,
                 valid_target_len=float('inf'),

                 img_width_range=(12, 320),
                 word_len=30):

        self.min_list = [0.0, 0, 0]
        self.max_list = [0.0, 0, 0]
        img_height = 32
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(self.data_root, annotation_fn)
        
        self.mean = mean
        self.channel = channel
        assert len(self.mean) == self.channel

        if not os.path.exists(lexicon_file):
            lexicon_file = os.path.join(self.data_root, lexicon_file)

        if evaluate:
            '[(16,32),(27,32),(35,32),(64,32),(80,32)]'
            # self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)),
            #                      (int(math.floor(108 / 4)), int(word_len + 2)),
            #                      (int(math.floor(140 / 4)), int(word_len + 2)),
            #                      (int(math.floor(256 / 4)), int(word_len + 2)),
            #                      (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
            self.bucket_specs = [(int(math.ceil(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            '[(16,11),(27,17),(35,19),(64,22),(80,32)]'
            # self.bucket_specs = [(int(64 / 4), 9 + 2),
            #                      (int(108 / 4), 15 + 2),
            #                      (int(140 / 4), 17 + 2),
            #                      (int(256 / 4), 20 + 2),
            #                      (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]  # BTBT TODO 调试时仅用一个bucket
            self.bucket_specs = [(int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

        self.bucket_min_width, self.bucket_max_width = img_width_range  # (12,320)
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.lexicon_dic = {}
        # lexicon_file.encode('utf-8')
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                self.lexicon_dic[line] = i+3  # ?*? 给GO,EOS等特殊字符腾出id的位置，0表示啥都没

        self.bucket_data = {i: BucketData() for i in range(self.bucket_max_width + 1)}  # 321

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r', encoding='utf-8') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len

        with open(self.annotation_path, 'r', encoding='utf-8') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            ture_imgs = 0
            false_imgs = 0
            for l in lines:
                s = l.strip('\n').split('\t')
                if len(s) == 1:
                    s = l.strip('\n').split(' ')
                if len(s) != 2:
                    print(l.strip('\n'))
                    continue
                
                img_path, lex = s

                try:
                    img_bw, word = self.read_data(img_path, lex)
                    if img_bw is None or word is None:
                        continue

                    width = img_bw.shape[-1]

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root, img_path))
                    if bs >= batch_size:
                        ture_imgs += bs
                        # print('batch length = ', b_idx)
                        b = self.bucket_data[b_idx].flush_out(self.bucket_specs,
                                                              valid_target_length=valid_target_len,
                                                              go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d'%width
                    else:
                        false_imgs += bs

                except Exception as e:
                    print('exception!!!')
                    msg = traceback.format_exc()
                    print(msg)
                    pass
            print('images of ture / false / total = ', ture_imgs, false_imgs, len(lines))
        self.clear()

    def read_data(self, img_path, lex):
        if len(lex) == 0 or len(lex) >= self.bucket_specs[-1][1]:
            return None, None
        
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            # print('before: w, h = ', w, h)
            if w < 10 and h < 10:
                return None, None
            aspect_ratio = w / h
            # if aspect_ratio < 5:
            #     self.min_list[0] = aspect_ratio
            #     self.min_list[1] = w
            #     self.min_list[2] = h
            # if aspect_ratio > 5:
            #     self.max_list[0] = aspect_ratio
            #     self.max_list[1] = w
            #     self.max_list[2] = h

            if aspect_ratio < float(self.bucket_min_width) / self.image_height:  # img_width_range[0]
                img = img.resize(
                    (self.bucket_min_width, self.image_height),Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:    # img_width_range[1]
                img = img.resize(
                    (self.bucket_max_width, self.image_height),Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            # w, h = img.size
            # print('after: w, h = ', w, h)

            if self.channel==1:
                img_bw = img.convert('L')
                img_bw = np.asarray(img_bw, dtype=np.float)
                img_bw = img_bw[np.newaxis, :]    # =>(1,h,w)
                img_bw[0] = (img_bw[0]-self.mean[0])/255.0
            elif self.channel==3:
                img_bw = np.asarray(img,dtype=np.float)
                img_bw = img_bw[:,:,::-1]  # RGB => BGR
                img_bw = img_bw.transpose(2,0,1) # => (3,h,w)
                img_bw[0] = (img_bw[0]-self.mean[0])/255.0   # B
                img_bw[1] = (img_bw[1]-self.mean[1])/255.0   # G
                img_bw[2] = (img_bw[2]-self.mean[2])/255.0   # R
            else:
                raise ValueError

        word = [self.GO]
        
        lex = lex.lower()
        for c in lex:
            if c in self.lexicon_dic.keys():
                word.append(self.lexicon_dic[c])

        word.append(self.EOS)
        
        if len(word) == 2:
            return img_bw, None

        word = np.array(word, dtype=np.int32)

        return img_bw, word

