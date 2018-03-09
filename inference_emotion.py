import time
from os import listdir
from os.path import join, exists, isdir, split
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import config as cfg
from lib.au_engine import AuEngine


class InferenceEngine(object):
    def __init__(self, model_path, img_path, gt_facs_label=None, gt_emotion_label_path=None):
        if not exists(model_path):
            raise ValueError ('model {} not exist'.format(model_path))
        if not exists(img_path):
            raise ValueError ('image path {} not exist'.format(img_path))

        if isdir(img_path):
            flist = [join(img_path, f) for f in listdir(img_path)]
        else:
            flist = [img_path]
        self.images = flist

        self.cfg = cfg

        self.net = torch.load(model_path)
        self.net.eval()
        self.net.cuda()

        self.test_transforms = transforms.Compose([
            transforms.Scale(size=(self.cfg.crop_height, self.cfg.crop_width)),
            transforms.ToTensor(),
        ])

        self.au_engine = AuEngine()

        #self.gt_facs = {}
        #if gt_facs_label is not None:
        #    assert exists(gt_facs_label), 'gt facs label {} does not exist'.format(gt_facs_label)
        #    with open(gt_facs_label, 'r') as gt_facs_infor:
        #        for line in gt_facs_infor.readlines():
        #            line = line.strip()


        self.gt_emotion = {}
        if gt_emotion_label_path is not None:
            assert exists(gt_emotion_label_path), 'gt emotion label {} does not exist'.format(gt_emotion_label_path)
            with open(gt_emotion_label_path, 'r') as gt_emotion_label:
                for line in gt_emotion_label.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    data = line.split()
                    img_name = data[0]
                    emotion_code = data[1]
                    self.gt_emotion[img_name] = int(emotion_code)

    def __len__(self):
        return len(self.img_list)

    @property
    def img_list(self):
        return self.images

    def inference_facs(self):
        facs_result = {}
        for img_path in self.images:
            img = Image.open(img_path)
            img_tensor = self.test_transforms(img)
            img_tensor = img_tensor.resize_(1, 3, self.cfg.crop_height, self.cfg.crop_width)
            img_tensor = img_tensor.cuda()

            pred = self.net(Variable(img_tensor))
            result = self.net.pred_to_labels(pred.data, thresh=cfg.thresh)

            #print(img_path, result[0])
            facs_result[img_path] = result[0].cpu().numpy()[0]
        return facs_result

    def predict_to_facs(self, pred):
        facs_code = []
        for i, indicator in enumerate(pred):
            if indicator > 0:
                facs_code.append(self.cfg.class_facs_converter[i])
        return facs_code

    def inference_emotion(self):
        facs_result = self.inference_facs()
        for img_name in facs_result:
            pred = facs_result[img_name]
            curr_facs = self.predict_to_facs(pred)
            emotion = self.au_engine.inference_emotion(curr_facs)
            print(img_name,curr_facs, emotion )
            _, just_name = split(img_name)
            if just_name in self.gt_emotion:
                emotion_code = self.gt_emotion[just_name]
                print(self.cfg.emotion_list[emotion_code])


if __name__ == '__main__':
    root_dir = '/media/lhj/新加卷/Data/Emotion_ck_pain/Vion_CK+'
    image_path = 'test_face_images'
    gt_emotion_file = 'emotion_info.txt'
    model_path = './models/AU_Detection_DRML_iter_50_18_03_08.pkl'
    gt_emotion_label = ''
    test_dir = join(root_dir, image_path)
    test_img = join(root_dir, image_path, 'S075_008_00000012.png')
    gt_emotion = join(root_dir, gt_emotion_file)
    img_init_eng = InferenceEngine(model_path=model_path, img_path=test_dir,
                                   gt_emotion_label_path=gt_emotion)
    #result = img_init_eng.inference_facs()
    #print(result)
    #for key in result:
    #    facs_code = img_init_eng.predict_to_facs(result[key])
    emotion_result = img_init_eng.inference_emotion()
    print(emotion_result)






















