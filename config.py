import os

cuda_num = 0

height = 180
width = 180
crop_height = 170
crop_width = 170

lr = 0.0001
lr_decay_every_epoch = 100
lr_decay_rate = 0.9

epoch = 50
train_batch_size = 64
test_batch_size = 64
thresh = 0.8
test_every_epoch = 1

class_facs_converter = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 23, 24, 26]
class_number = len(class_facs_converter)

emotion_list = {0:'neutral', 1:'Angry', 2:'Contempt', 3:'Disgust', 4:'Fear', 5:'Happy',
                6:'Sadness', 7:'Surprise' }

data_root = '/media/lhj/新加卷/Data/Emotion_ck_pain/Redist_Vion_CK+'
train_info = os.path.join(data_root, 'train_info_aug.txt')
test_info = os.path.join(data_root, 'test_info_aug.txt')

image_dir = os.path.join(data_root, 'face_images_aug/')

