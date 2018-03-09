import unittest
import os

import config as cfg
from inference_emotion import InferenceEngine

class TestInferenceEmotion(unittest.TestCase):
    def setUp(self):
        #self.Engine = InferenceEngine()
        self.root_dir = '/media/lhj/新加卷/Data/Emotion_ck_pain/Vion_CK+'
        self.image_path = 'face_images'
        self.model_path = '../models/AU_Detection_DRML_iter_50_18_03_08.pkl'
        self.test_dir = os.path.join(self.root_dir, self.image_path)
        self.test_img = os.path.join(self.root_dir, self.image_path, 'S075_008_00000012.png')

    def test_generate_right_path_list(self):
        img_init_eng = InferenceEngine(model_path=self.model_path, img_path=self.test_img)
        self.assertIsInstance(img_init_eng.img_list, list)
        self.assertEqual(len(img_init_eng), 1)
        dir_init_eng = InferenceEngine(model_path=self.model_path, img_path=self.test_dir)
        self.assertIsInstance(dir_init_eng.img_list, list)
        self.assertEqual(len(dir_init_eng), 588)

    def test_engine_raise_value_error_when_img_path_wrong(self):
        self.assertRaises(ValueError, InferenceEngine, self.model_path, '1' )

    def test_engine_raise_value_error_when_model_path_wrong(self):
        self.assertRaises(ValueError, InferenceEngine, '1', self.test_dir)
        self.assertRaises(ValueError, InferenceEngine, '1', self.test_img)
        #self.assertRaises(ValueError, InferenceEngine, self.model_path, self.test_img)

    def test_generate_facs_code_correctly(self):
        img_init_eng = InferenceEngine(model_path=self.model_path, img_path=self.test_img)
        result = img_init_eng.inference_facs()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        #self.assertEqual(len(result[0]), cfg.class_number)

        img_init_eng = InferenceEngine(model_path=self.model_path, img_path=self.test_dir)
        #result = img_init_eng.inference_facs()
        #self.assertIsInstance(result, dict)
        #self.assertIsInstance(result[0], dict)
        #self.assertEqual(len(result), 588)
        #self.assertEqual(len(result[0]), cfg.class_number)

    def test_generate_facs_code_from_code_correctly(self):
        pred = [1, 1, 0, 0, 0, 1 ,1, 1, 0, 1, 0,1,1,0,1,0]
        img_init_eng = InferenceEngine(model_path=self.model_path, img_path=self.test_img)
        self.assertEqual(img_init_eng.predict_to_facs(pred), [1,2,7, 9,10,12,15,16,24])


    def test_generate_emotion_correctly(self):
        pass

