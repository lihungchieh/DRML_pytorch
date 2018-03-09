import unittest
from lib.au_engine import AuEngine

class TestAuEngine(unittest.TestCase):
    def setUp(self):
        self.au_combinations = []
        self.auEng = AuEngine()
        self.facs_codes = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29,
                           31, 34, 38, 39, 43]
        self.max_length = 12


    def test_AuEngine_has_no_conficts_code(self):
        facs_results = []
        code_length = len(self.facs_codes)
        one_result = []
        for first_idx in range(code_length):
            result = []
            result.append(self.facs_codes[first_idx])
            for second_idx in range(first_idx + 1, code_length):
                result.append(self.facs_codes[second_idx])
                for third_idx in range(second_idx + 1, code_length):
                    result.append(self.facs_codes[third_idx])
                    for forth_idx in range(third_idx + 1, code_length):
                        result.append(self.facs_codes[forth_idx])
                        facs_results.append(result)
                        result = result[:-1]
                    result = result[:-2]
                result = result[:-3]



        print('total facs code combinations ', len(facs_results))
        for facs in facs_results:
            print(facs)
            print('fuck')
            self.assertTrue(len(self.auEng.inference(facs)) <= 1, "cannot has two emotions {} {}".format(facs, self.auEng.inference(facs)))





    def test_AuEngine_infer_method_returns_correct_result(self):
        self.assertEqual(self.auEng.inference([23, 24])[0], 1)
        self.assertEqual(self.auEng.inference([9])[0], 3)
        self.assertEqual(self.auEng.inference([10])[0], 3)
        self.assertEqual(self.auEng.inference([1,2,4])[0], 4)
        self.assertEqual(self.auEng.inference([12])[0], 5)
        self.assertEqual(self.auEng.inference([1, 4, 15])[0], 6)
        self.assertEqual(self.auEng.inference([11])[0], 6)
        self.assertEqual(self.auEng.inference([6, 15])[0], 6)
        self.assertEqual(self.auEng.inference([1, 2])[0], 7)
        self.assertEqual(self.auEng.inference([14])[0], 2)
        self.assertEqual(self.auEng.inference([13, 18])[0], 0)

    def test_AuEngine_return_none_with_duplicate_facs(self):
        self.assertEqual(self.auEng.inference([23, 24, 25])[0], 1)
        self.assertEqual(self.auEng.inference([9, 20, 10])[0], 2)
        self.assertEqual(self.auEng.inference([10, 1])[0], 2)
        self.assertEqual(self.auEng.inference([1,2,4, 5])[0], 3)
        self.assertEqual(self.auEng.inference([12, 13])[0], 4)
        self.assertEqual(self.auEng.inference([1, 4, 15, 17])[0], 5)
        self.assertEqual(self.auEng.inference([11, 13])[0], 5)
        self.assertEqual(self.auEng.inference([6, 15, 16])[0], 5)
        self.assertEqual(self.auEng.inference([1, 2, 3])[0], 6)
        self.assertEqual(self.auEng.inference([14,15])[0], 7)
        self.assertEqual(self.auEng.inference([13, 18, 26])[0], 0)

    def test_AuEngine_return_none_with_unclear_facs(self):
        self.assertEqual(self.auEng.inference([1, 3, 9])[0], 2)
        self.assertEqual(self.auEng.inference([1, 2, 4, 5, 15, 26])[0], 6)



