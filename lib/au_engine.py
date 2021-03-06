class AuEngine(object):
    def __init__(self):
        self.emotion_list = {0:'neutral', 1:'Angry', 2:'Contempt', 3:'Disgust', 4:'Fear', 5:'Happy',
                             6:'Sadness', 7:'Surprise' }
        self.facs_codes = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29,
                           31, 34, 38, 39, 43]

        # based on paper Extended Cohn-Kanade (CK+) Paper and wikipedia
        # https://en.wikipedia.org/wiki/Facial_Action_Coding_System

        angry_FACS = [[23, 24], [4, 5, 7, 23]]
        comtempt_FACS = [[14], [12, 14]]
        disgust_FACS = [[9], [10], [9, 15, 16]]
        fear_FACS = [[1, 2, 4] ]#, [1, 2, 5]]
        happy_FACS = [[12], [6, 12]]
        sadness_FACS = [[1, 4, 15], [1, 4, 11], [11], [6, 15]]
        surprise_FACS = [[1, 2], [5], [1, 2, 5, 26]]
        self.all_FACS = [angry_FACS, comtempt_FACS, disgust_FACS, fear_FACS, happy_FACS, sadness_FACS, surprise_FACS]



    def inference(self, query_code):
        possible_emotion = {}
        for emotion_idx, emotion in enumerate(self.all_FACS):
            for facs_group in emotion:
                hit_count = 0
                for facs in facs_group:
                    if facs in query_code:
                        hit_count += 1
                    else:
                        break
                if len(facs_group) == hit_count : #and emotion_idx not in possible_emotion:
                    possible_emotion[emotion_idx+1] = hit_count  # add one bias because 0 is for neutral, the default
        emotion, hit_count = 0, -1
        for k in possible_emotion:
            if possible_emotion[k] > hit_count:
                emotion = k
                hit_count = possible_emotion[k]

        return [emotion ]

    def inference_emotion(self, query_code):
        emotion_codes = self.inference(query_code)
        emotion_names = []
        for emotion_code in emotion_codes:
            assert (emotion_code in self.emotion_list), 'emotion code {} not in emotion list'.format(emotion_code)
            emotion_names.append(self.emotion_list[emotion_code])

        return emotion_names





