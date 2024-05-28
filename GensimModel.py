import gensim.downloader
from tqdm import tqdm

class GensimModel:

    def __init__(self):
        self.model_name_list=[
            'glove-twitter-25',
            # 'glove-twitter-100',
            # 'glove-wiki-gigaword-50',
            # 'glove-wiki-gigaword-100',
            # 'word2vec-google-news-300',
        ]
        self.low_model_list=[
            'glove-twitter-25',
            'glove-wiki-gigaword-50',
        ]
        self.high_model_list=[
            'word2vec-google-news-300',
        ]
        self.model_dict={
            model_name:gensim.downloader.load(model_name)
            for model_name in self.model_name_list
        }
        self.simi_words_dict={
            model_name:{}
            for model_name in self.model_name_list
        }

    def find_simi_words(self, target_word, simi_num=10):
        found_words=[]
        words_scores=[]
        for model_name in self.model_name_list:
            try:
                if target_word in self.simi_words_dict[model_name]:
                    similar_words=self.simi_words_dict[model_name][target_word]
                else:
                    similar_words = self.model_dict[model_name].most_similar(target_word)
                    self.simi_words_dict[model_name][target_word]=similar_words
                for (word, score) in similar_words:
                    if word not in found_words:
                        found_words.append(word)
                        words_scores.append((word, score))
                    if len(found_words)>simi_num:
                        break
            except:
                # print(model_name+' do not find')
                continue
        if len(found_words)>simi_num:
            found_words=found_words[0:simi_num]
        # if len(finded_words)==0:
        # for model_name in self.high_model_list:
        #     try:
        #         similar_words = self.model_dict[model_name].most_similar(target_word)
        #         for (word, _) in similar_words:
        #             if word not in finded_words:
        #                 finded_words.append(word)
        #     except:
        #         # print(model_name+' do not find')
        #         continue
        return found_words

if __name__=='__main__':
    words=['like', 'like', 'like', 'like', 'is', 'pre', 'a', 'b', 'z', 'pro', 'iPhone', 'apple', 'apples', 'upscale']
    gensim_model=GensimModel()

    for word in tqdm(words):
        similar_words = gensim_model.find_simi_words(word, simi_num=20)
        print(word, similar_words)