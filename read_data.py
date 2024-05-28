import gzip
import json
import os
import numpy as np

class c4:

    def __init__(self, dir_path, file_num=50, file_data_num=10):
        self.dir_path=dir_path
        self.file_num=file_num
        self.file_data_num=file_data_num
        self.all_file_list=np.array(os.listdir(self.dir_path))
        # np.random.randint(0, len(self.file_list), self.file_num)
        np.random.seed(123)
        self.file_index=np.random.choice(len(self.all_file_list), self.file_num).tolist()
        self.file_list=self.all_file_list[self.file_index].tolist()
        
        self.data=[]
    
    def load_data(self):
        for file_name in self.file_list:
            file_path=os.path.join(self.dir_path, file_name)
            json_file = gzip.open(file_path, 'rb')
            # json_list = json_file.readlines()
            # data_index = np.random.choice(len(json_list), self.file_data_num).tolist()
            counter=0
            while counter<self.file_data_num:
                data_json = json_file.readline()
                data_dict = json.loads(data_json)
                self.data.append(data_dict)
                counter+=1

# def parse(path):
#     g = gzip.open(path, 'rb')
#     for l in g:
#         yield json.loads(l)

# def get_data(dir_path, file_name):
#     file_path=os.path.join(dir_path, file_name)
#     for d in parse(file_path):
#         print(d)

if __name__=='__main__':
    dir_path='/home/plll/dataset/c4/realnewslike'
    # file_name='c4-train.00000-of-00512.json.gz'
    # get_data(dir_path, file_name)
    c4_dataset=c4(dir_path=dir_path)
    c4_dataset.load_data()
    print()