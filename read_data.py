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
        np.random.seed(123)
        self.file_index=np.random.choice(len(self.all_file_list), self.file_num).tolist()
        self.file_list=self.all_file_list[self.file_index].tolist()
        
        self.data=[]
    
    def load_data(self):
        for file_name in self.file_list:
            file_path=os.path.join(self.dir_path, file_name)
            json_file = gzip.open(file_path, 'rb')
            counter=0
            while counter<self.file_data_num:
                data_json = json_file.readline()
                data_dict = json.loads(data_json)
                self.data.append(data_dict)
                counter+=1
