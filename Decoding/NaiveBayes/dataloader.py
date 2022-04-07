import numpy as np
import pandas as pd
import os, csv
import pickle

class DataLoader:

    def __init__(self, path, to_search):
        self.regions = to_search
       # print("New Dataloader")
        self.path = path
        self.data = self.convertToDataFrame(self.path)
        # fix validation split
        self.train, self.test = self.cross_validation_split(self.data, 0.5)
        self.summaries = dict()
        self.class_count = dict()
        self.imp_features = self.all_features() 
        
        #self.load_obj("features_mean") 
        
    
    def all_features(self):
        features = []
        for i in range(0, len(self.data)*len(self.data)):
            features.append((i, 0, 0))
        return features
            
    def load_obj(self, name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
        
    def cross_validation_split(self, data, train_size):
        train = data.sample(frac=train_size)
        test = data.drop(train.index).sample(frac=1.0)
        
        return train, test
        
    def convertToDataFrame(self, path):
        df = pd.DataFrame(columns=("Class", "Data"))
        for subdir, dirs, files in os.walk(path):
        
        
            for file in files:
                reader = csv.reader(open(os.path.join(subdir, file), "rt"), delimiter=",")
                x = list(reader)
                result = np.array(x).astype("float")
                
                row = pd.DataFrame({"Class" : os.path.basename(os.path.normpath(subdir)), "Data" : [result] })
            
                df = df.append(row, ignore_index=True)
        
        return df
    
    
    def summarizeData(self, label, important_features):
        mean_matrix = np.zeros([len(self.data), len(self.data)])
        std_matrix = np.zeros([len(self.data), len(self.data)])
        label_count = 0
        
        for index, row in self.data.iterrows():   
            #if self.shouldAnalyze(row["Class"], label):

            if row["Class"] == label:
                label_count += 1
                data = row["Data"]
                
                for feature in important_features:
                    row = int(feature[0] / mean_matrix.shape[0])
                    col = feature[0] % std_matrix.shape[1]
                    mean_matrix[row][col] += (1 - feature[1]) * (1 - feature[2]) * data[row][col]
                
                        
        mean_matrix = mean_matrix / label_count
        for index, row in self.data.iterrows():
             #if self.shouldAnalyze(row["Class"], label):
            if row["Class"] == label:
           
                data = row["Data"]
                for feature in important_features:
                    row = int(feature[0] / std_matrix.shape[0])
                    col = feature[0] % std_matrix.shape[1]
                    std_matrix[row][col] += (((1 - feature[1]) * (1 - feature[2]) * data[row][col]) - mean_matrix[row][col]) ** 2
        std_matrix = std_matrix / (label_count - 1)
        std_matrix = np.sqrt(std_matrix)
        
        return mean_matrix, std_matrix, label_count
    
    def generateClassSummaries(self):
        for region in self.regions:
            means, stds, class_count = self.summarizeData(region, self.imp_features)
            self.summaries[region] = [means, stds]
            self.class_count[region] = class_count
        
                    
   
loader = DataLoader("/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Naive_Bayes/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/train", ["False", "True"])
loader.generateClassSummaries()
print(loader.summaries)