import dataloader
import itertools
import math
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
class NaiveBayes:
    
    def __init__(self, path, to_search):
        self.loader = dataloader.DataLoader(path, to_search)
        self.loader.generateClassSummaries()
        
    def calculate_probability(self, x, mean, stdev):
        exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
        
        
    # Calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(self, corr_map):
    	total_rows = len(self.loader.data.index)
    	probabilities = dict()
    	for class_value, class_summaries in self.loader.summaries.items():
            probabilities[class_value] = math.log(self.loader.class_count[class_value]/float(total_rows))
            for feature in self.loader.imp_features:
                    row = int(feature[0] / self.loader.summaries[class_value][0].shape[0])
                    col = feature[0] % self.loader.summaries[class_value][1].shape[0]

                    mean = class_summaries[0][row][col]
                    stdev = class_summaries[1][row][col]

                    probability = self.calculate_probability((1 - feature[1]) * (1 - feature[2]) * corr_map[row][col], mean, stdev) + .0000000000000000000001
                    probabilities[class_value] += math.log(probability)

    	return probabilities
    
all_regions = ["False", "True"]
to_search = ["False", "True"]
nb = NaiveBayes("/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Naive_Bayes/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)", to_search)
accurate = 0
total = 0


f = open("results4.txt", "a+")
for groupings in range(2, 7):
    for pair in list(itertools.combinations(all_regions, groupings)):
        to_search = []

        for i in range(0, groupings):
            to_search.append(pair[i])
        
        nb = NaiveBayes("Data\\allBrainRegionsraw", to_search)
        accurate = 0
        total = 0
        print('\n -------------------------')
        print('\n' + str(to_search))
        for index, row in nb.loader.test.iterrows():
            if row["Class"] in to_search:
                probabilities = nb.calculate_class_probabilities(row["Data"])
                prediction = max(probabilities, key=probabilities.get)            
                if  prediction == row["Class"]:                
                    accurate += 1            
                total += 1
        
        print('\n' + "Accurate: " + str(accurate))
        print('\n' + "Total: " + str(total))
        print('\n' + "Accuracy: {accuracy:.3f}".format(accuracy = (accurate / total)))