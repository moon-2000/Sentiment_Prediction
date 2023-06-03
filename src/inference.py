





import numpy as np

# Confusion matrix
confusion_matrix = np.array([[210, 87, 22],
                            [20, 1100, 120],
                            [3, 110, 1800]])

# Calculate class accuracies
class_accuracies = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)

# Calculate class weights
num_samples = np.sum(confusion_matrix, axis=1)
class_weights = num_samples / (len(class_accuracies) * num_samples.sum())

# Calculate weighted average accuracy (CACR)
cacr = np.sum(class_accuracies * class_weights)

print("CACR:", cacr)

# CACR_dict = {"without_lin_C01": 0.2611933081906972, 
#              "without_lin_C1" : 0.2808721317718747,
#              "without_lin_C10" :0.28485025851136475, 
#              "without_lin_C100":0.2863332391447678,
# 
#               "without_poly_C01": 0.1963059665409535,
#               "without_poly_C1": 0.2757964812173086, 
#               "without_poly_C10": 0.2817310485042115,
#               "without_poly_C100": 0.2534057349718967,
# 
#               "without_sig_C01": 0.25976149259472975,
#               "without_sig_C1":  0.2644438056912906, 
#               "without_sig_C10": 0.24574488429910119,
#               "without_sig_C100": 0.2441101334090264
#   
#               "without_rbf_C10": 0.29857910906298,
#               "without_rbf_C1000": 0.29857910906298}