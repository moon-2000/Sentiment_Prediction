import numpy as np
import pandas as pd
from sklearn import model_selection
import config



def create_folds():
    data = pd.read_csv(config.TRAINING_PROCESSED)  # reading the training data
    
    data["kfold"] = -1  # create a new column called kfold and fill it with -1
    data = data.sample(frac=1).reset_index(drop=True)  # randomize the rows of the data

    num_bins = int(np.floor(1 + np.log2(len(data))))  # calculate the number of bins by Sturge's rule with the floor of the value

    
    print("Unique sentiments:", data["sentiments"].unique()) 
    data["sentiments"] = data["sentiments"].round().astype(int)  # convert sentiments to integers
    # Bin targets
    data.loc[:, "bins"] = pd.cut(data["sentiments"], bins=num_bins, labels=False)


    # Initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # Fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # Drop the bins column if it exists
    if "bins" in data.columns:
        data = data.drop("bins", axis=1)
    
    
    # Save the new csv with kfold column
    data.to_csv("../input/train_folds_sentiment.csv", index=False)
    print("Folds for sentiment are created successfully")
    

if __name__ == "__main__":
    df = create_folds()   # to create folds for sentiment classification 
