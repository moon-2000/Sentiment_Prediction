# Review Classification: SVM vs. Random Forests   

### Project Overview   
    The model predicts the review_sentiment (positive, neutral, negative).     

    For text vectoization, TF-IDF vectorizer is used for Sentiments vectorization.   

    The training data will be used in two different versions (with and without cross validation due to the imbalance in target distribution).        

    The classifier, Support Vector Machine (SVM) is used in many versions, each with different parameters.   


### Project Structure   
    input/: This folder consists of all the input files and data.  

    src/: All python scripts associated with the project here.  

    models/: This folder keeps all the trained models.  

    notebooks/: All jupyter notebooks (i.e. any *.ipynb file).

    outputs/: All images for data visualisation are stored here.  


### Files to run in order    
    1- explore_data.py  : to have an over view of data distributions   

    2- clean_split.py   : to clean the data and handle missing values before splitting data into train.csv and test.csv    

    3- preprocess.py    : to handle categorical variables, preprocess and vectorize the text data   

    4- create_folds.py  : to take the preprocess training data, and create folds for applying the cross-validation technique   

    5- train.py         : to train the desired model on the target (sentiment)      

    6- inference.py     : to use the trained model on new data   

to run train.py  file       
    python3 train.py --fold 0 --model model_name --cross_validation       

    --cross_validation "True"  to run the version with cross validation technique        

    --cross_validation "False"  to run the version without cross validation technique    


### Resources      
Dataset:   https://www.kaggle.com/datasets/danielihenacho/amazon-reviews-dataset 