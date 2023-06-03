import pandas as pd
from sklearn import metrics
import joblib
import os
import argparse
import config
import model_dispatcher
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def read_data_with_folds(data_path, fold):
    df = pd.read_csv(data_path)

    df_train = df[df.kfold != fold].reset_index(drop=True)  # training data is where kfold is not equal to provided fold
    df_test = df[df.kfold == fold].reset_index(drop=True)  # test data is where kfold is equal to provided fold
    
    X_train = df_train.drop("sentiments", axis=1).values
    y_train = df_train["sentiments"].values

    X_test = df_test.drop("sentiments", axis=1).values
    y_test = df_test["sentiments"].values
    return X_train, y_train, X_test, y_test

def read_data_without_folds(train_data_path, test_data_path):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    X_train = df_train.drop("sentiments", axis=1).values
    y_train = df_train["sentiments"].values

    X_test = df_test.drop("sentiments", axis=1).values
    y_test = df_test["sentiments"].values
    return X_train, y_train, X_test, y_test

def calculate_metrics(y_test, preds):
    metrics_dict = {}
    metrics_dict['accuracy'] = metrics.accuracy_score(y_test, preds)
    metrics_dict['precision'] = metrics.precision_score(y_test, preds, average='weighted')
    metrics_dict['recall'] = metrics.recall_score(y_test, preds, average='weighted')
    metrics_dict['f1_score'] = metrics.f1_score(y_test, preds, average='weighted')
    metrics_dict['specificity'] = np.mean(calculate_specificity(y_test, preds))  # Calculate average specificity across all classes

    return metrics_dict


def calculate_specificity(y_test, preds):
    cm = metrics.confusion_matrix(y_test, preds)
    num_classes = cm.shape[0]
    specificities = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificities.append(specificity)
    return specificities


def run(fold, model, cross_validation):
    print(f"fold={fold}, model={model}, cross_validation={cross_validation}")  # Debug print

    # read the corresponding data based on the target class and cross-validation technique
    if cross_validation == "True":
        X_train, y_train, X_test, y_test = read_data_with_folds(config.TRAINING_SETNT_FOLDS, fold)
        print("Sentiment prediction with cross-validation has been implemented")
    else:
        X_train, y_train, X_test, y_test = read_data_without_folds(config.TRAINING_PROCESSED, config.TEST_PROCESSED)
        print("Sentiment prediction without cross-validation has been implemented")
        


    # train the model
    clf = model_dispatcher.models[model]
    clf.fit(X_train, y_train)
    
    # test the model accuracy
    preds = clf.predict(X_test)

    # Creating  a confusion matrix,which compares the y_test and the model predictions: preds
    cm = confusion_matrix(y_test, preds)
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm,
                    index = ['Negative','Neutral','Positive'], 
                    columns = ['Negative','Neutral','Positive'])
    
    #Plotting the confusion matrix
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    
    if cross_validation == "True":
        name = f"../outputs/cm_with_cross_with_{model}_fold_{fold}.png"
    else: 
        name = f"../outputs/cm_without_cross_with_{model}.jpg"
    plt.savefig(name, dpi=300, bbox_inches='tight') 
    plt.show()

    evaluation_metrics = calculate_metrics(y_test, preds)
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")
    
    output_directory = "../models"
    os.makedirs(output_directory, exist_ok=True)

    # save the trained model
    model_filename = f"{model}_{fold}_CV_{cross_validation}.bin"
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, model_filename))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",
                        type=int)
    parser.add_argument("--model",
                        type=str)
    parser.add_argument("--cross_validation", 
                        type=str, 
                        choices=["True", "False"],
                        default="True")
    

    args = parser.parse_args()

    run(fold=args.fold,
        model=args.model,
        cross_validation=args.cross_validation
        )