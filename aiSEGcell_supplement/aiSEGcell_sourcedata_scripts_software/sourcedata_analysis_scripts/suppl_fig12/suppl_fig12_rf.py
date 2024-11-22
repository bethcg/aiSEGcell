from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import os
import sklearn
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn import tree
from typing import Any, Dict, List, Tuple
import random
import pickle

def args_parse():
    '''
        Catches user input.

        Parameters
        ----------
        
        -

        Return
        ------

        Returns a namespace from `argparse.parse_args()`.
    '''
    desc = "Program to extract feature importance with random forest."

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--df_file',
                        type = str,
                        default = './classification_df_new.csv',
                        help = 'Path to file with feature dataframe.')
    
    parser.add_argument('--out_path',
                        type = str,
                        default = '../output/suppl_fig12/',
                        help = 'Path to directory to save all outputs to.')
    
    return parser.parse_args()

args = args_parse()
out_path = args.out_path
os.makedirs(out_path, exist_ok=True)

features = ['cell_identifier',
'Nucleus.Gradient.Mag.Skewness', 
'focus',
'cell_mask_cut_offness', 
'cell_mask_Nucleus.Gradient.Mag.Std',
'cell_mask_Shape.Extent', 
'cell_mask_Shape.Solidity',
'cell_mask_Nucleus.Intensity.Skewness', 
'cell_mask_Shape.Eccentricity',
'cell_mask_Size.Area',
'distance_closest_nuc', 
'cut_offness', 
'cell_mask_Nucleus.Gradient.Mag.Skewness',
'cell_mask_Nucleus.Intensity.Std',
'Nucleus.Gradient.Mag.Std',
'Shape.Eccentricity',
'is_mutlinucleated',
'Shape.Extent',
'Shape.Solidity',
'Nucleus.Intensity.Std',
'Nucleus.Intensity.Skewness',
'Size.Area',
'IOU', 
'label', 
'movie_int'
]

def CV_for_rf_param(df_x_train: pd.DataFrame, df_y_train: pd.DataFrame) -> RandomForestClassifier:
    """
    Get random forest estimator from cross validation
    
    Parameter
    ---------
    
    per_movie_feature_df: pd.DataFrame
        dataframe with features for one movie
    cnt_data: int
        number of good/bad examples to be extracted    

    Return
    ------
    
    Returns best RandomForestClassifier after cross validation
    """
    # a random classifier
    clf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0, bootstrap=False)
    
    # set up grid search for number of trees and depth
    param_grid = {
                 'n_estimators': [20, 50, 100, 200, 300, 400, 500, 600],
                 'max_depth': [5, 7, 9, 11, 13]
                }
    
    # get best parameters from grid search cross validation
    grid_clf = GridSearchCV(clf, param_grid, cv=4)
    grid_clf.fit(df_x_train, df_y_train)
    
    # get results from gridsearch
    params = pd.DataFrame (grid_clf.cv_results_['params'])
    mean_test_score = pd.DataFrame (grid_clf.cv_results_['mean_test_score'], columns = [str(len(df_x_train)) + '_mean_test_score'])
    std_test_score = pd.DataFrame (grid_clf.cv_results_['std_test_score'], columns = [str(len(df_x_train)) + '_std_test_score'])
    rank_test_score = pd.DataFrame (grid_clf.cv_results_['rank_test_score'], columns = [str(len(df_x_train)) + '_rank_test_score'])
    overview_HP = pd.concat([params, mean_test_score, std_test_score, rank_test_score], axis=1)
    overview_HP = overview_HP.sort_values(by=str(len(df_x_train)) + '_rank_test_score')
    
    # print choices for parameters
    print('best score: ', grid_clf.best_score_)
    print(grid_clf.best_params_)
    
    return grid_clf.best_estimator_, overview_HP, grid_clf.best_score_

def reduce_datatsize(df_class:pd.DataFrame, data_size:int) -> pd.DataFrame:
    """
    Reduce data size by taking most extreme examples
    
    Parameter
    ---------
    
    df_class: pd.DataFrame
        dataframe with features to be reduced
    data_szie: int
        dividable by 10
        number of data points after reduction 

    Return
    ------
    
    Returns dataframe with most extreme examples per movie
    """
    good_bad_per_movie = int(data_size/10)
    
    # reduce size of dataset (take most extreme examples)
    df_class_reduced = pd.DataFrame()
    
    # loop through movies
    for i in range(1,6):
        good_movie = df_class.loc[(df_class['movie_int'] == i) & (df_class['label'] == 1)]
        good_movie = good_movie.sort_values(by='IOU')
        bad_movie = df_class.loc[(df_class['movie_int'] == i) & (df_class['label'] == 0)]
        bad_movie = bad_movie.sort_values(by='IOU')
        reduced_good_set = good_movie.iloc[len(good_movie)-good_bad_per_movie:,:]
        reduced_bad_set =bad_movie.iloc[:good_bad_per_movie,:]
        df_class_reduced = pd.concat([df_class_reduced, reduced_good_set, reduced_bad_set])
        
    return df_class_reduced

def prepare_training_test_data(df_class: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Split data in train and test data
    
    Parameter
    ---------
    
    df_class: pd.DataFrame
        dataframe with features for whole dataset

    Return
    ------
    
    Returns list with x_train, y_train, x_test, y_test
    """
    global out_path

    # replace NaN
    df_class = df_class.fillna(99999999)
    bad_data = df_class.loc[df_class['label'] == 0].copy()
    good_data = df_class.loc[df_class['label'] == 1].copy()
    
    # shuffle good and bad data
    bad_data = shuffle(bad_data, random_state=0)
    good_data = shuffle(good_data, random_state=0)
    
    # get training and test data
    test_size = 100
    train_good = good_data.iloc[:-test_size,:]
    train_bad = bad_data.iloc[:-test_size,:]
    test_good = good_data.iloc[-test_size:,:]
    test_bad = bad_data.iloc[-test_size:,:]
    
    # make sure good and bad training size is the same
    diff_training_size = len(train_good)-len(train_bad)
    
    # set random seed
    np.random.seed(123)
    smpl_delete = np.random.choice(np.arange(0, len(train_good)), size = diff_training_size, replace = False)
    train_good = train_good.drop(train_good.index[smpl_delete])
    
    # take 100/100 as test set 
    df_x_train_full = pd.concat([train_bad, train_good])
    df_x_test_full = pd.concat([test_bad, test_good])

    # save train and test set
    df_x_train_full.to_csv(out_path + 'classification_df_' + str(len(df_class)) + '_train' + '.csv')
    df_x_test_full.to_csv(out_path + 'classification_df_' + str(len(df_class)) + '_test' +'.csv')
    
    # exlude filename and label columns
    df_x_train = df_x_train_full.drop(['label','cell_identifier'],axis=1)
    df_x_test = df_x_test_full.drop(['label', 'cell_identifier'],axis=1)
    print('num features:', len(df_x_train.columns))
    
    # print training and test sizes
    print('good train set:', len(train_good))
    print('bad train set:', len(train_bad))
    print('good test set:', len(test_good))
    print('bad test set:', len(test_bad))
    
    # get labels
    df_y_train = df_x_train_full['label']
    df_y_test = df_x_test_full['label']
    
    # return df_x_train, df_y_train, df_x_test, df_y_test
    return df_x_train, df_y_train, df_x_test, df_y_test

def get_best_hyperparameter(feature_df: pd.DataFrame):
    global out_path
    
    dataset_size = [250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]
    current_best_datasetsize = 0
    current_best_score = 0
    current_best_clf = 0
    df_HP = pd.DataFrame()

    for num_data in dataset_size:
        
        df_class = feature_df.copy()
        
        # reduce size of dataset (take most extreme examples)
        df_class = reduce_datatsize(df_class, num_data)
        df_class = df_class.drop(['movie_int', 'IOU'], axis=1)
        
        # split into training and test data
        df_x_train, df_y_train, df_x_test, df_y_test = prepare_training_test_data(df_class)
        
        # do cv and get classifier, hyperparameter df, and best score
        clf, HP_df, score = CV_for_rf_param(df_x_train, df_y_train)
        
        # add new value to hyperparam df 
        df_HP = pd.concat([df_HP, HP_df], axis = 1)
        
        if score > current_best_score:
            current_best_datasetsize = num_data
            current_best_score = score
            current_best_clf = clf

    print('best CV score between all options:', current_best_score)
    return df_HP, current_best_datasetsize, current_best_clf

def get_testscore(feature_df: pd.DataFrame, best_datasize: int, best_clf: RandomForestClassifier=0) -> float:
    global out_path
    
    df_class = feature_df.copy()
    
    # reduce size of dataset (take most extreme examples)
    df_class = reduce_datatsize(df_class, best_datasize)
    df_class = df_class.drop(['movie_int', 'IOU'], axis=1)
    
    # split into training and test data
    df_x_train, df_y_train, df_x_test, df_y_test = prepare_training_test_data(df_class)
    
    # do cv and get classifier, hyperparameter df, and best score
    if best_clf==0:
        clf, HP_df, score = CV_for_rf_param(df_x_train, df_y_train)
        clf.fit(df_x_train, df_y_train) 
        test_score  = clf.score(df_x_test, df_y_test)
        predictions = clf.predict(df_x_test)
    else:
        clf=best_clf
        test_score  = clf.score(df_x_test, df_y_test)   
        predictions = clf.predict(df_x_test)
    
    # save test data
    test_df = pd.DataFrame({"gt_label": df_y_test, "predicted_label": predictions})    
    test_df.to_csv(out_path + 'test_predictions.csv', index=False)

    # get feature importances 
    feature_importances = clf.feature_importances_
    df_importance = pd.DataFrame()
    df_importance.insert(loc=0, column='features', value=df_x_train.columns, allow_duplicates=True)
    df_importance.insert(loc=1, column='importance', value=feature_importances, allow_duplicates=True)
    df_importance = df_importance.sort_values(by= 'importance')
    df_importance.to_csv(out_path + 'feature_importance_numfeatures_'+ str(len(df_x_train.columns)) + '.csv', index=False)
    
    return clf, test_score

def main():
    global out_path
    args = args_parse()
    feature_df = args.df_file
    
    df_class = pd.read_csv(feature_df) 
    df_class = df_class.loc[:,features].copy()
    
    df_HP, best_datasize, best_clf = get_best_hyperparameter(df_class)
    df_HP.to_csv(out_path + 'HP_overview_df.csv')
    
    # save model
    best_model_dict = {'datasize': best_datasize,
                       'model': best_clf
                       }
    with open(out_path + 'best_rf_model.pickle', 'wb') as f:
        pickle.dump(best_model_dict, f)
    
    # load model    
    # with open('./best_rf_model.pickle', 'rb') as f:
    #     model_dict = pickle.load(f)   
    # best_datasize = model_dict['datasize']
    # best_clf = model_dict['model']
    feature_importances = pd.DataFrame({'features': features[1:22], 'importance': best_clf.feature_importances_})
    feature_importances.sort_values(by='importance', inplace=True, ascending=True)
    feature_importances.to_csv('./feature_importance_numfeatures_21.csv', index=False)

    clf, accurcay = get_testscore(df_class, best_datasize, best_clf)
    print('test set accuracy: ', accurcay)
        

if __name__ == '__main__':
    main()    
