import os
import sys
import pandas as pd
import numpy as np
from src.GoldPricePrediction.logger import logging
from src.GoldPricePrediction.exception import customexception

class DataTransformation:            
    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            target_column_name = 'GLD'
            drop_columns = [target_column_name,'Date']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[np.array(input_feature_train_df), np.array(target_feature_train_df)]
            test_arr = np.c_[np.array(input_feature_test_df), np.array(target_feature_test_df)]
            
            logging.info("Data Preprocessing Process is Completed")
            
            return (train_arr,test_arr)
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys) 