import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from messages and categories into single dataframe
    :param messages_filepath: Messages input file
    :param categories_filepath: Categories input file
    :return df: Panda dataframe with data read from messages and categories
    """
    
    # Merge datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
        
    return df


def clean_data(df):
    """
    Cleans up data by cleaning up category names, converting values to integers and removing duplicates
    :param df: message and categories dataframe before clean up 
    :return df: dataframe after data cleaned
    """
    
    # Fix Categories Names
    categories = df['categories']
    row_names = categories.iloc[0].split(";")
    categories = df['categories'].str.split(";", expand=True)
    row_names = [category_name.split("-")[0] for category_name in row_names]
    categories.columns = row_names

    # Change categories values to binary
    for column in categories:

        categories[column] = categories[column].str.replace(column + "-","")
        categories[column] = categories[column].astype(int)  

        # Check if category value is not binary - change to binary
        categories[column] = np.where((categories[column] < 0), 0, categories[column])
        categories[column] = np.where((categories[column] > 1), 1, categories[column])

    # Replace Categories
    df = df.drop(['categories'], axis=1)
    df = df.join(categories)
    
    # Remove Duplicates
    df = df.drop_duplicates(keep='first')
    
    return df


def save_data(df, database_filename):
    """
    Saves Dataframe into Database
    :param df: Dataframe to be saved to DB
    :param database_filename: path and filename for DB
    
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('project2db', engine, index=False, if_exists='replace')
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
