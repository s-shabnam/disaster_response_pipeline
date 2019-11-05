import sys
import pandas as pd
import sqlalchemy as db

def load_data(messages_filepath, categories_filepath):
    """
        Load messages and related categories from input paths arguments

         :param messages_filepath: The path of message file
         :param categories_filepath: The path of category file
         :type messages_filepath: string
         :type categories_filepath: string
         :return: The result of the merge of message and related categories
         :rtype: DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on = 'id')
    
    return df


def clean_data(df):
    """
        Cleans and drops unused columns and duplicated lines of the dataframe passed in argument.

         :param df: The dataframe to be cleaned
         :type df: DataFrame
         :return: The result of the cleaning processing
         :rtype: DataFrame
    """
    
    categories = df.categories.str.split(pat = ';', expand = True)
    row = categories.iloc[0]
    category_colnames = list(row.str[0:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories',axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.replace(2,1,inplace =True)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """
        Saves the dataframe df passed as argument in the given directory
         :param df: The dataframe to be saved
         :param database_filename: The path of the repository for db saving
         :type df: DataFrame
         :type database_filename: Path string
         :return: None
    """
    engine = db.create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False)  


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
              'to as the third argument. \n\nExample: '\
              'python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()