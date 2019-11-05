import sys
import nltk
nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



def load_data(database_filepath):
    """
        Loads data from datebase saved in the repository assed as argument.
        :param database_filepath: The path of the db file.
        :type database_filepath; string
        :return:  X, y, category_names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name = 'InsertTableName', con = engine)
    
    X = df['message']
    y = df.drop(['message', 'original', 'genre', 'id'], axis = 1)
    category_names = np.unique(y.columns)

    return X, y, category_names


def tokenize(text):
    """
        Transforms the text to a list of tokens and cleans it.
        :param text: The text object to be tokenized.
        :type text: String
        :return: The list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds model for prediction.
    For this a pipeline is created and  parameters are optimised by usig grid search.
    :return: Builed and optimised model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 2, random_state = 1)))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__min_samples_leaf': [2, 3, 4],
    }

    #model = GridSearchCV(pipeline, param_grid=parameters)
    #model.estimator.get_params().keys()

    model = RandomizedSearchCV(pipeline, param_distributions = parameters)
    # return model
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluates the performance of the model based on the test set.
        :param model: The model used and trained to classifier the data.
        :param X_test: Data set to be classified.
        :param Y_test: The classes of the test set.
        :param category_names: The list of all classes.
    """

    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)

    for i in category_names:
        print(classification_report(Y_test[i], y_pred[i], target_names = category_names))  


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
        

def load_model(model_filepath):
   return pickle.load(open(model_filepath, 'rb'))


def main():
    if len(sys.argv) == 3:
        database_filename, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filename))
        X, y, category_names = load_data(database_filename)
        X_train, X_test, y_train, y_test = train_test_split(X, y)        
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    