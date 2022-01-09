import sys
import nltk
from sqlalchemy import create_engine
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Reads data from SQLite DB and returns X, labels and category names
    :param database_filepath: file path for input SQLite DB
    :return X: set of features 
    :return y: set with labels
    :return category_names: Category names for labels
    """
    database_name = 'sqlite:///' + database_filepath
    engine = create_engine(database_name)
    df = pd.read_sql_table('project2db', database_name)
    X = df['message']
    Y = df.drop(['message', 'id', 'genre', 'original'], axis=1)
    category_names = Y.columns
    #print("categories: ", list(category_names))
    return X, Y.values, category_names


def tokenize(text):
    """
    Tokenizes the text messages
    :param text: text message
    :return clean_tokens: list of clean tokens extracted from text input
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
    Builds the model by creating the pipeline and preparing gridsearch to fine tune parameters
    :return model: returns fine tuned model
    """
    # Creates pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Parameters for Hypertuning
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 150],
        'clf__estimator__min_samples_split': [2, 4],
      
        } 
    
    #parameters = {
    #   #'tfidf__use_idf': (True),
    #    'clf__estimator__n_estimators': [50],
    #    'clf__estimator__min_samples_split': [2, 4],
    #    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates the model performance on the Test subset
    :param model: Model to evaluate
    :param X_test: samples / features in Test subset
    :param Y_test: Labels for Test subset
    :param category_names: Category names for labels
    :return : No output
    """
    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        #y_testclean = y_test[i]
        print(f'Result for {column}:')
        print(classification_report(y_test[:, i], y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Exports model to a pickle file
    :param model: Model to export
    :param model_filepath: Destination file path for model export into pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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