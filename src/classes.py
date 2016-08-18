from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

def tag_counter(lst, tag_values):
    '''
    INPUT: lst = list of crime descriptions
           tag_values = words you're looking for in the descriptions
    OUTPUT: tot = the amount of times that a word in tag_values appears
            in the list of descriptions

    Counts the number of times that words in tag_values appear in a list of
    crime descriptions.
    '''
    tot = 0
    for val in tag_values:
        for descrip in lst:
            if val in descrip:
                tot += 1
    return tot

class CustomMixin(TransformerMixin):
    '''
    Template class that allows the setting of parameters for subclasses.
    '''
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])


class ListSplitter(CustomMixin):
    '''
    Pipeline step to turn 'Description' and 'Offense Code' back into lists.
    '''
    def fit(self, X, y):
        return self

    def transform(self, X):
        X.fillna('no info', inplace=True)
        #v***vMy webscraper scraped these fields as lists, but I
        #v***vhad to separate them with ';' them when I wrote to csv
        X['Description'] = X['Description'].apply(lambda x: x.split(';'))
        X['Offense Code'] = X['Offense Code'].apply(lambda x: x.split(';'))
        #^***^
        return X


class RaceDummies(CustomMixin):
    '''
    Pipeline step that makes dummy columns for race.
    '''
    def fit(self, X, y):
        return self

    def transform(self, X):
        races = list(X['Ethnicity'].unique())
        races.remove('OTHER')
        X[races] = pd.get_dummies(X['Ethnicity'])[races]
        return X


class CrimeAndSentence(CustomMixin):
    '''
    Pipeline step that creates the Minimum and Maximum Sentence features (the
    minimum sentence for the crime they committed with the lowest minimum sentence
    and the mximum sentence for the crime they commited with the highest maximum
    sentence).

    This step also creates columns that count how many times they committed crimes
    of particular types.
    '''
    code_sentencing = {}
    with open('data/code_sentencing.txt') as f:
        for line in f:
            if 'skip' in line:
                continue
            code = line.split(':')[0]
            sentences = set(float(sentence.strip()) for sentence in line.split(':')[1].split(','))
            code_sentencing[code] = sentences

    descrip_tags = {}
    with open('data/Words_for_Dummies.txt') as f:
        for line in f:
            if 'skip' in line:
                continue
            key = line.split(':')[0]
            values = line.split(':')[1].split(',')
            values = [val.strip() for val in values]
            descrip_tags[key] = values

    def fit(self, X, y):
        return self

    def transform(self, X):
        X['Possible Sentences'] = X['Offense Code'].apply(lambda lst: set.union(set(),
                                                    *[self.code_sentencing[code]
                                                    for code in lst if code in self.code_sentencing]))
        X['Minimum Sentence'] = X['Possible Sentences'].apply(lambda x: min(x) if len(x) > 0 else 0.0)
        X['Maximum Sentence'] = X['Possible Sentences'].apply(lambda x: max(x) if len(x) > 0 else 0.0)

        for key, value in self.descrip_tags.iteritems():
            X[key] = X['Description'].apply(lambda x: tag_counter(x, value))
        return X


class FeatureEngineer(CustomMixin):
    '''
    Pipeline step that creates the rest of the features:
    'Number of Offenses'
    'Priors' number of offenses that have the word 'Prior' in their Description
    'Height in Inches'
    'BMI' body mass index
    'Female'
    'Age'
    'Years in Violation'
    'SVP' whether they are a sexually violent predator
    'Age in Question' age now if they did not violate or age at time they violated
    'Constant' for training logistic regression
    '''
    def fit(self, X, y):
        return self

    def transform(self, X):
        X['Number of Offenses'] = X['Offense Code'].apply(lambda x: len(x))
        X['Priors'] = X['Description'].apply(lambda x: sum(1 if 'PRIOR' in y else 0 for y in x))
        X['Height in Inches'] = X['Height'].apply(lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1].strip('"')))
        X['BMI'] = X['Weight'] * 0.45 / (X['Height in Inches'] * 0.025) ** 2
        X['Female'] = pd.get_dummies(X['Sex'])['FEMALE']
        X['Transient'] = X['redtext0'].str.contains('TRANSIENT') * 1
        X['Age'] = 2016 - X['Date of Birth'].apply(lambda x: int(x.split('-')[-1]))
        X['Years in Violation'] = X['redtext1'].apply(lambda x: 2016 - int(x.split('/')[-1]
                                                        .split('.')[0]) if 'VIOLATION' in x else 0)
        X['SVP'] = X['redtext1'].str.contains('VIOLENT') * 1
        X['Age in Question'] = X['Age'] - X['Years in Violation']
        X['Constant'] = 1
        return X


class ColumnFilter(CustomMixin):
    '''
    Filters out the columns that the model should not train on.

    Parameters:
                prejudice -
                            If set to true the data will include prejudicial
                            features (everything included in exclude2).
                            If set to false, the data will not include those
                            features.
    '''
    exclude1 = [u'Description', u'Offense Code', u'Score', u'Score Date',
                u'Tool Name', u'Year of Last Conviction', u'Year of Last Release',
                u'redtext0', u'redtext1', u'redtext2', u'so_id', u'Date of Birth',
                u'Ethnicity', u'Eye Color', u'Hair Color', u'Height', 'Age', 'Violation',
                u'Last Known Address', u'Sex', 'Possible Sentences', 'Years in Violation']
    exclude2 = ['Age in Question', u'BLACK', u'HISPANIC', u'FILIPINO', u'OTHER ASIAN',
                u'CHINESE', u'PACIFIC ISLANDER', u'WHITE', u'UNKNOWN', u'GUAMANIAN',
                u'KOREAN', u'VIETNAMESE', u'AMERICAN INDIAN', u'ASIAN INDIAN', u'SAMOAN',
                u'HAWAIIAN', u'CAMBODIAN', u'JAPANESE', u'LAOTIAN', u'Height in Inches',
                u'BMI', u'Female', 'Weight']

    def __init__(self, prejudice=True):
        self.prejudice = prejudice

    def get_params(self, **kwargs):
        return {'prejudice':self.prejudice}

    def fit(self, X, y):
        return self

    def transform(self, X):
        X.drop(self.exclude1, axis=1, inplace=True)
        if not self.prejudice:
            X.drop(self.exclude2, axis=1, inplace=True)
        return X


class PickEstimator(BaseEstimator):
    '''
    Estimator that needs another estimator as input, which allows you to
    gridsearch over various estimators.

    Parameters:
                estimator -
                            The estimator you wish to use. Default = AdaBoostClassifier
    '''
    def __init__(self, estimator=AdaBoostClassifier()):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X,y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
