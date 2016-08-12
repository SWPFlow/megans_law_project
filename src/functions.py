from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, precision_score, accuracy_score, precision_recall_curve, auc
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from classes import tag_counter, ListSplitter, RaceDummies, CrimeAndSentence, FeatureEngineer, ColumnFilter, PickEstimator
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

def oversample(X, y):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    if y.sum() * 1.0 / y.size > .5:
        minority = 0
        sample_add = y.sum() * 2 - y.size
    else:
        minority = 1
        sample_add = y.size - y.sum() * 2

    X_overs = X[y==1].sample(sample_add, replace=True)
    y_overs = y[y==1].sample(sample_add, replace=True)
    X = pd.concat([X, X_overs])
    y = pd.concat([y, y_overs])

    return X, y

def oversample_train_test(df, y):
    '''
    INPUT: X = features, DataFrame
           y = target variable binary, imbalanced classes
    OUPUT: X_train_over = training set features oversampled to have balanced target classes
           X_test = test set features
           y_train_over = training set target variable oversampled to have balanced classes
           y_test = test set target variable

    Splits data into training and test sets and then oversamples the minority
    class of the training set.
    '''
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
    X_train_over, y_train_over = oversample(X_train, y_train)
    return X_train_over, X_test, y_train_over, y_test

def plot_pr_curve(probas, y_test, label):
    '''
    INPUT: probas = predicted probabilities of the positive class
           y_test = actual values of target variable
           label = label for the matplotlib graph
    OUPUT: None

    Plots the precision recall curve of your model.
    '''
    precision, recall, thresholds = precision_recall_curve(y_test, probas)
    plt.plot(recall, precision, label=label)

def pr_auc(estimator, X, y):
    '''
    INPUT: estimator = sklearn classification model
           X = features
           y = target variable binary
    OUPUT: prauc = Area under the precision recall curve (float [0, 1])

    Returns the area under the precision recall curve for your model.
    '''
    probas = estimator.predict_proba(X.copy())[:,1]
    precision, recall, thresholds = precision_recall_curve(y, probas)
    prauc = auc(recall, precision)
    return prauc

def get_train_test_sets(p, df, y):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
    X_train_over, y_train_over = oversample(X_train, y_train)

    data_settings = {'prej': True,
                     'noprej': False,
                     'imba': (X_train, y_train),
                     'bal': (X_train_over, y_train_over)}
    training_sets = {}
    test_sets = {}
    for prejudice_setting in ['prej', 'noprej']:
        p.set_params(columns__prejudice=data_settings[prejudice_setting])
        test_sets[prejudice_setting] = (p.fit_transform(X_test.copy(), y_test), y_test)
        for balance_setting in ['bal']: #removed 'imba' from list
            features, target = data_settings[balance_setting]
            training_sets[prejudice_setting] = (p.fit_transform(features.copy(), target), target)

    return training_sets, test_sets


def big_grid_search(training_sets, test_sets):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    classifier_names = ['rfc', 'ada', 'gbc', 'log']
    classifiers = {'rfc': RandomForestClassifier,
                   'ada': AdaBoostClassifier,
                   'gbc': GradientBoostingClassifier,
                   'log': LogisticRegression}
    param_grids = {'rfc': {'n_estimators': [100, 250, 500],
                           'max_features': ['auto', 5, 10]},
                   'ada': {'n_estimators': [100, 250, 500],
                           'learning_rate': [1, 0.5, 0.1]},
                   'gbc': {'n_estimators': [100, 250, 500],
                           'learning_rate': [1, 0.1, 0.01]},
                   'log': {'penalty': ['l1', 'l2'],
                           'C': [1, 0.1, 0.01]}}

    best_score = {'prej': 0, 'noprej': 0}
    best_estimator = {}
    all_ests = {}
    for classifier in classifier_names:
        print '\n', classifiers[classifier].__name__
        for key, val in training_sets.iteritems():
            print '\nTraining set: ', key
            gs = GridSearchCV(classifiers[classifier](), param_grids[classifier], verbose=2, scoring=pr_auc)
            gs.fit(*val)
            print "\nCV PR_AUC: ", gs.best_score_
            all_ests[(key, classifier)] = gs.best_estimator_
            preds = gs.best_estimator_.predict(test_sets[key][0])
            y_true = test_sets[key][1]
            print 'Recall: ', recall_score(y_true, preds)
            print 'Precision: ', precision_score(y_true, preds)
            print 'Accuracy: ', accuracy_score(y_true, preds)
            if classifier != 'log':
                feat_imp = zip(val[0].columns, gs.best_estimator_.feature_importances_)
                print 'Feature Importances:\n', sorted(feat_imp, key=lambda x: x[1], reverse=True)[:5]

            if gs.best_score_ > best_score[key]:
                print "We've got a winner!"
                best_estimator[key] = gs.best_estimator_
                best_score[key] = gs.best_score_


    return best_score, best_estimator, all_ests



def more_precise_grid(df, y):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    X_train, X_test, y_train, y_test = oversample_train_test(df, y)

    prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('est', PickEstimator())
    ])

    no_prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter(prejudice=False)),
        ('est', PickEstimator())
    ])

    for esto in [LogisticRegression(),
                 AdaBoostClassifier(n_estimators=500),
                 GradientBoostingClassifier(n_estimators=500),
                 RandomForestClassifier(n_estimators=100)]:
        prej_model.set_params(est__estimator=esto)
        no_prej_model.set_params(est__estimator=esto)
        print esto.__class__.__name__

        print '\nPrejudice:\n'
        # prej_grid = {'gbc__learning_rate': [1.25, 1, 0.75],
        #              'gbc__n_estimators': [750]}
        #
        # gs_prej = GridSearchCV(prej_model, prej_grid, scoring=my_fbeta, verbose=1)
        # gs_prej.fit(X_train.copy(), y_train)
        # print '\nCV FBeta: ', gs_prej.best_score_
        # print gs_prej.best_params_
        prej_model.fit(X_train.copy(), y_train)
        prej_preds = prej_model.predict(X_test.copy())
        prej_scores = prej_model.predict_proba(X_test.copy())[:,1]
        print 'Proba Range: {}-{}'.format(prej_scores.min(), prej_scores.max())
        print 'Recall: ', recall_score(y_test, prej_preds)
        print 'Precision: ', precision_score(y_test, prej_preds)
        print 'Accuracy: ', accuracy_score(y_test, prej_preds)
        print 'PRAUC: ', pr_auc(prej_model, X_test, y_test)
        plot_pr_curve(prej_scores, y_test, 'Prej, {}'.format(esto.__class__.__name__))

        print '\nNo Prejudice:\n'

        no_prej_grid = {'gbc__learning_rate': [0.07, 0.05, 0.04],
                        'gbc__n_estimators': [75, 80, 85]}

        # gs_no_prej = GridSearchCV(no_prej_model, no_prej_grid, scoring=my_fbeta, verbose=1)
        # gs_no_prej.fit(X_train.copy(), y_train)
        # print '\nCV FBeta: ', gs_no_prej.best_score_
        # print gs_no_prej.best_params_
        no_prej_model.fit(X_train.copy(), y_train)
        no_prej_preds = no_prej_model.predict(X_test.copy())
        no_prej_scores = no_prej_model.predict_proba(X_test.copy())[:,1]
        print 'Proba Range: {}-{}'.format(no_prej_scores.min(), no_prej_scores.max())
        print 'Recall: ', recall_score(y_test, no_prej_preds)
        print 'Precision: ', precision_score(y_test, no_prej_preds)
        print 'Accuracy: ', accuracy_score(y_test, no_prej_preds)
        print 'PRAUC: ', pr_auc(no_prej_model, X_test, y_test)
        plot_pr_curve(no_prej_scores, y_test, 'No Prej, {}'.format(esto.__class__.__name__))

    plt.legend(loc='best')
    plt.show()

def partial_dependence(df, y):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    X_train, X_test, y_train, y_test = oversample_train_test(df, y)
    # X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)

    feature_engineering = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter(prejudice=False))
    ])

    X = feature_engineering.fit_transform(X_train.copy(), y_train)
    X_test = feature_engineering.fit_transform(X_test.copy(), y_test)

    gbc = GradientBoostingClassifier(n_estimators=850, learning_rate=.75)
    gbc.fit(X.copy(), y_train)
    most_imp = np.argsort(gbc.feature_importances_)[-6:]

    names = list(X_test.columns)
    feats = list(most_imp)
    fig, axs = plot_partial_dependence(gbc, X_test, feats, feature_names=names,
                                       n_jobs=3, grid_resolution=50)

def pickle_no_prej(df, y):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    X_train, X_test, y_train, y_test = oversample_train_test(df, y)

    no_prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('gbc', GradientBoostingClassifier())
    ])

    param_grid = {'gbc__n_estimators': [775, 800, 850],
                  'gbc__learning_rate': [1, .75, .5]}
    gs = GridSearchCV(no_prej_model, param_grid, scoring=pr_auc, verbose=3)
    gs.fit(X_train.copy(), y_train)
    print gs.best_estimator_
    print gs.best_score_
    return gs

def pickle_prej(df, y):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)

    prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('ada', AdaBoostClassifier())
    ])

    param_grid = {'ada__n_estimators': [400, 450, 500],
                  'ada__learning_rate': [.75, 0.5, .25]}
    gs = GridSearchCV(prej_model, param_grid, scoring=pr_auc, verbose=3)
    gs.fit(X_train.copy(), y_train)
    print gs.best_estimator_
    print gs.best_score_
    with open('prej_ada.pkl', 'w') as f:
        pickle.dump(gs.best_estimator_, f)

def pickle_gbc_prej(df, y):
    '''
    INPUT: X = features
           y = target variable binary, imbalanced classes
    OUPUT: X = features oversampled to have balanced target classes
           y = target variable oversample to have balanced classes

    Discovers the minority class and then oversamples until eah class makes up
    50% of your data.
    '''
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)

    prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('gbc', AdaBoostClassifier())
    ])

    param_grid = {'gbc__n_estimators': [425, 450, 475],
                  'gbc__learning_rate': [.75, 0.5, .25]}
    gs = GridSearchCV(prej_model, param_grid, scoring=pr_auc, verbose=3)
    gs.fit(X_train.copy(), y_train)
    print gs.best_estimator_
    print gs.best_score_
    return gs
    # with open('prej_gbc.pkl', 'w') as f:
    #     pickle.dump(gs.best_estimator_, f)
