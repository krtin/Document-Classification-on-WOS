from preprocess import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib



train, test = get_dataset(clean_text=False)

def create_classifier(X: list[str], y: list[int], X_test: list[str], y_test: list[int], do_hpo=False):
    """
    This will find the best classifier for the given data and return it + the test accuracy
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # going to test SVM and Random Forest
    svm_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', SVC(probability=True))
    ])

    rf_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    svm_search_space = {
        'C': Real(1e-6, 1e+6, prior='log-uniform'),
        'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf', 'poly']),
        'degree': Integer(1, 5)
    }

    rf_search_space = {
        'n_estimators': Integer(10, 500),
        'max_depth': Integer(1, 50),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'bootstrap': Categorical([True, False])
    }
    
    if do_hpo:
        svm_pipeline = BayesSearchCV(
            svm_pipeline,
            svm_search_space,
            n_iter=50,
            cv=3,
            n_jobs=-1,
            random_state=42
        )

        rf_pipeline = BayesSearchCV(
            rf_pipeline,
            rf_search_space,
            n_iter=50,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
        

    # train
    svm_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)
    
    if do_hpo:
        # Print best parameters
        print("best svm params")
        print(svm_pipeline.best_params_)
        print("best rf params")
        print(rf_pipeline.best_params_)

    # validate
    print("svm val perf")
    y_pred_svm = svm_pipeline.predict(X_val)
    print(classification_report(y_val, y_pred_svm))
    acc_svm = accuracy_score(y_val, y_pred_svm)
    print(f"Accuracy: {acc_svm:.4f}")

    print("Dtree val perf")
    y_pred_dt = rf_pipeline.predict(X_val)
    print(classification_report(y_val, y_pred_dt))
    acc_rf = accuracy_score(y_val, y_pred_dt)
    print(f"Accuracy: {acc_rf:.4f}")

    threshold = 0.1
    # defaults to RF
    best_model_name = "rf"
    best_model = rf_pipeline
    if acc_rf - acc_svm > threshold:
        print("Random Forest has better performance, using it for testing")
        best_model = rf_pipeline
        best_model_name = "rf"
    elif acc_svm - acc_rf > threshold:
        print("SVM has better performance, using it for testing")
        best_model = svm_pipeline
        best_model_name = "svm"
        
    print("\nTest Results:")
    y_pred_test = best_model.predict(X_test)  
    print(classification_report(y_test, y_pred_test))
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    
    # save the model
    if not do_hpo:
        return best_model, best_model_name, y_pred_test
    else:
        return best_model.best_estimator_, best_model_name, y_pred_test


# Train the Top level classifier
best_model, best_model_name, y_pred = create_classifier(
    X=train["X"].tolist(),
    y=train["Y1"].tolist(),
    X_test=test["X"].tolist(),
    y_test=test["Y1"].tolist(),
    do_hpo=False
)

joblib.dump(best_model, f'top_level_{best_model_name}_model.pkl')

test["y_pred"] = y_pred
train_splitted_df = {category: group for category, group in train.groupby('Y1')}
# use pred from top level classifier for splitting so that acc of the entire system can be calculated
# there is only one category which overalps at level 2 with other level 2 ids, so this should mostly work
test_splitted_df = {category: group for category, group in test.groupby('y_pred')}


# category wise training
for category in train_splitted_df.keys():
    print(f"\nTraining for category {category}")
    best_model, best_model_name, _ = create_classifier(
        X=train_splitted_df[category]["X"].tolist(),
        y=train_splitted_df[category]["Y2"].tolist(),
        X_test=test_splitted_df[category]["X"].tolist(),
        y_test=test_splitted_df[category]["Y2"].tolist(),
        do_hpo=False
    )
    joblib.dump(best_model, f'{category}_{best_model_name}_model.pkl')
import pdb; pdb.set_trace()
print()