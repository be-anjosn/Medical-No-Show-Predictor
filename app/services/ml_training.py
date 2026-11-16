import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_validate, train_test_split
from catboost import CatBoostClassifier, Pool, cv
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import lightgbm as lgb
from xgboost import XGBClassifier

# ==========================================
# 1. DATA LOADING AND PREPARATION
# ==========================================
df = pd.read_csv('../../data/df_new.csv', delimiter=',')
target = df['No-show']
features = df.drop('No-show', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

# ==========================================
# 2. LIGHTGBM
# ==========================================
def run_lightgbm():
    print("="*40)
    print("LIGHTGBM")
    print("="*40)
    
    model_lgbm = lgb.LGBMClassifier(
        num_leaves=150, 
        max_depth=4, 
        learning_rate=0.05, 
        max_bin=100,
        n_estimators=200,        
        class_weight='balanced',  
        random_state=0,
        verbose=-1
    )   

    model_lgbm.fit(X_train, y_train)
    
    probabilities = model_lgbm.predict_proba(X_test)[:, 1]
    adjusted_predictions = (probabilities >= 0.65).astype(int)

    print("Test Accuracy with Threshold 0.65: %.2f%%" % (accuracy_score(y_test, adjusted_predictions) * 100.0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, adjusted_predictions))
    print("\nClassification Report:\n", classification_report(y_test, adjusted_predictions))
    print()

# ==========================================
# 3. XGBOOST
# ==========================================
def run_xgboost():
    print("="*40)
    print("OPTIMIZED XGBOOST")
    print("="*40)
    
    balance_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model_xgb = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        scale_pos_weight=balance_weight,
        random_state=0,
        eval_metric='logloss'
    )

    model_xgb.fit(X_train, y_train)
    
    probabilities = model_xgb.predict_proba(X_test)[:, 1]
    adjusted_predictions = (probabilities >= 0.65).astype(int)

    print("Test Accuracy with Threshold 0.65: %.2f%%" % (accuracy_score(y_test, adjusted_predictions) * 100.0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, adjusted_predictions))
    print("\nClassification Report:\n", classification_report(y_test, adjusted_predictions))
    print()

# ==========================================
# 4. RANDOM FOREST
# ==========================================
def run_random_forest():
    print("="*40)
    print("OPTIMIZED RANDOM FOREST")
    print("="*40)
    
    model_rf = RandomForestClassifier(
        n_estimators=150, 
        criterion='entropy', 
        random_state=0, 
        max_depth=10,            
        class_weight='balanced',  
        n_jobs=-1                
    )

    model_rf.fit(X_train, y_train)
    
    probabilities = model_rf.predict_proba(X_test)[:, 1]
    adjusted_predictions = (probabilities >= 0.65).astype(int)

    print("Test Accuracy with Threshold 0.65: %.2f%%" % (accuracy_score(y_test, adjusted_predictions) * 100.0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, adjusted_predictions))
    print("\nClassification Report:\n", classification_report(y_test, adjusted_predictions))
    print()

# ==========================================
# 5. DECISION TREE
# ==========================================
def run_decision_tree():
    print("="*40)
    print("DECISION TREE")
    print("="*40)
    
    model_tree = DecisionTreeClassifier(
        criterion='gini', 
        random_state=0, 
        max_depth=10,            
        class_weight='balanced'   
    )

    model_tree.fit(X_train, y_train)
    probabilities = model_tree.predict_proba(X_test)[:, 1]
    adjusted_predictions = (probabilities >= 0.65).astype(int)

    print("Test Accuracy with Threshold 0.65: %.2f%%" % (accuracy_score(y_test, adjusted_predictions) * 100.0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, adjusted_predictions))
    print("\nClassification Report:\n", classification_report(y_test, adjusted_predictions))
    print()

# ==========================================
# MAIN MODELS EXECUTION
# ==========================================
if __name__ == "__main__":
    run_lightgbm()
    run_xgboost()
    run_random_forest()
    run_decision_tree()