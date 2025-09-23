# streamlit_app.py
# Professional Result Prediction & Advice System
# Voting ensemble: KNN + Naive Bayes

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from pymongo import MongoClient

# ----------------- Config -----------------
MONGO_URI = "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/result_prediction_db?retryWrites=true&w=majority"
DB_NAME = "result_prediction_db"
USERS_COLLECTION = "users"
HISTORY_COLLECTION = "history"
MODEL_PATH = "ensemble_model.joblib"
USERS_CSV = "users_backup.csv"
HISTORY_CSV = "advice_history_backup.csv"


# ----------------- Setup -----------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db[USERS_COLLECTION]
history_col = db[HISTORY_COLLECTION]

# ensure CSV backups exist
if not os.path.exists(USERS_CSV):
    pd.DataFrame(columns=['username','password_hash','role','reg_number']).to_csv(USERS_CSV,index=False)
if not os.path.exists(HISTORY_CSV):
    pd.DataFrame(columns=['reg_number','predicted_category','advice','admin','timestamp']).to_csv(HISTORY_CSV,index=False)

# ----------------- Utilities -----------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_admin(username: str, password: str) -> tuple:
    if users_col.find_one({"username": username}):
        return False, "Username already exists"
    rec = {"username": username, "password_hash": hash_password(password), "role": "admin"}
    users_col.insert_one(rec)
    df = pd.read_csv(USERS_CSV)
    df = pd.concat([df, pd.DataFrame([{**rec, "reg_number": ""}])], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True, "Admin registered"

def register_student(reg_number: str) -> tuple:
    if users_col.find_one({"username": reg_number}):
        return False, "Student already exists"
    rec = {"username": reg_number, "password_hash": hash_password(reg_number), "role": "student", "reg_number": reg_number}
    users_col.insert_one(rec)
    df = pd.read_csv(USERS_CSV)
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True, "Student registered"

def authenticate(username: str, password: str) -> tuple:
    user = users_col.find_one({"username": username})
    if user and user.get('password_hash') == hash_password(password):
        return True, user.get('role')
    df = pd.read_csv(USERS_CSV)
    match = df[(df['username']==username) & (df['password_hash']==hash_password(password))]
    if not match.empty:
        return True, match.iloc[0]['role']
    return False, None

# ----------------- ML & Helpers -----------------
def infer_score_column(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    for cand in ['next_score','next_semester_score','score','final_score','percentage','gpa','GPA']:
        if cand in df.columns:
            return cand
    return numeric_cols[0]

def to_letter_score(x, scale='percent'):
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
    except:
        return np.nan
    if scale == 'percent':
        if x < 40: return 'D_or_lower'
        if x < 50: return 'C'
        if x < 60: return 'B'
        return 'A'
    else:
        if x < 2.0: return 'D_or_lower'
        if x < 2.5: return 'C'
        if x < 3.5: return 'B'
        return 'A'

def map_category(letter):
    if pd.isna(letter):
        return np.nan
    if letter == 'D_or_lower':
        return 'low'
    if letter == 'C':
        return 'average'
    return 'high'

def build_and_train_model(df: pd.DataFrame, reg_col_hint: str = None):
    score_col = infer_score_column(df)
    if score_col is None:
        raise ValueError("No numeric score column found in the uploaded CSV.")
    max_score = df[score_col].max()
    scale = 'percent' if (pd.notna(max_score) and float(max_score) > 12) else 'gpa'
    df['_letter_grade'] = df[score_col].apply(lambda x: to_letter_score(x, scale=scale))
    df['grade_category'] = df['_letter_grade'].apply(map_category)
    df = df.dropna(subset=['grade_category']).copy()
    exclude = [score_col, '_letter_grade', 'grade_category']
    if reg_col_hint and reg_col_hint in df.columns:
        exclude.append(reg_col_hint)
    features = [c for c in df.columns if c not in exclude]
    if not features:
        raise ValueError("No features available to train on.")
    X, y = df[features], df['grade_category']
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    knn, nb = KNeighborsClassifier(n_neighbors=5), GaussianNB()
    ensemble = VotingClassifier([('knn', knn), ('nb', nb)], voting='soft')
    pipeline = Pipeline([('preprocessor', preprocessor), ('clf', ensemble)])
    pipeline.fit(X, y)
    meta = {'model': pipeline, 'features': features, 'score_col': score_col, 'scale': scale, 'reg_col': reg_col_hint}
    joblib.dump(meta, MODEL_PATH)
    return meta, df

def predict_with_existing_model(df: pd.DataFrame):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No trained model found. Train a model first.")
    meta = joblib.load(MODEL_PATH)
    model, features = meta['model'], meta['features']
    X = df.reindex(columns=features).copy()
    for c in features:
        if c not in X.columns:
            X[c] = pd.NA
    preds = model.predict(X)
    return preds, meta

def make_advice(category: str) -> str:
    if category == 'low':
        return "Attend remedial classes, meet your lecturer, focus on fundamentals, and seek counseling."
    if category == 'average':
        return "Target weak topics, join study groups, and track progress weekly."
    return "Good performance — aim higher with advanced readings and peer-teaching."

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title='Result Prediction System', layout='wide')
st.title('Result Prediction & Advice System')

tabs = st.tabs(["Admin Portal","Student Portal","About & Help"])

# --- ADMIN PORTAL ---
with tabs[0]:
    st.header('Admin Portal')
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader('Admin Access')
        admin_action = st.selectbox('Action', ['Login','Register Admin'])
        if admin_action == 'Register Admin':
            au = st.text_input('Admin username')
            apw = st.text_input('Password', type='password')
            if st.button('Register'):
                ok, msg = register_admin(au, apw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            au = st.text_input('Admin username (login)')
            apw = st.text_input('Password (login)', type='password')
            if st.button('Login'):
                ok,role = authenticate(au,apw)
                if ok and role == 'admin':
                    st.session_state['role']='admin'
                    st.session_state['username']=au
                    st.success('Logged in as admin')
                else:
                    st.error('Invalid credentials')

    with col2:
        if 'role' in st.session_state and st.session_state['role']=='admin':
            st.subheader('Operations')
            action = st.selectbox('Choose operation', ['Register Student','Single Questionnaire','Upload CSV & Batch Predict','View History','View Student Replies'])

            if action == 'Register Student':
                reg = st.text_input('Student Reg. Number')
                if st.button('Create Student'):
                    if reg:
                        ok, msg = register_student(str(reg))
                        if ok:
                            st.success(msg + ' (password = reg number)')
                        else:
                            st.error(msg)
                    else:
                        st.error('Enter reg number')

            elif action == 'Single Questionnaire':
                st.write("Enter features for prediction (example input, adapt as needed)")
                feat1 = st.number_input("Feature 1", value=0.0)
                feat2 = st.number_input("Feature 2", value=0.0)
                df_in = pd.DataFrame([{"feat1":feat1,"feat2":feat2}])
                if st.button("Predict Single"):
                    try:
                        preds, meta = predict_with_existing_model(df_in)
                        pred, advice = preds[0], make_advice(preds[0])
                        st.success(f"Predicted Category: {pred}")
                        st.info(f"Advice: {advice}")
                    except Exception as e:
                        st.error(f'Prediction failed: {e}')

            elif action == 'Upload CSV & Batch Predict':
                file = st.file_uploader("Upload CSV", type='csv')
                if file:
                    df = pd.read_csv(file)
                    st.write("Preview:", df.head())
                    reg_col = st.text_input("Enter Reg Number column name (if any)")
                    if st.button("Train & Predict"):
                        try:
                            meta, trained_df = build_and_train_model(df, reg_col)
                            preds, meta = predict_with_existing_model(df)
                            advices = [make_advice(p) for p in preds]
                            df['predicted_category'] = preds
                            df['advice'] = advices
                            st.dataframe(df[['predicted_category','advice']].head())
                            if reg_col and reg_col in df.columns:
                                for i,row in df.iterrows():
                                    history_col.insert_one({'reg_number':row[reg_col],'predicted_category':row['predicted_category'],'advice':row['advice'],'admin':st.session_state['username'],'timestamp':datetime.utcnow()})
                                df_hist = pd.read_csv(HISTORY_CSV)
                                new_rows = pd.DataFrame([{'reg_number':row[reg_col],'predicted_category':row['predicted_category'],'advice':row['advice'],'admin':st.session_state['username'],'timestamp':datetime.utcnow()} for _,row in df.iterrows()])
                                df_hist = pd.concat([df_hist,new_rows],ignore_index=True)
                                df_hist.to_csv(HISTORY_CSV,index=False)
                                st.success("Predictions saved to history.")
                        except Exception as e:
                            st.error(f'Error during training/prediction: {e}')

            elif action == 'View History':
                data = list(history_col.find({}))
                if data:
                    st.write(pd.DataFrame(data))
                else:
                    st.info("No history found.")

            elif action == 'View Student Replies':
                st.info("Feature not implemented yet.")

# --- STUDENT PORTAL ---
with tabs[1]:
    st.header('Student Portal')
    sreg = st.text_input('Reg Number')
    spw = st.text_input('Password', type='password')
    if st.button('Login as Student'):
        ok, role = authenticate(sreg, spw)
        if ok and role=='student':
            st.session_state['role']='student'
            st.session_state['username']=sreg
            st.experimental_rerun()
        else:
            st.error('Invalid credentials')
    if 'role' in st.session_state and st.session_state['role']=='student':
        st.success(f"Logged in as student {st.session_state['username']}")
        data = list(history_col.find({'reg_number':st.session_state['username']}))
        if data:
            st.write(pd.DataFrame(data))
        else:
            st.info("No advice history yet.")

# --- ABOUT & HELP ---
with tabs[2]:
    st.header('About this System')
    st.write("This system predicts students' future performance based on questionnaire/CSV data using an ensemble of KNN and Naive Bayes. Admins can train models and register students. Students can view predictions and advice.")
