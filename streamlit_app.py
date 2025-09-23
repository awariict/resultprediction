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
        admin_action = st.selectbox('Action', ['Login','Register Admin'], key='admin_action_select')
        if admin_action == 'Register Admin':
            au = st.text_input('Admin username', key='admin_register_username')
            apw = st.text_input('Password', type='password', key='admin_register_password')
            if st.button('Register', key='admin_register_btn'):
                ok,msg = register_admin(au,apw)
                if ok:
                    st.success(msg)
                    # rerun to refresh admin UI (so operations list appears)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            au = st.text_input('Admin username (login)', key='admin_login_username')
            apw = st.text_input('Password (login)', type='password', key='admin_login_password')
            if st.button('Login', key='admin_login_btn'):
                ok,role = authenticate(au,apw)
                if ok and role == 'admin':
                    st.session_state['role']='admin'
                    st.session_state['username']=au
                    st.success('Logged in as admin')
                    st.rerun()
                else:
                    st.error('Invalid credentials')

    with col2:
        if 'role' in st.session_state and st.session_state['role']=='admin':
            st.subheader('Operations')
            action = st.selectbox('Choose operation', ['Register Student','Single Questionnaire','Upload CSV & Batch Predict','View History','View Student Replies'], key='admin_ops_select')

            if action == 'Register Student':
                reg = st.text_input('Student Reg. Number', key='admin_create_student_reg')
                if st.button('Create Student', key='admin_create_student_btn'):
                    if reg:
                        ok,msg = register_student(str(reg))
                        if ok:
                            st.success(msg + ' (password = reg number)')
                        else:
                            st.error(msg)
                    else:
                        st.error('Enter reg number')

            elif action == 'Single Questionnaire':
                if os.path.exists(MODEL_PATH):
                    meta = joblib.load(MODEL_PATH)
                    features = meta['features']
                    # create inputs with unique keys to avoid duplicate element ids
                    inputs = {}
                    for i, f in enumerate(features):
                        inputs[f] = st.text_input(f, key=f"admin_single_in_{i}")
                    if st.button('Predict & Save', key='admin_single_predict_btn'):
                        try:
                            df_row = pd.DataFrame([inputs])
                            preds, _ = predict_with_existing_model(df_row)
                            cat = preds[0]
                            advice = make_advice(cat)
                            st.success(f'Category: {cat}')
                            st.write(advice)
                            rec = {'reg_number': '', 'predicted_category': cat, 'advice': advice, 'admin': st.session_state.get('username',''), 'timestamp': datetime.utcnow()}
                            history_col.insert_one(rec)
                            hdf = pd.read_csv(HISTORY_CSV)
                            hdf = pd.concat([hdf, pd.DataFrame([rec])], ignore_index=True)
                            hdf.to_csv(HISTORY_CSV, index=False)
                        except Exception as e:
                            st.error(f'Prediction failed: {e}')
                else:
                    st.info('No trained model found. Upload CSV to train.')

            elif action == 'Upload CSV & Batch Predict':
                uploaded = st.file_uploader('Upload students CSV', type=['csv'], key='admin_upload_csv')
                reg_col_hint = st.text_input('Registration column name (optional)', key='admin_reg_col_hint')
                if uploaded is not None and st.button('Train & Predict', key='admin_train_predict_btn'):
                    try:
                        df_uploaded = pd.read_csv(uploaded)
                        meta, train_df = build_and_train_model(df_uploaded, reg_col_hint)
                        preds, _ = predict_with_existing_model(df_uploaded)
                        df_uploaded['predicted_category'] = preds
                        df_uploaded['advice'] = df_uploaded['predicted_category'].apply(make_advice)
                        reg_col = next((c for c in [reg_col_hint,'Registration Number','reg_number','RegNo','Reg_Number'] if c in df_uploaded.columns), None)
                        created, pass_count, fail_count, records = 0,0,0,[]
                        for _,row in df_uploaded.iterrows():
                            regval = str(row.get(reg_col,'')) if reg_col else ''
                            if regval and not users_col.find_one({'username':regval}):
                                users_col.insert_one({'username':regval,'password_hash':hash_password(regval),'role':'student','reg_number':regval})
                                udf = pd.read_csv(USERS_CSV)
                                udf = pd.concat([udf, pd.DataFrame([{'username':regval,'password_hash':hash_password(regval),'role':'student','reg_number':regval}])], ignore_index=True)
                                udf.to_csv(USERS_CSV,index=False)
                                created += 1
                            rec = {'reg_number': regval, 'predicted_category': row['predicted_category'], 'advice': row['advice'], 'admin': st.session_state.get('username',''), 'timestamp': datetime.utcnow()}
                            records.append(rec)
                            pass_count += 1 if row['predicted_category'] in ['average','high'] else 0
                            fail_count += 1 if row['predicted_category'] == 'low' else 0
                        if records:
                            history_col.insert_many(records)
                            hdf = pd.read_csv(HISTORY_CSV)
                            hdf = pd.concat([hdf, pd.DataFrame(records)], ignore_index=True)
                            hdf.to_csv(HISTORY_CSV, index=False)
                        st.success(f'Bulk prediction done. Created {created} students.')
                        st.info(f'Pass: {pass_count} | Fail: {fail_count}')
                        st.dataframe(df_uploaded.head(200))
                        st.download_button('Download predictions CSV', df_uploaded.to_csv(index=False).encode('utf-8'), file_name='predictions.csv', key='admin_download_preds')
                    except Exception as e:
                        st.error(f'Error during training/prediction: {e}')

            elif action == 'View History':
                q = st.text_input('Filter by reg number (optional)', key='admin_view_history_filter')
                # build cursor and sort correctly, then limit & list
                if q:
                    cursor = history_col.find({'reg_number': str(q)})
                else:
                    cursor = history_col.find()
                cursor = cursor.sort('timestamp', -1).limit(500)
                recs = list(cursor)
                if recs:
                    df_hist = pd.DataFrame(recs)
                    # show safe subset of fields
                    cols = [c for c in ['reg_number','predicted_category','advice','admin','timestamp'] if c in df_hist.columns]
                    st.dataframe(df_hist[cols])
                    st.download_button('Download history CSV', df_hist.to_csv(index=False).encode('utf-8'), file_name='advice_history.csv', key='admin_download_history')
                else:
                    st.info('No history records found.')

            elif action == 'View Student Replies':
                cursor = history_col.find({'advice': {'$regex':'STUDENT_REPLY'}}).sort('timestamp', -1).limit(200)
                recs = list(cursor)
                if recs:
                    df_rep = pd.DataFrame(recs)
                    cols = [c for c in ['reg_number','advice','timestamp'] if c in df_rep.columns]
                    st.dataframe(df_rep[cols])
                else:
                    st.info('No student replies yet.')

# --- STUDENT PORTAL ---
with tabs[1]:
    st.header('Student Portal')
    if 'role' in st.session_state and st.session_state['role']=='student':
        reg = st.session_state.get('username')
        st.subheader(f'Welcome {reg}')
        # fetch and show recent history for this student
        cursor = history_col.find({'reg_number': str(reg)}).sort('timestamp', -1).limit(100)
        recs = list(cursor)
        if recs:
            df_recs = pd.DataFrame(recs)
            cols = [c for c in ['predicted_category','advice','admin','timestamp'] if c in df_recs.columns]
            st.dataframe(df_recs[cols])
        else:
            st.info('No advice yet.')
        reply = st.text_area('Reply to Admin', key='student_reply_box')
        if st.button('Send Reply', key='student_send_reply_btn'):
            if reply.strip():
                rec = {'reg_number': reg, 'predicted_category':'', 'advice': f'STUDENT_REPLY: {reply}', 'admin': reg, 'timestamp': datetime.utcnow()}
                history_col.insert_one(rec)
                hdf = pd.read_csv(HISTORY_CSV)
                hdf = pd.concat([hdf, pd.DataFrame([rec])], ignore_index=True)
                hdf.to_csv(HISTORY_CSV, index=False)
                st.success('Reply saved')
            else:
                st.error('Empty reply')
    else:
        sreg = st.text_input('Registration number', key='student_login_reg')
        spw = st.text_input('Password (use reg number)', type='password', key='student_login_password')
        if st.button('Login as Student', key='student_login_btn'):
            ok, role = authenticate(sreg, spw)
            if ok and role=='student':
                st.session_state['role']='student'
                st.session_state['username']=sreg
                st.rerun()
            else:
                st.error('Invalid credentials')

# --- ABOUT & HELP ---
with tabs[2]:
    st.header('About & Help')
    st.write('This application predicts student grade categories (low / average / high) using a soft-voting ensemble of KNN and Naive Bayes. Admins can register students, upload CSVs, train models, make predictions, and view advice history & student replies. Students can login with their registration number to view advice and reply.')
    st.write('Pass/Fail summary: Pass = average/high, Fail = low')


