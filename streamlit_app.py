import streamlit as st
import pandas as pd
import sqlite3
import os
import pickle
from sklearn.linear_model import LogisticRegression

# --------------------------
# DATABASE SETUP
# --------------------------
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 username TEXT PRIMARY KEY,
                 password TEXT,
                 role TEXT)''')
    conn.commit()
    conn.close()

def register_admin(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, password, "admin"))
        conn.commit()
        return True, "Admin registered successfully"
    except sqlite3.IntegrityError:
        return False, "Admin already exists"
    finally:
        conn.close()

def register_student(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, username, "student"))
        conn.commit()
        return True, "Student registered successfully"
    except sqlite3.IntegrityError:
        return False, "Student already exists"
    finally:
        conn.close()

def authenticate(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    if result:
        return True, result[0]
    else:
        return False, None

# --------------------------
# MODEL TRAINING
# --------------------------
MODEL_FILE = "model.pkl"

def train_and_save_model():
    X = pd.DataFrame({
        "Feature1": [0, 1, 2, 3, 4],
        "Feature2": [5, 4, 3, 2, 1]
    })
    y = [0, 1, 0, 1, 0]
    model = LogisticRegression()
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

if not os.path.exists(MODEL_FILE):
    train_and_save_model()

def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

# --------------------------
# STREAMLIT APP
# --------------------------
st.title("Student Result Prediction System")

# init db
init_db()

if "role" not in st.session_state:
    st.session_state["role"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

menu = ["Home", "Admin Portal", "Student Portal"]
choice = st.sidebar.selectbox("Menu", menu, key="main_menu")

# --------------------------
# HOME
# --------------------------
if choice == "Home":
    st.write("Welcome! Please login or register using the side menu.")

# --------------------------
# ADMIN PORTAL
# --------------------------
elif choice == "Admin Portal":
    st.subheader("Admin Portal")

    tab = st.radio("Choose action:", ["Register Admin", "Login as Admin", "Register Student"], key="admin_tabs")

    # --- Register Admin
    if tab == "Register Admin":
        au = st.text_input("Admin username", key="admin_register_username")
        apw = st.text_input("Password", type="password", key="admin_register_password")
        if st.button("Register", key="admin_register_btn"):
            ok, msg = register_admin(au, apw)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    # --- Login as Admin
    elif tab == "Login as Admin":
        au = st.text_input("Admin username (login)", key="admin_login_username")
        apw = st.text_input("Password (login)", type="password", key="admin_login_password")
        if st.button("Login", key="admin_login_btn"):
            ok, role = authenticate(au, apw)
            if ok and role == "admin":
                st.session_state["role"] = "admin"
                st.session_state["username"] = au
                st.success("Logged in as admin")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # --- Register Student
    elif tab == "Register Student":
        reg = st.text_input("Student Reg. Number", key="admin_create_student_reg")
        if st.button("Create Student", key="admin_create_student_btn"):
            if reg:
                ok, msg = register_student(str(reg))
                if ok:
                    st.success(msg + " (password = reg number)")
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Enter reg number")

# --------------------------
# STUDENT PORTAL
# --------------------------
elif choice == "Student Portal":
    st.subheader("Student Portal")

    if st.session_state["role"] != "student":
        sreg = st.text_input("Reg Number", key="student_login_reg")
        spw = st.text_input("Password", type="password", key="student_login_password")
        if st.button("Login as Student", key="student_login_btn"):
            ok, role = authenticate(sreg, spw)
            if ok and role == "student":
                st.session_state["role"] = "student"
                st.session_state["username"] = sreg
                st.success("Logged in as student")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.success(f"Welcome student {st.session_state['username']}")

        # Prediction options
        subtab = st.radio("Choose action:", ["Single Questionnaire", "Upload CSV", "History"], key="student_tabs")

        if subtab == "Single Questionnaire":
            feat1 = st.number_input("Feature 1", value=0.0, key="single_feat1")
            feat2 = st.number_input("Feature 2", value=0.0, key="single_feat2")
            if st.button("Predict", key="single_predict_btn"):
                try:
                    model = load_model()
                    X_new = pd.DataFrame({"Feature1": [feat1], "Feature2": [feat2]})
                    pred = model.predict(X_new)[0]
                    st.success(f"Prediction result: {pred}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        elif subtab == "Upload CSV":
            file = st.file_uploader("Upload CSV", type="csv", key="upload_csv")
            if file:
                try:
                    df = pd.read_csv(file)
                    model = load_model()
                    preds = model.predict(df)
                    df["Prediction"] = preds
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error during training/prediction: {e}")

        elif subtab == "History":
            st.info("History feature not implemented yet.")

        # logout
        if st.button("Logout", key="student_logout_btn"):
            st.session_state["role"] = None
            st.session_state["username"] = None
            st.success("Logged out")
            st.rerun()
