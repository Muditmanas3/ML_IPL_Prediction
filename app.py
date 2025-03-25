import streamlit as st
import pandas as pd
import pickle
import warnings
from sklearn import __version__ as sklearn_version

warnings.filterwarnings("ignore")

# Version check
st.sidebar.write(f"Scikit-learn version: {sklearn_version}")

# Load model with compatibility check
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"""
    Model loading failed. This is likely due to scikit-learn version mismatch.
    Error: {str(e)}
    Required version: 1.3.2
    Current version: {sklearn_version}
    """)
    st.stop()

# UI Components
st.title('IPL Win Predictor')

teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
])

cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# Input Layout
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Bowling Team', teams)

selected_city = st.selectbox('Match Location', cities)
target = st.number_input('Target Runs', min_value=1, value=150)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, max_value=target-1)
with col4:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=9)
with col5:
    overs = st.number_input('Overs Completed', min_value=0, max_value=20)

# Prediction Logic
if st.button('Predict Winning Probability'):
    try:
        # Calculate match parameters
        runs_left = max(target - score, 0)
        balls_left = max(120 - (overs * 6), 1)  # Ensure at least 1 ball
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if runs_left > 0 else 0
        
        # Create input DataFrame with exact column names expected by the model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })
        
        # Get probabilities
        probabilities = pipe.predict_proba(input_df)
        batting_prob = round(probabilities[0][1] * 100)
        bowling_prob = round(probabilities[0][0] * 100)
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"{batting_team}", value=f"{batting_prob}%")
        with col2:
            st.metric(label=f"{bowling_team}", value=f"{bowling_prob}%")
            
        # Visual indicator
        st.progress(batting_prob/100)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Common issues:")
        st.write("- Division by zero (check overs/balls inputs)")
        st.write("- Model expects specific column names/order")
        st.write(f"Input data shape: {input_df.shape if 'input_df' in locals() else 'N/A'}")
