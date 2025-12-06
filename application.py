import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Function to load local background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@600&display=swap');
    
    .stApp {{
        background-image: url(data:image/jpg;base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    div.stSelectbox, div.stNumberInput {{
        background-color: rgba(0, 0, 0, 0.8);
        border-radius: 8px;
        padding: 12px;
        backdrop-filter: blur(12px);
        border: 3px solid #E31837;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.7);
        margin-bottom: 15px;
    }}
    
    div.stSelectbox > div > div {{
        background-color: white !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }}
    
    .stNumberInput > div > div > input {{
        background-color: white !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }}
    
    /* Labels for the boxes */
    .stSelectbox label, .stNumberInput label {{
        color: #FFD700 !important;
        text-shadow: 2px 2px 3px black;
        font-weight: bold !important;
        font-size: 18px !important;
        background-color: rgba(0, 0, 0, 0.8);
        padding: 6px 12px;
        border-radius: 5px;
        border-left: 3px solid #E31837;
    }}
    
    .title-text {{
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.8);
        border-radius: 10px;
        margin-bottom: 20px;
        font-family: 'Bebas Neue', sans-serif;
        text-shadow: 2px 2px 4px #E31837;
        border: 2px solid #E31837;
        box-shadow: 0 0 20px rgba(227, 24, 55, 0.5);
    }}

    .stButton > button {{
        background-color: #E31837;
        color: white;
        font-weight: bold;
        font-size: 20px;
        padding: 12px 30px;
        border-radius: 8px;
        border: 2px solid gold;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        margin-top: 10px;
    }}
    
    .result-text {{
        color: white;
        font-size: 28px;
        font-weight: bold;
        background-color: rgba(0, 0, 0, 0.85);
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0px;
        border-left: 6px solid #E31837;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.6);
    }}
    
    .team-name {{
        font-family: 'Teko', sans-serif;
        color: #E31837;
        font-size: 35px;
        text-shadow: 2px 2px 3px black;
    }}

    .probability {{
        font-size: 32px;
        color: gold;
        text-shadow: 1px 1px 2px black;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Load the background image
add_bg_from_local('background_image.jpg')

# Title with custom styling
st.markdown("<h1 class='title-text'>IPL WINNER PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
        @font-face {
            font-family: 'Facon';
            src: url('Facon.ttf') format('truetype');
        }
        .title-text {
            font-family: 'Facon', sans-serif;
            font-size: 32px;
            font-weight: bold;
            color: #FF4500;
            text-shadow: 2px 2px 4px #000000;
            text-align: center;
        }
    </style>
    <p class='title-text'>Archit's IPL Predictor</p>
""", unsafe_allow_html=True)

team_list = ['Royal Challengers Bangalore','Kolkata Knight Riders',
            'Punjab Kings','Delhi Capitals',
            'Sunrisers Hyderabad','Mumbai Indians',
            'Gujrat Titans','Rajasthan Royals',
            'Chennai Super Kings']

city_list = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
       'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
       'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
       'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 
       'Mohali', 'Bengaluru']

pipe_variable = pickle.load(open('ipl_pred.pkl','rb'))

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(team_list))

with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(team_list))

selected_city = st.selectbox('Select the host city',sorted(city_list))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score')

with col4:
    overs = st.number_input('Overs completed')
    
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'target': [target],
        'runs_left': [runs_left],
        'remaining_balls': [balls_left],
        'wickets_rem': [wickets_left],
        'crr': [crr],
        'rrr': [rrr]
    })
    
    result = pipe_variable.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    
    st.markdown(f"""
    <div class='result-text'>
        <span class='team-name'>{batting_team}</span> winning probability: 
        <span class='probability'>{win*100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='result-text'>
        <span class='team-name'>{bowling_team}</span> winning probability: 
        <span class='probability'>{loss*100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
    
    
