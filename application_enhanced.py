import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import requests
from fuzzywuzzy import fuzz, process
import ollama
import json
from datetime import datetime, timedelta
import warnings

# Page configuration with favicon
st.set_page_config(
    page_title="Advanced AI Powered IPL Prediction Model",
    page_icon="favicon.png",
    layout="wide"
)

# Suppress sklearn version warnings when loading pickle file
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

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
    
    div.stSelectbox, div.stNumberInput, div.stTextInput {{
        background-color: rgba(0, 0, 0, 0.8);
        border-radius: 8px;
        padding: 12px;
        backdrop-filter: blur(12px);
        border: 3px solid #E31837;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.7);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }}
    
    div.stSelectbox:hover, div.stNumberInput:hover, div.stTextInput:hover {{
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.9);
        transform: translateY(-2px);
    }}
    
    div.stSelectbox > div > div {{
        background-color: white !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }}
    
    .stNumberInput > div > div > input, .stTextInput > div > div > input {{
        background-color: white !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }}
    
    /* Labels for the boxes */
    .stSelectbox label, .stNumberInput label, .stTextInput label {{
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
    
    .ai-decision-box {{
        color: white;
        font-size: 20px;
        background-color: rgba(0, 0, 0, 0.9);
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0px;
        border: 3px solid #4CAF50;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.5);
    }}
    
    .backend-details {{
        color: #E0E0E0;
        font-size: 14px;
        background-color: rgba(0, 0, 0, 0.85);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0px;
        border-left: 4px solid #2196F3;
        font-family: 'Courier New', monospace;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Load the background image
add_bg_from_local('background_image.jpg')

# Title with custom styling
st.markdown("<h1 class='title-text'>Advanced AI Powered IPL Prediction Model</h1>", unsafe_allow_html=True)
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
    <p class='title-text'>Advanced AI Powered IPL Prediction Model</p>
""", unsafe_allow_html=True)

# City name mapping for weather API (wttr.in uses city names directly)
CITY_WEATHER_MAP = {
    'Hyderabad': 'Hyderabad',
    'Pune': 'Pune',
    'Rajkot': 'Rajkot',
    'Indore': 'Indore',
    'Bangalore': 'Bangalore',
    'Bengaluru': 'Bangalore',
    'Mumbai': 'Mumbai',
    'Kolkata': 'Kolkata',
    'Delhi': 'Delhi',
    'Chandigarh': 'Chandigarh',
    'Kanpur': 'Kanpur',
    'Jaipur': 'Jaipur',
    'Chennai': 'Chennai',
    'Ahmedabad': 'Ahmedabad',
    'Cuttack': 'Cuttack',
    'Nagpur': 'Nagpur',
    'Dharamsala': 'Dharamsala',
    'Kochi': 'Kochi',
    'Visakhapatnam': 'Visakhapatnam',
    'Raipur': 'Raipur',
    'Ranchi': 'Ranchi',
    'Mohali': 'Mohali',
    # International cities
    'Cape Town': 'Cape Town',
    'Port Elizabeth': 'Port Elizabeth',
    'Durban': 'Durban',
    'Centurion': 'Centurion',
    'East London': 'East London',
    'Johannesburg': 'Johannesburg',
    'Kimberley': 'Kimberley',
    'Bloemfontein': 'Bloemfontein',
    'Abu Dhabi': 'Abu Dhabi',
    'Sharjah': 'Sharjah',
}

# Sample player database (in production, this would come from an API or database)
SAMPLE_PLAYERS = {
    'Virat Kohli': {
        'matches': [
            {'runs': 45, 'balls': 32, 'strike_rate': 140.6, 'date': '2024-04-15'},
            {'runs': 67, 'balls': 45, 'strike_rate': 148.9, 'date': '2024-04-12'},
            {'runs': 23, 'balls': 18, 'strike_rate': 127.8, 'date': '2024-04-10'},
            {'runs': 89, 'balls': 58, 'strike_rate': 153.4, 'date': '2024-04-08'},
            {'runs': 34, 'balls': 28, 'strike_rate': 121.4, 'date': '2024-04-05'},
            {'runs': 56, 'balls': 42, 'strike_rate': 133.3, 'date': '2024-04-02'},
            {'runs': 78, 'balls': 52, 'strike_rate': 150.0, 'date': '2024-03-30'},
            {'runs': 41, 'balls': 35, 'strike_rate': 117.1, 'date': '2024-03-28'},
            {'runs': 62, 'balls': 48, 'strike_rate': 129.2, 'date': '2024-03-25'},
            {'runs': 55, 'balls': 40, 'strike_rate': 137.5, 'date': '2024-03-22'},
        ]
    },
    'MS Dhoni': {
        'matches': [
            {'runs': 28, 'balls': 22, 'strike_rate': 127.3, 'date': '2024-04-14'},
            {'runs': 35, 'balls': 28, 'strike_rate': 125.0, 'date': '2024-04-11'},
            {'runs': 42, 'balls': 30, 'strike_rate': 140.0, 'date': '2024-04-09'},
            {'runs': 19, 'balls': 15, 'strike_rate': 126.7, 'date': '2024-04-06'},
            {'runs': 48, 'balls': 35, 'strike_rate': 137.1, 'date': '2024-04-03'},
            {'runs': 31, 'balls': 24, 'strike_rate': 129.2, 'date': '2024-03-31'},
            {'runs': 25, 'balls': 20, 'strike_rate': 125.0, 'date': '2024-03-29'},
            {'runs': 38, 'balls': 29, 'strike_rate': 131.0, 'date': '2024-03-26'},
            {'runs': 22, 'balls': 18, 'strike_rate': 122.2, 'date': '2024-03-23'},
            {'runs': 40, 'balls': 31, 'strike_rate': 129.0, 'date': '2024-03-20'},
        ]
    },
    'Rohit Sharma': {
        'matches': [
            {'runs': 52, 'balls': 38, 'strike_rate': 136.8, 'date': '2024-04-13'},
            {'runs': 38, 'balls': 29, 'strike_rate': 131.0, 'date': '2024-04-10'},
            {'runs': 64, 'balls': 45, 'strike_rate': 142.2, 'date': '2024-04-07'},
            {'runs': 29, 'balls': 22, 'strike_rate': 131.8, 'date': '2024-04-04'},
            {'runs': 71, 'balls': 50, 'strike_rate': 142.0, 'date': '2024-04-01'},
            {'runs': 43, 'balls': 33, 'strike_rate': 130.3, 'date': '2024-03-29'},
            {'runs': 58, 'balls': 42, 'strike_rate': 138.1, 'date': '2024-03-27'},
            {'runs': 35, 'balls': 27, 'strike_rate': 129.6, 'date': '2024-03-24'},
            {'runs': 47, 'balls': 36, 'strike_rate': 130.6, 'date': '2024-03-21'},
            {'runs': 61, 'balls': 44, 'strike_rate': 138.6, 'date': '2024-03-18'},
        ]
    },
    'KL Rahul': {
        'matches': [
            {'runs': 39, 'balls': 31, 'strike_rate': 125.8, 'date': '2024-04-12'},
            {'runs': 54, 'balls': 41, 'strike_rate': 131.7, 'date': '2024-04-09'},
            {'runs': 28, 'balls': 23, 'strike_rate': 121.7, 'date': '2024-04-06'},
            {'runs': 66, 'balls': 48, 'strike_rate': 137.5, 'date': '2024-04-03'},
            {'runs': 41, 'balls': 32, 'strike_rate': 128.1, 'date': '2024-03-31'},
            {'runs': 49, 'balls': 37, 'strike_rate': 132.4, 'date': '2024-03-28'},
            {'runs': 33, 'balls': 26, 'strike_rate': 126.9, 'date': '2024-03-25'},
            {'runs': 57, 'balls': 43, 'strike_rate': 132.6, 'date': '2024-03-22'},
            {'runs': 44, 'balls': 34, 'strike_rate': 129.4, 'date': '2024-03-19'},
            {'runs': 50, 'balls': 38, 'strike_rate': 131.6, 'date': '2024-03-16'},
        ]
    },
    'Jasprit Bumrah': {
        'matches': [
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 3, 'economy': 6.2, 'date': '2024-04-13'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 2, 'economy': 7.1, 'date': '2024-04-10'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 4, 'economy': 5.8, 'date': '2024-04-07'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 1, 'economy': 8.3, 'date': '2024-04-04'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 3, 'economy': 6.5, 'date': '2024-04-01'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 2, 'economy': 7.4, 'date': '2024-03-29'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 3, 'economy': 6.0, 'date': '2024-03-27'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 1, 'economy': 8.1, 'date': '2024-03-24'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 2, 'economy': 7.2, 'date': '2024-03-21'},
            {'runs': 0, 'balls': 0, 'strike_rate': 0, 'wickets': 3, 'economy': 6.3, 'date': '2024-03-18'},
        ]
    },
}

def get_weather(city):
    """Get weather information for a city using wttr.in (free, no API key required)"""
    try:
        if city not in CITY_WEATHER_MAP:
            return None
        
        weather_city = CITY_WEATHER_MAP[city]
        # Using wttr.in API - completely free, no API key needed
        # Format: ?format=j1 returns JSON format
        url = f"https://wttr.in/{weather_city}?format=j1"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            current = data.get('current_condition', [{}])[0]
            
            # Extract weather data
            temp_c = float(current.get('temp_C', 0))
            condition = current.get('weatherDesc', [{}])[0].get('value', 'Unknown')
            humidity = int(current.get('humidity', 0))
            wind_speed_kmph = float(current.get('windspeedKmph', 0))
            
            return {
                'temperature': temp_c,
                'condition': condition,
                'humidity': humidity,
                'wind_speed': wind_speed_kmph
            }
        else:
            # Fallback to mock data if API fails
            return {
                'temperature': 32,
                'condition': 'Clear',
                'humidity': 65,
                'wind_speed': 12
            }
    except Exception as e:
        # Return mock data on error
        st.warning(f"Weather API error: {str(e)}. Using default values.")
        return {
            'temperature': 32,
            'condition': 'Clear',
            'humidity': 65,
            'wind_speed': 12
        }

def find_player_fuzzy(player_name, player_list):
    """Find player using fuzzy matching"""
    if not player_name:
        return None
    
    # Try exact match first (case insensitive)
    for p in player_list:
        if p.lower() == player_name.lower():
            return p
    
    # Use fuzzy matching
    best_match = process.extractOne(player_name, player_list, scorer=fuzz.ratio)
    if best_match and best_match[1] >= 70:  # 70% similarity threshold
        return best_match[0]
    return None

def calculate_player_form(player_name):
    """Calculate player form from last 10 matches"""
    player_name = find_player_fuzzy(player_name, list(SAMPLE_PLAYERS.keys()))
    
    if not player_name or player_name not in SAMPLE_PLAYERS:
        return None
    
    matches = SAMPLE_PLAYERS[player_name]['matches'][:10]  # Last 10 matches
    
    if not matches:
        return None
    
    # Calculate batting form (average runs, strike rate)
    total_runs = sum(m.get('runs', 0) for m in matches)
    total_balls = sum(m.get('balls', 0) for m in matches)
    avg_runs = total_runs / len(matches) if matches else 0
    avg_strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
    
    # For bowlers, calculate wickets and economy
    total_wickets = sum(m.get('wickets', 0) for m in matches)
    avg_wickets = total_wickets / len(matches) if matches else 0
    avg_economy = np.mean([m.get('economy', 0) for m in matches if 'economy' in m]) if any('economy' in m for m in matches) else 0
    
    # Recent form (last 3 matches)
    recent_matches = matches[:3]
    recent_runs = sum(m.get('runs', 0) for m in recent_matches)
    recent_balls = sum(m.get('balls', 0) for m in recent_matches)
    recent_strike_rate = (recent_runs / recent_balls * 100) if recent_balls > 0 else 0
    
    return {
        'player_name': player_name,
        'avg_runs': round(avg_runs, 2),
        'avg_strike_rate': round(avg_strike_rate, 2),
        'avg_wickets': round(avg_wickets, 2),
        'avg_economy': round(avg_economy, 2),
        'recent_strike_rate': round(recent_strike_rate, 2),
        'total_matches': len(matches),
        'is_bowler': total_wickets > 0
    }

def generate_ai_decision(model_output, weather_data, player_form):
    """Generate AI decision using Ollama"""
    try:
        # Check if Ollama is available
        try:
            # Test Ollama connection
            ollama.list()
        except Exception as e:
            return f"Ollama connection error: {str(e)}\n\nPlease ensure:\n1. Ollama is installed (https://ollama.ai)\n2. Ollama service is running\n3. llama3.2 model is installed (run: ollama pull llama3.2)"
        
        # Prepare the prompt
        batting_win_prob = model_output.get('batting_win_prob', 0) * 100
        bowling_win_prob = model_output.get('bowling_win_prob', 0) * 100
        
        weather_info = ""
        if weather_data:
            weather_info = f"""
Weather Conditions:
- Temperature: {weather_data.get('temperature', 'N/A')}°C
- Condition: {weather_data.get('condition', 'N/A')}
- Humidity: {weather_data.get('humidity', 'N/A')}%
- Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
"""
        else:
            weather_info = "Weather data: Not available"
        
        player_info = ""
        if player_form:
            if player_form['is_bowler']:
                player_info = f"""
Player Form ({player_form['player_name']}):
- Average Wickets per Match: {player_form['avg_wickets']}
- Average Economy Rate: {player_form['avg_economy']}
- Recent Performance: Last 3 matches analyzed
"""
            else:
                player_info = f"""
Player Form ({player_form['player_name']}):
- Average Runs per Match: {player_form['avg_runs']}
- Average Strike Rate: {player_form['avg_strike_rate']}
- Recent Strike Rate: {player_form['recent_strike_rate']}
- Matches Analyzed: {player_form['total_matches']}
"""
        else:
            player_info = "Player form: Not provided"
        
        prompt = f"""You are an expert cricket analyst. Based on the following match data, provide a detailed analysis and prediction for the IPL match outcome.

ML Model Prediction:
- Batting Team Win Probability: {batting_win_prob:.2f}%
- Bowling Team Win Probability: {bowling_win_prob:.2f}%

{weather_info}

{player_info}

Please provide:
1. A clear match prediction (which team is likely to win)
2. Key factors influencing the outcome
3. How weather conditions might affect the match
4. Impact of the player's current form on the match (if provided)
5. Final recommendation

Keep the response concise but informative (200-300 words)."""

        # Try llama3.2 first, fallback to smaller models if memory issue
        models_to_try = ['llama3.2', 'llama3.2:3b', 'llama3.2:1b']
        last_error = None
        
        for model_name in models_to_try:
            try:
                # Call Ollama
                response = ollama.chat(model=model_name, messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ])
                return response['message']['content']
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                # If it's a memory error, try next smaller model
                if 'memory' in error_msg.lower() or 'system memory' in error_msg.lower():
                    continue
                # If it's a model not found error, try next model
                if 'not found' in error_msg.lower() or 'model' in error_msg.lower():
                    continue
                # For other errors, break and show error
                break
        
        # If we get here, all models failed
        error_msg = last_error or "Unknown error"
        
        # Provide helpful error message based on error type
        if 'memory' in error_msg.lower() or 'system memory' in error_msg.lower():
            return f"""
⚠️ **Insufficient Memory Error**

**Error:** {error_msg}

**Solutions:**

1. **Use a smaller model (Recommended):**
   ```powershell
   ollama pull llama3.2:3b
   ```
   Then the app will automatically use the smaller model.

2. **Or use the smallest model:**
   ```powershell
   ollama pull llama3.2:1b
   ```

3. **Free up system memory:**
   - Close other applications
   - Restart your computer
   - Close browser tabs

4. **Increase Ollama's memory limit** (if possible in settings)

**Note:** The AI analysis feature is optional. You can still use the ML predictions without it.
"""
        elif 'not found' in error_msg.lower() or 'model' in error_msg.lower():
            return f"""
⚠️ **Model Not Found**

**Error:** {error_msg}

**Solution:** Install a model:
```powershell
# Full model (requires ~2.3 GB RAM)
ollama pull llama3.2

# Or smaller models (recommended if low memory):
ollama pull llama3.2:3b    # ~2 GB RAM
ollama pull llama3.2:1b    # ~1 GB RAM
```

**Note:** The AI analysis feature is optional. You can still use the ML predictions without it.
"""
        else:
            return f"""
⚠️ **AI Analysis Error**

**Error:** {error_msg}

**Troubleshooting:**
1. Make sure Ollama is running: `ollama list`
2. Check if a model is installed: `ollama list`
3. If memory issues, try: `ollama pull llama3.2:3b`
4. Restart Ollama service if needed

**Note:** The AI analysis feature is optional. You can still use the ML predictions without it.
"""
    except Exception as e:
        # Catch any unexpected errors
        return f"""
⚠️ **Unexpected Error in AI Analysis**

**Error:** {str(e)}

**Troubleshooting:**
1. Make sure Ollama is installed and running
2. Check if a model is installed: `ollama list`
3. Try installing a smaller model: `ollama pull llama3.2:3b`
4. Restart Ollama service if needed

**Note:** The AI analysis feature is optional. You can still use the ML predictions without it.
"""

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

# Load the pipeline with compatibility handling
try:
    import sys
    import numpy as np
    
    # Check for version incompatibilities
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    numpy_version = np.__version__
    
    # Python 3.14+ and NumPy 2.x are incompatible with old models
    if sys.version_info >= (3, 14) or int(numpy_version.split('.')[0]) >= 2:
        st.error(f"""
        🚨 **CRITICAL: Version Incompatibility Detected!**
        
        **Your Environment:**
        - Python: {python_version}
        - NumPy: {numpy_version}
        
        **The Problem:**
        - The model requires **NumPy 1.x** and **sklearn 1.2.2**
        - Python 3.14+ only has **NumPy 2.x** and **sklearn 1.7.2+** available
        - These versions are **incompatible** with the model
        
        **✅ REQUIRED FIX:**
        You **MUST** use **Python 3.11 or 3.12** for this application!
        
        **Quick Fix:**
        1. Install Python 3.11/3.12 from python.org
        2. Run: `py -3.11 -m pip install numpy<2.0 scikit-learn==1.2.2`
        3. Run: `py -3.11 -m pip install -r requirements_enhanced.txt`
        4. Run: `py -3.11 -m streamlit run application_enhanced.py`
        
        See `URGENT_FIX.md` for detailed instructions!
        """)
        st.stop()
    
    pipe_variable = pickle.load(open('ipl_pred.pkl','rb'))
    # Try to fix any compatibility issues
    if hasattr(pipe_variable, 'steps'):
        # Ensure all steps are properly formatted
        for i, (name, step) in enumerate(pipe_variable.steps):
            if isinstance(step, str):
                st.warning(f"⚠️ Pipeline step '{name}' is a string instead of transformer. This may cause errors.")
except Exception as e:
    error_msg = str(e)
    import sys
    import numpy as np
    
    # Check if it's a numpy/sklearn version issue
    if "numpy.dtype" in error_msg or "binary incompatibility" in error_msg.lower():
        st.error(f"""
        🚨 **NumPy Version Incompatibility!**
        
        **Error:** {error_msg}
        
        **Your Environment:**
        - Python: {sys.version_info.major}.{sys.version_info.minor}
        - NumPy: {np.__version__}
        
        **The Problem:**
        The model was created with **NumPy 1.x**, but you have **NumPy {np.__version__}**.
        Python 3.14+ only has NumPy 2.x available, which is incompatible.
        
        **✅ REQUIRED FIX:**
        Use **Python 3.11 or 3.12** which supports NumPy 1.x!
        
        See `URGENT_FIX.md` for step-by-step instructions.
        """)
    else:
        st.error(f"Error loading model: {error_msg}")
        st.info("💡 This might be a version compatibility issue. Try using Python 3.11/3.12.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(team_list),
                               help='Choose the team that is currently batting')

with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(team_list),
                               help='Choose the team that is currently bowling')

selected_city = st.selectbox('Select the host city',sorted(city_list),
                            help='Select the city where the match is being played')

st.markdown("---")
st.markdown("<div style='background-color: rgba(0, 0, 0, 0.6); padding: 15px; border-radius: 8px; border-left: 4px solid #E31837; margin: 20px 0;'>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #FFD700; margin: 0;'>📊 Match Details</h4>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

target = st.number_input('Target Runs', 
                        min_value=0, 
                        max_value=300, 
                        value=150, 
                        step=1,
                        help='Total runs the batting team needs to score')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', 
                           min_value=0, 
                           max_value=300, 
                           value=0, 
                           step=1,
                           help='Runs scored by batting team so far')

with col4:
    overs = st.number_input('Overs Completed', 
                           min_value=0.0, 
                           max_value=20.0, 
                           value=0.0, 
                           step=0.1,
                           format="%.1f",
                           help='Number of overs completed (e.g., 10.5 for 10.5 overs)')
    
with col5:
    wickets = st.number_input('Wickets Out', 
                             min_value=0, 
                             max_value=10, 
                             value=0, 
                             step=1,
                             help='Number of wickets lost by batting team')

# New feature: Player name input
st.markdown("---")
st.markdown("<h3 style='color: #FFD700; text-align: center;'>Player Form Analysis (Optional)</h3>", unsafe_allow_html=True)
player_name = st.text_input('Enter Player Name (for form analysis)', 
                           placeholder='e.g., Virat Kohli, MS Dhoni, Rohit Sharma, KL Rahul, Jasprit Bumrah',
                           help='Enter any player name - fuzzy matching will handle typos and case variations',
                           key='player_input')
st.caption(f"💡 **Available players:** {', '.join(list(SAMPLE_PLAYERS.keys()))} | Fuzzy matching enabled for typos")

if st.button('Predict Probability', use_container_width=True):
    # Input validation
    if batting_team == bowling_team:
        st.error("⚠️ Batting team and bowling team cannot be the same!")
        st.stop()
    
    if score > target:
        st.warning("⚠️ Current score cannot be greater than target!")
        st.stop()
    
    if overs > 20:
        st.warning("⚠️ Overs cannot exceed 20 in T20 cricket!")
        st.stop()
    
    if wickets > 10:
        st.warning("⚠️ Wickets cannot exceed 10!")
        st.stop()
    
    # Ensure values are native Python types (not numpy/pandas types)
    target = float(target) if hasattr(target, '__float__') else target
    score = float(score) if hasattr(score, '__float__') else score
    overs = float(overs) if hasattr(overs, '__float__') else overs
    wickets = float(wickets) if hasattr(wickets, '__float__') else wickets
    
    # Calculate match metrics (EXACTLY matching original application.py)
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left
    
    # Create DataFrame EXACTLY as in original application.py
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
    
    try:
        # Debug: Check DataFrame structure
        if st.session_state.get('debug_mode', False):
            st.write("**Debug Info:**")
            st.write(f"DataFrame columns: {list(input_df.columns)}")
            st.write(f"DataFrame dtypes:\n{input_df.dtypes}")
            st.write(f"DataFrame values:\n{input_df}")
            st.write(f"Pipeline steps: {[step[0] for step in pipe_variable.steps] if hasattr(pipe_variable, 'steps') else 'N/A'}")
        
        with st.spinner('🔄 Analyzing match data and generating prediction...'):
            result = pipe_variable.predict_proba(input_df)
    except Exception as e:
        import traceback
        import sklearn
        error_details = traceback.format_exc()
        error_msg = str(e)
        
        st.error(f"❌ **Error making prediction:** {error_msg}")
        
        # Check if it's a sklearn version issue
        if "'str' object has no attribute 'transform'" in error_msg or "transform" in error_msg.lower():
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            
            st.error("""
            🚨 **CRITICAL: Python Version Incompatibility!**
            
            **The Problem:**
            - You're using **Python """ + python_version + """**
            - The model requires **sklearn 1.2.2**
            - **sklearn 1.2.2 is NOT available for Python 3.14+**
            
            **✅ THE FIX (Required):**
            You MUST use **Python 3.11 or 3.12** for this application!
            
            1. Download Python 3.11/3.12 from python.org
            2. Install it (check "Add Python to PATH")
            3. Run: `py -3.11 -m pip install scikit-learn==1.2.2`
            4. Run: `py -3.11 -m pip install -r requirements_enhanced.txt`
            5. Run: `py -3.11 -m streamlit run application_enhanced.py`
            
            **See `URGENT_FIX.md` for step-by-step instructions!**
            """)
        
        # Show detailed error in expander
        with st.expander("🔍 View Detailed Error Information"):
            st.code(error_details)
            st.write("**DataFrame Info:**")
            st.write(input_df)
            st.write(f"**DataFrame dtypes:**\n{input_df.dtypes}")
            st.write(f"**Sklearn version:** {sklearn.__version__}")
        
        st.info("💡 **General Troubleshooting:**\n- Check that all inputs are valid numbers\n- Ensure overs are between 0-20\n- Verify wickets are between 0-10\n- Make sure current score ≤ target")
        if st.button('🔄 Try Again'):
            st.rerun()
        st.stop()
    
    loss = result[0][0]
    win = result[0][1]
    
    # Store results in session state
    st.session_state['model_result'] = {
        'batting_win_prob': win,
        'bowling_win_prob': loss,
        'batting_team': batting_team,
        'bowling_team': bowling_team
    }
    
    st.markdown("---")
    st.markdown("<h3 style='color: #FFD700; text-align: center;'>🎯 Prediction Results</h3>", unsafe_allow_html=True)
    
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
    
    # Show match summary
    st.markdown(f"""
    <div style='background-color: rgba(0, 0, 0, 0.7); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4CAF50;'>
        <p style='color: #E0E0E0; margin: 5px 0;'><strong>Match Summary:</strong></p>
        <p style='color: #E0E0E0; margin: 5px 0;'>📊 Score: {score}/{wickets} in {overs:.1f} overs | Target: {target}</p>
        <p style='color: #E0E0E0; margin: 5px 0;'>🏏 Runs needed: {runs_left} | Balls remaining: {balls_left}</p>
        <p style='color: #E0E0E0; margin: 5px 0;'>⚡ Current RR: {crr:.2f} | Required RR: {rrr:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get weather data
    with st.spinner('Fetching weather data...'):
        weather_data = get_weather(selected_city)
        if weather_data:
            st.session_state['weather_data'] = weather_data
            st.markdown(f"""
            <div style='background-color: rgba(0, 0, 0, 0.85); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196F3; box-shadow: 0 0 15px rgba(0, 0, 0, 0.6);'>
                <p style='color: #E0E0E0; margin: 5px 0; font-size: 18px; font-weight: bold;'>🌤️ Weather: <span style='color: #4CAF50;'>{weather_data['temperature']}°C, {weather_data['condition']}</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.session_state['weather_data'] = None
            st.markdown("""
            <div style='background-color: rgba(0, 0, 0, 0.85); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #FF9800; box-shadow: 0 0 15px rgba(0, 0, 0, 0.6);'>
                <p style='color: #E0E0E0; margin: 5px 0; font-size: 18px;'>⚠️ Weather data unavailable</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Get player form
    player_form = None
    if player_name:
        with st.spinner('Analyzing player form...'):
            player_form = calculate_player_form(player_name)
            if player_form:
                st.session_state['player_form'] = player_form
                if player_form['is_bowler']:
                    st.markdown(f"""
                    <div style='background-color: rgba(0, 0, 0, 0.85); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196F3; box-shadow: 0 0 15px rgba(0, 0, 0, 0.6);'>
                        <p style='color: #E0E0E0; margin: 5px 0; font-size: 18px; font-weight: bold;'>👤 Player: <span style='color: #4CAF50;'>{player_form['player_name']}</span> | Avg Wickets: <span style='color: #FFD700;'>{player_form['avg_wickets']}</span> | Economy: <span style='color: #FFD700;'>{player_form['avg_economy']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: rgba(0, 0, 0, 0.85); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196F3; box-shadow: 0 0 15px rgba(0, 0, 0, 0.6);'>
                        <p style='color: #E0E0E0; margin: 5px 0; font-size: 18px; font-weight: bold;'>👤 Player: <span style='color: #4CAF50;'>{player_form['player_name']}</span> | Avg Runs: <span style='color: #FFD700;'>{player_form['avg_runs']}</span> | Strike Rate: <span style='color: #FFD700;'>{player_form['avg_strike_rate']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: rgba(0, 0, 0, 0.85); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #FF9800; box-shadow: 0 0 15px rgba(0, 0, 0, 0.6);'>
                    <p style='color: #E0E0E0; margin: 5px 0; font-size: 18px;'>⚠️ Player '{player_name}' not found. Available players: {', '.join(list(SAMPLE_PLAYERS.keys())[:5])}...</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state['player_form'] = None
    else:
        st.session_state['player_form'] = None
    
    # Generate AI decision
    if 'model_result' in st.session_state:
        with st.spinner('Generating AI-powered match analysis...'):
            ai_decision = generate_ai_decision(
                st.session_state['model_result'],
                st.session_state.get('weather_data'),
                st.session_state.get('player_form')
            )
            st.session_state['ai_decision'] = ai_decision
            st.session_state['backend_details'] = {
                'model_output': st.session_state['model_result'],
                'weather': st.session_state.get('weather_data'),
                'player_form': st.session_state.get('player_form'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

# Display AI Decision Box
if 'ai_decision' in st.session_state:
    ai_decision = st.session_state['ai_decision']
    
    # Check if it's an error message
    is_error = any(keyword in ai_decision.lower() for keyword in ['error', '⚠️', 'insufficient', 'not found', 'troubleshooting'])
    
    st.markdown("---")
    if is_error:
        st.markdown("<h3 style='color: #FF9800; text-align: center;'>⚠️ AI Analysis Unavailable</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background-color: rgba(255, 152, 0, 0.1); padding: 20px; border-radius: 10px; margin: 20px 0px; border: 2px solid #FF9800;'>
            <div style='color: #E0E0E0; font-size: 16px; line-height: 1.6;'>
                {ai_decision.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #4CAF50; text-align: center;'>🤖 AI-Powered Match Analysis</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='ai-decision-box'>
            {ai_decision}
        </div>
        """, unsafe_allow_html=True)
    
    # Checkbox for backend details
    show_details = st.checkbox('Show Detailed Backend Information', value=False)
    
    if show_details and 'backend_details' in st.session_state:
        details = st.session_state['backend_details']
        
        backend_info = f"""
=== BACKEND DETAILS ===

Timestamp: {details['timestamp']}

1. ML MODEL OUTPUT:
{json.dumps(details['model_output'], indent=2)}

2. WEATHER DATA:
{json.dumps(details['weather'], indent=2) if details['weather'] else 'Not Available'}

3. PLAYER FORM ANALYSIS:
{json.dumps(details['player_form'], indent=2) if details['player_form'] else 'Not Provided'}

4. AI MODEL:
- Model: llama3.2 (via Ollama)
- Input: Combined analysis of ML prediction, weather, and player form
- Output: Natural language match prediction and analysis
"""
        
        st.markdown(f"""
        <div class='backend-details'>
            <pre>{backend_info}</pre>
        </div>
        """, unsafe_allow_html=True)

