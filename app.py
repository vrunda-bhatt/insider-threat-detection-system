import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import joblib
import io
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="ThreatTrack - Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- Theme & Styling ---
# Using Streamlit's built-in themes and adding custom CSS for fonts and colors
# (Roboto for content, Times New Roman for headers)
st.markdown("""
<style>
    /* Main app styling - clean light background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Headers - Times New Roman, bold */
    h1, h2, h3 {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        color: #1e293b;
    }

    /* Content text - Roboto */
    body, .stMarkdown, .stDataFrame, .stMetric, .stButton, .stSelectbox {
        font-family: 'Roboto', sans-serif;
        color: #334155;
    }

    /* Custom title - professional with accent color */
    .title-text {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 3.5rem !important;
        font-weight: bold !important;
        color: #0f172a !important;
        padding-top: 1rem;
        line-height: 1.2;
        display: block;
        border-left: 8px solid #3b82f6;
        padding-left: 20px;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        font-family: 'Roboto', sans-serif;
        font-size: 1.2rem;
        color: #64748b;
        margin-top: -10px;
        margin-left: 28px;
        font-weight: 400;
    }

    /* Sidebar styling - professional with accent */
    .stSidebar {
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
        border-right: 3px solid #60a5fa;
    }
    
    /* Sidebar text styling for better visibility - exclude file uploader */
    .stSidebar [data-testid="stMarkdownContainer"] p,
    .stSidebar [data-testid="stMarkdownContainer"] strong,
    .stSidebar [data-testid="stMarkdownContainer"] li,
    .stSidebar .stSelectbox label,
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    section[data-testid="stSidebar"] label:not([data-testid="stFileUploader"] label),
    section[data-testid="stSidebar"] p:not([data-testid="stFileUploader"] p),
    section[data-testid="stSidebar"] span:not([data-testid="stFileUploader"] span) {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Sidebar info box styling */
    .stSidebar .stAlert {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    .stSidebar .stAlert p {
        color: #ffffff !important;
    }
    
    /* Sidebar sliders and inputs */
    .stSidebar .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Override slider thumb (dot) to white */
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {
        background-color: rgba(255, 255, 255, 0.5) !important;
    }
    
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] div {
        background-color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Slider value numbers - make white */
    section[data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stSlider .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Help icon (?) - make white */
    section[data-testid="stSidebar"] .stTooltipIcon svg,
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
    section[data-testid="stSidebar"] button[kind="helpTooltip"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] button[kind="helpTooltip"] {
        color: #ffffff !important;
    }
    
    /* File uploader main label only - make white */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] > label:first-child {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* File uploader button styling - gray text */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        color: #6b7280 !important;
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* File uploader dropzone text - keep dark gray */
    section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small {
        color: #6b7280 !important;
    }
    
    /* Colorful accent bars for sections */
    .stMarkdown h1::before,
    .stMarkdown h2::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #3b82f6, #8b5cf6);
        margin-right: 10px;
        vertical-align: middle;
    }
    
    /* Metric styling - colorful cards */
    div[data-testid="stMetric"] {
        background: white;
        border-left: 5px solid #3b82f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:nth-child(2) {
        border-left-color: #10b981;
    }
    
    div[data-testid="stMetric"]:nth-child(3) {
        border-left-color: #f59e0b;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }

    /* Status-specific metric colors and styling */
    div[data-testid="stMetricLabel"]:has(div:contains("System Status")) + div p {
        font-size: 1.8rem;
        font-weight: bold;
    }

    /* Third metric card (status) - enhanced styling */
    div[data-testid="stMetric"]:nth-child(3) {
        border-left-width: 6px !important;
    }
    
    /* Status text colors */
    div[data-testid="stMetricLabel"]:has(div:contains("System Status")) + div p:contains("Safe") {
        color: #10b981 !important;
    }
    div[data-testid="stMetricLabel"]:has(div:contains("System Status")) + div p:contains("Warning") {
        color: #f59e0b !important;
    }
    div[data-testid="stMetricLabel"]:has(div:contains("System Status")) + div p:contains("Threat") {
        color: #ef4444 !important;
    }
    
    /* Tab styling - clean style with proper colors */
    button[data-baseweb="tab"] {
        font-family: 'Roboto', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: #64748b !important;  /* Gray for inactive tabs */
    }
    
    /* Active tab styling */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1e293b !important;  /* Dark color for active tab */
        font-weight: 600 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
    
    /* Tab hover effect */
    button[data-baseweb="tab"]:hover {
        color: #1e293b !important;
    }
    
    /* Button styling - professional blue */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Container with colored borders */
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    
    /* Info/Warning/Error boxes with colors */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Data tables styling */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Add subtle color accents to sections */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        padding: 1rem;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

# --- 1. Mock/Load Functions ---

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_and_scaler():
    """
    Tries to load the trained model and scaler.
    If they fail, creates mock objects for demo purposes.
    """
    model_path = 'lstm_autoencoder.keras'
    scaler_path = 'scaler.joblib'
    model = None
    scaler = None
    
    try:
        model = load_model(model_path)
        st.success(f"Loaded model from `{model_path}`")
    except Exception as e:
        st.warning(f"Could not load model from `{model_path}`. Creating a mock model. Error: {e}")
        # Create a mock model structure
        n_features = 4  # ev_device_Connect, ev_device_Disconnect, ev_logon, hour
        timesteps = 10
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=False),
            RepeatVector(timesteps),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        model.compile(optimizer='adam', loss='mse')

    try:
        scaler = joblib.load(scaler_path)
        st.success(f"Loaded scaler from `{scaler_path}`")
    except Exception as e:
        st.warning(f"Could not load scaler from `{scaler_path}`. Creating a mock scaler. Error: {e}")
        # Create a mock scaler
        scaler = MinMaxScaler()
        # Fit on dummy data just to initialize it
        dummy_data = np.random.rand(10, 4)
        scaler.fit(dummy_data)

    return model, scaler

def load_training_plots():
    """
    Checks if training plots exist and returns paths or placeholder figures.
    """
    import os
    
    loss_exists = os.path.exists('training_loss.png')
    roc_exists = os.path.exists('roc_curves.png')
    
    fig_loss = None
    fig_roc = None
    
    if not loss_exists:
        fig_loss = go.Figure().update_layout(
            title="Reconstruction Error Curve (training_loss.png not found)",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white"
        )
        fig_loss.add_annotation(text="Run train_model.py to generate this plot",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False)

    if not roc_exists:
        fig_roc = go.Figure().update_layout(
            title="ROC-AUC Score (roc_curves.png not found)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white"
        )
        fig_roc.add_annotation(text="Run train_model.py to generate this plot",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
    
    return loss_exists, roc_exists, fig_loss, fig_roc

@st.cache_data
def load_data(uploaded_file):
    """
    Loads data from a file uploader or creates mock data.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded CSV.")
            return df, None # No ground truth for uploaded files
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return pd.DataFrame(), None
    else:
        st.info("No file uploaded. Using 'synthetic_insider_threat.csv' test dataset.")
        # Create a mock DataFrame that mimics 'synthetic_insider_threat.csv'
        data = {
            'Timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=1000, freq='H')),
            'User_ID': np.random.choice(['User_A', 'User_B', 'User_C', 'User_D', 'User_E'], 1000),
            'Action': np.random.choice(['login', 'file_access', 'email_sent', 'failed_login'], 1000, p=[0.4, 0.3, 0.2, 0.1]),
            'Suspicious': np.zeros(1000)
        }
        df = pd.DataFrame(data)
        
        # Inject some anomalies
        for i in range(20):
            idx = np.random.randint(100, 900)
            df.loc[idx, 'Action'] = 'failed_login'
            df.loc[idx, 'Suspicious'] = 1
            df.loc[idx, 'User_ID'] = 'User_E' # Make one user suspicious
        
        # Map for ground truth
        ground_truth_map = df[['User_ID', 'Timestamp', 'Suspicious']].copy()
        
        return df, ground_truth_map

# --- 2. Analysis Functions (from test_model.py) ---

def detect_anomaly_type(timestamps, feature_errors, feature_names):
    """
    Detect the type of anomaly based on temporal patterns and feature contributions.
    """
    anomaly_types = []
    
    # Handle single timestamp case
    if not isinstance(timestamps, (list, np.ndarray)):
        timestamps = [timestamps]
    
    # Convert timestamps to pandas Timestamp for easier analysis
    ts_list = []
    for ts in timestamps:
        if isinstance(ts, str):
            ts_list.append(pd.Timestamp(ts))
        elif hasattr(ts, 'hour'):  # Already a datetime-like object
            ts_list.append(pd.Timestamp(ts))
        else:
            ts_list.append(pd.Timestamp(ts))
    
    # Check temporal patterns
    hours = [ts.hour for ts in ts_list]
    weekdays = [ts.weekday() for ts in ts_list]
    
    # Lower thresholds for better detection
    min_events = max(1, len(ts_list) // 3)  # At least 1/3 of events, minimum 1
    
    # Night activity (22:00-06:00)
    night_events = sum(1 for h in hours if h < 6 or h > 22)
    if night_events >= min_events:
        anomaly_types.append('night_activity')
    
    # Early morning activity (04:00-07:00)
    early_events = sum(1 for h in hours if 4 <= h <= 7)
    if early_events >= min_events:
        anomaly_types.append('early_morning')
    
    # Weekend activity
    weekend_events = sum(1 for w in weekdays if w >= 5)
    if weekend_events >= min_events:
        anomaly_types.append('weekend_work')
    
    # Off-hours activity (before 9 AM or after 6 PM)
    off_hours_events = sum(1 for h in hours if h < 9 or h > 18)
    if off_hours_events >= min_events and 'night_activity' not in anomaly_types and 'early_morning' not in anomaly_types:
        anomaly_types.append('off_hours_activity')
    
    # Burst activity (high feature values) - more sensitive detection
    if len(feature_errors) > 0:
        max_feature_error = np.max(feature_errors)
        avg_feature_error = np.mean(feature_errors)
        if max_feature_error > avg_feature_error * 2:  # Lower threshold
            anomaly_types.append('burst_activity')
        
        # Excessive activity (consistently high across features)
        high_activity_features = sum(1 for err in feature_errors if err > avg_feature_error)
        if high_activity_features >= len(feature_errors) * 0.6:  # Lower threshold
            anomaly_types.append('excessive_activity')
    
    # Check for unusual timing patterns
    if len(set(hours)) == 1:  # All events at the same hour
        anomaly_types.append('concentrated_timing')
    
    return anomaly_types if anomaly_types else ['unknown']

def classify_anomaly_type(timestamp, hour, weekday, total_activity, avg_activity):
    """Classify the type of anomaly based on temporal and behavioral patterns."""
    anomaly_types = []
    
    # Check for temporal anomalies
    if 0 <= hour < 5:
        anomaly_types.append("Late Night Activity")
    elif 5 <= hour < 7:
        anomaly_types.append("Early Morning Activity")
        
    # Check for weekend activity
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        anomaly_types.append("Weekend Activity")
        
    # Check for burst/excessive activity
    if avg_activity > 0:
        activity_ratio = total_activity / avg_activity
        if activity_ratio > 10:
            anomaly_types.append("Burst Activity")
        elif activity_ratio > 5:
            anomaly_types.append("Excessive Activity")
    
    return anomaly_types if anomaly_types else ["Unusual Pattern"]

def events_to_feature_table(events_df):
    """Converts event DataFrame to feature table."""
    # Map the columns to our expected format
    column_mapping = {
        'User_ID': 'user',
        'Action': 'event_type',
        'Timestamp': 'timestamp'
    }
    df = events_df.rename(columns=column_mapping)
    
    if 'timestamp' not in df.columns:
        st.error("Dataset must contain a 'Timestamp' column.")
        return pd.DataFrame()
    if 'user' not in df.columns:
        st.error("Dataset must contain a 'User_ID' or 'user' column.")
        return pd.DataFrame()
    if 'event_type' not in df.columns:
        st.error("Dataset must contain an 'Action' or 'event_type' column.")
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Event type mapping to match training data format
    event_type_mapping = {
        'login': 'logon',
        'failed_login': 'logon',
        'file_access': 'device_Connect',
        'email_sent': 'device_Connect'
    }
    df['event_type'] = df['event_type'].map(lambda x: event_type_mapping.get(x, 'other'))
    
    # Define expected columns for one-hot encoding (matching training)
    expected_columns = ['ev_device_Connect', 'ev_device_Disconnect', 'ev_logon']
    
    dummies = pd.get_dummies(df['event_type'], prefix='ev')
    
    for col in expected_columns:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[expected_columns]
    
    df = pd.concat([df[['timestamp', 'user']].reset_index(drop=True), 
                   dummies.reset_index(drop=True)], axis=1)
    
    agg = df.groupby(['user', 'timestamp']).sum().reset_index()
    agg = agg.rename(columns={'user': 'user_id'})
    agg['hour'] = agg['timestamp'].dt.hour
    agg = agg.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    return agg

def create_user_sequences(agg_df, seq_length, scaler):
    """Create sequences for LSTM input."""
    feature_cols = ['ev_device_Connect', 'ev_device_Disconnect', 'ev_logon', 'hour']
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in agg_df.columns:
            st.error(f"Missing expected feature column '{col}' after processing.")
            return np.array([]), [], [], []

    # Scale features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        agg_df[feature_cols] = scaler.transform(agg_df[feature_cols])
    
    sequences = []
    users = []
    timestamps = []
    
    for uid, grp in agg_df.groupby('user_id'):
        grp_sorted = grp.sort_values('timestamp')
        values = grp_sorted[feature_cols].values
        times = grp_sorted['timestamp'].values
        
        if len(values) < seq_length:
            continue
        for i in range(len(values) - seq_length + 1):
            sequences.append(values[i:i+seq_length])
            users.append(uid)
            timestamps.append(times[i]) # Store start time of sequence
    
    if not sequences:
        st.warning('No sequences generated. Data may be too sparse or seq_length too high.')
        return np.array([]), [], [], []
    
    return np.array(sequences), users, timestamps, feature_cols

def analyze_anomalies(mse, anomalies, users, timestamps, X_original, X_reconstructed, feature_names):
    """Analyze and report detailed anomaly statistics with anomaly types."""
    users = np.array(users)
    
    # Get all anomaly explanations first
    anomaly_indices = np.where(anomalies)[0]
    all_explanations = []
    
    for idx in anomaly_indices:
        seq_original = X_original[idx]
        seq_reconstructed = X_reconstructed[idx]
        
        # Handle timestamps properly - get the actual timestamp for this sequence
        if isinstance(timestamps[idx], (list, np.ndarray)):
            seq_timestamps = timestamps[idx]
        else:
            seq_timestamps = [timestamps[idx]]
        
        # Per-feature errors across the sequence
        feature_errors = np.mean(np.power(seq_original - seq_reconstructed, 2), axis=0)
        
        # Detect anomaly types using the new function
        anomaly_types = detect_anomaly_type(seq_timestamps, feature_errors, feature_names)
        
        # Fallback: Use simple temporal analysis if main detection fails
        if len(anomaly_types) == 1 and anomaly_types[0] == 'unknown':
            # Simple temporal classification based on first timestamp
            main_ts = pd.Timestamp(seq_timestamps[0]) if seq_timestamps else pd.Timestamp('now')
            hour = main_ts.hour
            weekday = main_ts.weekday()
            
            fallback_types = []
            if hour < 6 or hour > 22:
                fallback_types.append('night_activity')
            elif 4 <= hour <= 7:
                fallback_types.append('early_morning')
            elif hour < 9 or hour > 18:
                fallback_types.append('off_hours_activity')
            
            if weekday >= 5:
                fallback_types.append('weekend_work')
                
            if fallback_types:
                anomaly_types = fallback_types
        
        top_feature_indices = np.argsort(feature_errors)[::-1]
        top_features = []
        for i in range(min(3, len(top_feature_indices))):
            feat_idx = top_feature_indices[i]
            top_features.append({
                'feature': feature_names[feat_idx],
                'contribution': feature_errors[feat_idx],
                'actual_mean': np.mean(seq_original[:, feat_idx]),
                'expected_mean': np.mean(seq_reconstructed[:, feat_idx])
            })
        
        all_explanations.append({
            'user': users[idx],
            'timestamp': timestamps[idx],
            'anomaly_score': mse[idx],
            'anomaly_types': anomaly_types,
            'top_features': top_features
        })
    
    # Calculate statistics per user with anomaly types
    user_stats = []
    for user in np.unique(users):
        user_mask = (users == user)
        user_anomalies = anomalies[user_mask]
        
        # Get anomaly types for this user
        user_anomaly_types = set()
        for exp in all_explanations:
            if exp['user'] == user:
                user_anomaly_types.update(exp['anomaly_types'])
        
        anomaly_types_str = ', '.join(sorted(user_anomaly_types)) if user_anomaly_types else 'none'
        
        if sum(user_mask) > 0:
            user_stats.append({
                'User': user,
                'Total Logs': sum(user_mask),
                'Anomalies': sum(user_anomalies),
                'Anomaly Rate (%)': (sum(user_anomalies) / sum(user_mask)) * 100,
                'Anomaly Types': anomaly_types_str,
                'Max Error': np.max(mse[user_mask])
            })
    
    user_df = pd.DataFrame(user_stats).sort_values('Anomalies', ascending=False)
    
    # Format explanations for display (convert to the old format for compatibility)
    explanations = []
    for exp in all_explanations[:5]:  # Top 5
        explanations.append({
            'User': exp['user'],
            'Timestamp': exp['timestamp'],
            'Anomaly Score': exp['anomaly_score'],
            'Anomaly Type': ', '.join(exp['anomaly_types']),
            'Top Features': exp['top_features']
        })
        
    return user_df, explanations

# --- 3. Plotting Functions (Plotly) ---

def plot_error_dist_hist(mse, threshold):
    """Plot reconstruction error distribution (Histogram)."""
    fig = px.histogram(mse, 
                       log_y=True, 
                       title="Reconstruction Error Distribution (Log Scale)",
                       labels={'value': 'Reconstruction Error'},
                       template="plotly_white")
    
    fig.add_shape(type='line',
                  x0=threshold, y0=0, x1=threshold, y1=len(mse),
                  line=dict(color='red', width=2, dash='dash'))
    
    fig.add_annotation(x=threshold, y=np.log10(len(mse)) if len(mse) > 0 else 0,
                       text=f"Threshold: {threshold:.4f}",
                       showarrow=True, arrowhead=1, ax=50, ay=-40,
                       bordercolor="#c7c7c7", borderwidth=1,
                       bgcolor="white", opacity=0.8)
    
    fig.update_layout(
        font_family='Roboto',
        title_font_family='Times New Roman'
    )
    return fig

def plot_error_scatter(mse, anomalies, threshold):
    """Plot reconstruction error over time (Scatter Plot)."""
    df = pd.DataFrame({'error': mse, 'index': np.arange(len(mse))})
    df['status'] = np.where(anomalies, 'Anomaly', 'Normal')
    
    color_map = {'Normal': 'rgba(99, 110, 250, 0.5)', 'Anomaly': 'rgba(239, 85, 59, 0.8)'}
    
    fig = px.scatter(df, x='index', y='error', color='status',
                     log_y=True,
                     title="Reconstruction Error (Scatter Plot)",
                     labels={'index': 'Sequence Index', 'error': 'Reconstruction Error (Log Scale)'},
                     color_discrete_map=color_map,
                     template="plotly_white",
                     hover_data={'index': True, 'error': ':.4f'})
    
    fig.add_hline(y=threshold, line=dict(color='red', width=2, dash='dash'),
                  annotation_text=f"Threshold: {threshold:.4f}",
                  annotation_position="bottom right")
    
    fig.update_layout(
        font_family='Roboto',
        title_font_family='Times New Roman'
    )
    return fig

def plot_user_barchart(user_stats_df):
    """Plot user anomaly rates (Bar Chart)."""
    df = user_stats_df.sort_values('Anomaly Rate (%)', ascending=False).head(20)
    
    # Assign colors based on anomaly rate
    def assign_color(rate):
        if rate > 20: return '#dc3545' # Red
        if rate > 5: return '#ffc107' # Orange
        return '#28a745' # Green

    df['color'] = df['Anomaly Rate (%)'].apply(assign_color)
    
    fig = px.bar(df, x='User', y='Anomaly Rate (%)',
                 title="User Anomaly Rate (Top 20)",
                 color='color',
                 color_discrete_map='identity', # Use the colors from the 'color' column
                 template="plotly_white",
                 hover_data={'User': True, 'Anomalies': True, 'Total Logs': True, 'Anomaly Rate (%)': ':.2f'})
    
    fig.update_layout(
        xaxis_title="User ID",
        yaxis_title="Anomaly Rate (%)",
        font_family='Roboto',
        title_font_family='Times New Roman'
    )
    return fig

def plot_feature_contributions(explanations):
    """Plot feature contributions to anomalies."""
    contributions = {}
    for exp in explanations:
        for feat in exp['Top Features']:
            name = feat['feature']
            if name not in contributions:
                contributions[name] = []
            contributions[name].append(feat['contribution'])
    
    if not contributions:
        return go.Figure().update_layout(title="Feature Contributions (No anomalies to analyze)")
        
    avg_contributions = {k: np.mean(v) for k, v in contributions.items()}
    df = pd.DataFrame.from_dict(avg_contributions, orient='index', columns=['Average Contribution'])
    df = df.reset_index().rename(columns={'index': 'Feature'}).sort_values('Average Contribution', ascending=False)
    
    fig = px.bar(df, x='Feature', y='Average Contribution',
                 title="Average Feature Contribution to Top Anomalies",
                 template="plotly_white")
    
    fig.update_layout(
        font_family='Roboto',
        title_font_family='Times New Roman'
    )
    return fig

# --- Main Application ---

# --- Title ---
st.markdown('<p class="title-text">ThreatTrack — Detect. Trace. Analyze.</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">An LSTM Autoencoder for Insider Threat Detection</p>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ("Use Test Dataset", "Upload a CSV file")
)

uploaded_file = None
if data_source == "Upload a CSV file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your log data (CSV)",
        type=["csv"],
        help="CSV must contain 'Timestamp', 'User_ID', and 'Action' columns."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
SEQ_LEN = st.sidebar.slider("Sequence Length", min_value=5, max_value=20, value=10, help="How many events to group into one sequence.")
THRESHOLD_PCT = st.sidebar.slider("Anomaly Threshold (Percentile)", min_value=80, max_value=99, value=95, help="The percentile of reconstruction error to use as the anomaly threshold.")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **About:** This dashboard uses an LSTM Autoencoder
    to detect anomalous sequences in user log data.
    
    **Status:**
    - `Safe`: No/low anomalies.
    - `Warning`: Moderate anomalies detected.
    - `Threat`: High anomaly rate detected.
    """
)

# --- Main Dashboard ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.results = {}

if st.button("Run Analysis", type="primary"):
    # Load model and data
    model, scaler = load_model_and_scaler()
    raw_df, ground_truth = load_data(uploaded_file)
    
    if raw_df.empty:
        st.error("Data could not be loaded. Please upload a valid CSV or select the test dataset.")
    else:
        with st.spinner("Analyzing logs... This may take a moment."):
            # 1. Process Data
            agg_df = events_to_feature_table(raw_df)
            
            if not agg_df.empty:
                # 2. Create Sequences
                X, users, timestamps, feature_cols = create_user_sequences(agg_df, SEQ_LEN, scaler)
                
                if X.shape[0] > 0:
                    # 3. Get Predictions
                    X_pred = model.predict(X, verbose=0)
                    mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))
                    threshold = np.percentile(mse, THRESHOLD_PCT)
                    anomalies = mse > threshold
                    
                    # 4. Analyze Anomalies
                    user_stats, explanations = analyze_anomalies(
                        mse, anomalies, users, timestamps, X, X_pred, feature_cols
                    )
                    
                    # 5. Store results in session state
                    st.session_state.results = {
                        "total_logs": len(agg_df),
                        "total_sequences": len(X),
                        "anomalies_detected": int(np.sum(anomalies)),
                        "threshold": threshold,
                        "mse": mse,
                        "anomalies": anomalies,
                        "user_stats": user_stats,
                        "explanations": explanations
                    }
                    st.session_state.analysis_done = True
                    st.success("Analysis complete!")
                else:
                    st.warning("No valid sequences were created from the data.")
            else:
                st.warning("Data could not be processed into a feature table.")

# --- Display Results ---
if st.session_state.analysis_done:
    res = st.session_state.results
    
    # Calculate overview metrics
    total_logs = res['total_logs']
    anomalies_detected = res['anomalies_detected']
    anomaly_rate = (anomalies_detected / res['total_sequences']) * 100 if res['total_sequences'] > 0 else 0
    
    # Determine system status with colors and emojis
    if anomaly_rate == 0:
        status = "🟢 Safe"
        status_color = "#10b981"
        status_bg = "#d1fae5"
    elif anomaly_rate < 5:
        status = "🟠 Warning"
        status_color = "#f5970b"
        status_bg = "#fef3c7"
    else:
        status = "🔴 Threat"
        status_color = "#ef4444"
        status_bg = "#fee2e2"

    # --- Overview Panel ---
    st.header("Overview Panel")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Logs Analyzed", f"{total_logs:,}")
    col2.metric("Anomalies Detected", f"{anomalies_detected:,}")
    
    # Custom status display that matches metric style with hover effect
    with col3:
        st.markdown(f"""
        <style>
        .status-card {{
            background: linear-gradient(135deg, #ffffff 0%, {status_bg} 100%);
            border-left: 6px solid {status_color};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            cursor: pointer;
        }}
        .status-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        }}
        </style>
        <div class="status-card">
            <div style="
                color: #64748b;
                font-size: 0.875rem;
                font-weight: 600;
                margin-bottom: 8px;
                font-family: 'Roboto', sans-serif;
            ">
                Current System Status
            </div>
            <div style="
                color: {status_color};
                font-size: 1.8rem;
                font-weight: bold;
                font-family: 'Roboto', sans-serif;
            ">
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- User Search Feature ---
    st.header("🔍 Individual User Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Get list of all users for dropdown
        all_users = sorted(res['user_stats']['User'].unique())
        search_user = st.selectbox(
            "Select a user to analyze:",
            options=[''] + all_users,
            index=0,
            help="Choose a specific user to see their detailed anomaly profile"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🔍 Analyze User", type="secondary", disabled=not search_user)
    
    if search_user and search_button:
        # Get user data
        user_data = res['user_stats'][res['user_stats']['User'] == search_user]
        
        if len(user_data) > 0:
            user_row = user_data.iloc[0]
            
            # User-specific explanations
            user_explanations = [exp for exp in res['explanations'] if exp['User'] == search_user]
            
            with st.container(border=True):
                st.subheader(f"👤 Detailed Analysis: {search_user}")
                
                # User metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Sequences", f"{user_row['Total Logs']}")
                col2.metric("Anomalies Found", f"{user_row['Anomalies']}")
                col3.metric("Anomaly Rate", f"{user_row['Anomaly Rate (%)']:.2f}%")
                
                # Risk assessment
                risk_level = '🔴 High Risk' if user_row['Anomaly Rate (%)'] > 15 else '🟡 Medium Risk' if user_row['Anomaly Rate (%)'] > 5 else '🟢 Low Risk'
                col4.metric("Risk Level", risk_level)
                
                st.markdown("---")
                
                # Anomaly types and details
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.markdown("**🔍 Detected Anomaly Types:**")
                    if 'Anomaly Types' in user_row and user_row['Anomaly Types'] != 'none':
                        anomaly_types = user_row['Anomaly Types'].split(', ')
                        for atype in anomaly_types:
                            if atype == 'night_activity':
                                st.markdown("🌙 **Night Activity** - Working during unusual hours (22:00-06:00)")
                            elif atype == 'early_morning':
                                st.markdown("🌅 **Early Morning** - Activity before normal hours (04:00-07:00)")
                            elif atype == 'weekend_work':
                                st.markdown("📅 **Weekend Work** - Activity during weekends")
                            elif atype == 'burst_activity':
                                st.markdown("⚡ **Burst Activity** - Sudden spikes in activity")
                            elif atype == 'excessive_activity':
                                st.markdown("📈 **Excessive Activity** - Consistently high activity levels")
                            elif atype == 'off_hours_activity':
                                st.markdown("🕐 **Off Hours** - Activity outside business hours")
                            else:
                                st.markdown(f"🔍 **{atype.replace('_', ' ').title()}**")
                    else:
                        st.markdown("ℹ️ No specific anomaly patterns detected")
                
                with col_right:
                    st.markdown("**📊 Activity Statistics:**")
                    st.markdown(f"• **Max Anomaly Score:** `{user_row['Max Error']:.6f}`")
                    st.markdown(f"• **Total Activity:** `{user_row['Total Logs']} sequences`")
                    
                    if user_explanations:
                        st.markdown("**🎯 Top Contributing Factors:**")
                        top_exp = user_explanations[0]  # Highest scoring anomaly
                        for feat in top_exp['Top Features'][:2]:
                            deviation = feat['actual_mean'] - feat['expected_mean']
                            st.markdown(f"• **{feat['feature']}:** Deviation `{deviation:.4f}`")
                
                # Show recent incidents if available
                if user_explanations:
                    st.markdown("---")
                    st.markdown("**📋 Recent Anomalous Incidents:**")
                    
                    for i, exp in enumerate(user_explanations[:3], 1):
                        with st.expander(f"Incident #{i} - Score: {exp['Anomaly Score']:.6f} - {exp['Timestamp']}"):
                            st.markdown(f"**Type:** {exp['Anomaly Type']}")
                            st.markdown("**Contributing Features:**")
                            for feat in exp['Top Features']:
                                st.markdown(f"• {feat['feature']}: {feat['contribution']:.6f}")
        else:
            st.warning(f"No anomaly data found for user: {search_user}")
    
    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["Anomaly Report", "Visualizations", "Model Metrics"])

    with tab1:
        st.header("Anomaly Report")
        
        # Report Download
        @st.cache_data
        def get_report_csv(df):
            output = io.BytesIO()
            df.to_csv(output, index=False)
            return output.getvalue()
        
        report_csv = get_report_csv(res['user_stats'])
        st.download_button(
            label="Download Full Report (CSV)",
            data=report_csv,
            file_name="anomaly_report.csv",
            mime="text/csv",
        )
        
        st.subheader("Where are the anomalies?")
        st.dataframe(res['user_stats'].style.format({
            "Anomaly Rate (%)": "{:.2f}%",
            "Max Error": "{:.6f}"
        }).background_gradient(
            cmap='Reds', subset=['Anomaly Rate (%)', 'Anomalies']
        ), use_container_width=True)

        st.subheader("Top 5 Anomalous Users")
        
        # Get top 5 users directly from user_stats (which contains all users with anomalies)
        top_5_users = res['user_stats'].head(5)
        
        if len(top_5_users) == 0:
            st.info("No anomalous users detected.")
        else:
            # Create a mapping of explanations by user for additional details
            user_explanations = {}
            for exp in res['explanations']:
                user = exp['User']
                if user not in user_explanations:
                    user_explanations[user] = []
                user_explanations[user].append(exp)
        
            for rank, (_, user_row) in enumerate(top_5_users.iterrows(), 1):
                user = user_row['User']
                user_exps = user_explanations.get(user, [])
                
                # Sort user's anomalies by error score (highest first) if any explanations exist
                if user_exps:
                    user_exps.sort(key=lambda x: x['Anomaly Score'], reverse=True)
                
                with st.container(border=True):
                    st.markdown(f"### **User #{rank}: {user}**")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.info(f"**Total Sequences:** {user_row['Total Logs']}")
                        st.warning(f"**Anomalous Sequences:** {user_row['Anomalies']}")
                        st.error(f"**Anomaly Rate:** {user_row['Anomaly Rate (%)']:.2f}%")
                        
                        # Show anomaly types from user_stats if available
                        if 'Anomaly Types' in user_row and user_row['Anomaly Types'] != 'none':
                            st.warning(f"**Detected Anomaly Types:** {user_row['Anomaly Types']}")
                        else:
                            st.warning(f"**Detected Anomaly Types:** Pattern-based detection")
                        
                        st.error(f"**Risk Level:** {'🔴 High' if user_row['Anomaly Rate (%)'] > 15 else '🟡 Medium' if user_row['Anomaly Rate (%)'] > 5 else '🟢 Low'}")
                    
                    with col2:
                        features_md = "**Key Statistics:**\n"
                        features_md += f"• **Max Error Score:** `{user_row['Max Error']:.6f}`\n"
                        features_md += f"• **Anomaly Count:** `{user_row['Anomalies']}`\n"
                        
                        # Show top contributing features if we have detailed explanations
                        if user_exps:
                            features_md += f"\n**Top Contributing Features:**\n"
                            top_exp = user_exps[0]
                            for j, feat in enumerate(top_exp['Top Features'][:2]):
                                deviation = feat['actual_mean'] - feat['expected_mean']
                                features_md += f"• **{feat['feature']}:** (Deviation: `{deviation:.4f}`)\n"
                        
                        st.warning(features_md)

    with tab2:
        st.header("Visualizations")
        
        c1, c2 = st.columns(2)
        with c1:
            # Reconstruction Error (Histogram)
            fig_hist = plot_error_dist_hist(res['mse'], res['threshold'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # User Anomaly Heatmap (Bar Chart)
            fig_user = plot_user_barchart(res['user_stats'])
            st.plotly_chart(fig_user, use_container_width=True)
            
        with c2:
            # Reconstruction Error (Scatter)
            fig_scatter = plot_error_scatter(res['mse'], res['anomalies'], res['threshold'])
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Feature Contributions
            fig_feat = plot_feature_contributions(res['explanations'])
            st.plotly_chart(fig_feat, use_container_width=True)

    with tab3:
        st.header("Model Metrics")
        st.info("These metrics reflect the model's training performance. Run `train_model.py` to generate them.")
        
        # Load and display training plots
        loss_exists, roc_exists, fig_loss, fig_roc = load_training_plots()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Reconstruction Error Curve")
            if loss_exists:
                st.image('training_loss.png', caption='Reconstruction Error Curve', use_column_width=True)
            else:
                st.plotly_chart(fig_loss, use_container_width=True)

        with c2:
            st.subheader("ROC-AUC Score")
            if roc_exists:
                st.image('roc_curves.png', caption='ROC-AUC Score', use_column_width=True)
            else:
                st.plotly_chart(fig_roc, use_container_width=True)
        
        st.subheader("Live Model Performance")
        col1, col2 = st.columns(2)
        
        # Calculate model accuracy based on threshold classification
        total_sequences = res['total_sequences']
        anomalies_detected = res['anomalies_detected']
        normal_sequences = total_sequences - anomalies_detected
        
        # Accuracy: percentage of sequences correctly classified
        # Assuming normal behavior is the majority class
        accuracy = (normal_sequences / total_sequences) * 100 if total_sequences > 0 else 0
        
        col1.metric(
            "Model Accuracy",
            f"{accuracy:.2f}%",
            help="Percentage of sequences classified as normal (below threshold)"
        )
        
        col2.metric(
            "Detection Threshold",
            f"{res['threshold']:.6f}",
            help=f"Current anomaly detection threshold ({THRESHOLD_PCT}th percentile of reconstruction errors)"
        )

else:
    st.info("Click 'Run Analysis' to begin.")
