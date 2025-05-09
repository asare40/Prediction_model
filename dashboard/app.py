# -*- coding: utf-8 -*-
"""
Created on Fri May  9 06:33:54 2025

@author: kings
"""

# dashboard/app.py
import os
import sys
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import PROCESSED_DATA_DIR, MODEL_DIR

# Load data and models
def load_data():
    """Load processed data for dashboard"""
    try:
        jamb_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'jamb_enhanced.csv'))
        return jamb_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_models():
    """Load trained models"""
    models = {}
    try:
        for model_name in ["jamb_score_regressor", "jamb_pass_classifier", "jamb_xgb_regressor"]:
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return {}

# Load data and models
data = load_data()
models = load_models()

# Initialize Dash app
app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                title="Nigerian Educational Analytics Dashboard")

# Add custom CSS for header styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .header-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                margin-bottom: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .header-left {
                flex: 1;
            }
            .header-right {
                text-align: right;
                padding: 10px;
                background-color: #f0f7ff;
                border-radius: 4px;
                border-left: 4px solid #1C4E80;
            }
            .user-info {
                font-size: 0.9rem;
                color: #555;
            }
            .user-name {
                color: #1C4E80;
                font-weight: bold;
            }
            .time-info {
                font-size: 0.9rem;
                margin-top: 4px;
                color: #666;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define app layout
app.layout = html.Div([
    # Header with title, date/time, and user info
    html.Div([
        # Title and main header
        html.Div([
            html.H1("Nigerian Educational Analytics Dashboard", 
                    style={'color': '#1C4E80', 'marginBottom': 10}),
        ], className="header-left"),
        
        # User and time info
        html.Div([
            html.Div([
                html.Span("Current User: ", className="user-label"),
                html.Span("asare40", className="user-name")
            ], className="user-info"),
            html.Div([
                html.Span("Date/Time (UTC): ", className="time-label"),
                html.Span("2025-05-09 12:04:21", className="time-value")
            ], className="time-info")
        ], className="header-right")
    ], className="header-container"),
    
    html.Div([
        html.Div([
            html.H3("Overview Statistics", style={'color': '#1C4E80'}),
            html.Div(id="overview-stats", className="stats-container")
        ], className="overview-section"),
        
        html.Div([
            html.H3("Performance Analysis", style={'color': '#1C4E80'}),
            dcc.Tabs([
                dcc.Tab(label="Score Distribution", children=[
                    html.Div([
                        dcc.Graph(id="score-distribution")
                    ])
                ]),
                dcc.Tab(label="Factor Analysis", children=[
                    html.Div([
                        html.Label("Select Factor:"),
                        dcc.Dropdown(
                            id="factor-dropdown",
                            options=[
                                {'label': 'Study Hours', 'value': 'Study_Hours_Per_Week'},
                                {'label': 'Teacher Quality', 'value': 'Teacher_Quality'},
                                {'label': 'Distance to School', 'value': 'Distance_To_School'},
                                {'label': 'School Type', 'value': 'School_Type'},
                                {'label': 'Parent Involvement', 'value': 'Parent_Involvement'},
                                {'label': 'Access to Learning Materials', 'value': 'Access_To_Learning_Materials'}
                            ],
                            value='Study_Hours_Per_Week'
                        ),
                        dcc.Graph(id="factor-analysis")
                    ])
                ]),
                dcc.Tab(label="Correlation Matrix", children=[
                    html.Div([
                        dcc.Graph(id="correlation-matrix")
                    ])
                ])
            ])
        ], className="analysis-section"),
        
        html.Div([
            html.H3("Student Prediction Tool", style={'color': '#1C4E80'}),
            html.Div([
                html.Div([
                    html.Label("Study Hours Per Week:"),
                    dcc.Slider(
                        id="study-hours-input",
                        min=0,
                        max=40,
                        step=1,
                        value=20,
                        marks={i: str(i) for i in range(0, 41, 5)}
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("Teacher Quality (1-5):"),
                    dcc.Slider(
                        id="teacher-quality-input",
                        min=1,
                        max=5,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 6)}
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("Attendance Rate (%):"),
                    dcc.Slider(
                        id="attendance-input",
                        min=50,
                        max=100,
                        step=5,
                        value=80,
                        marks={i: str(i) for i in range(50, 101, 10)}
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("Distance to School (km):"),
                    dcc.Input(
                        id="distance-input",
                        type="number",
                        min=0.1,
                        max=20,
                        step=0.1,
                        value=5.0
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("School Type:"),
                    dcc.RadioItems(
                        id="school-type-input",
                        options=[
                            {'label': 'Public', 'value': 'Public'},
                            {'label': 'Private', 'value': 'Private'}
                        ],
                        value='Public'
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("School Location:"),
                    dcc.RadioItems(
                        id="location-input",
                        options=[
                            {'label': 'Urban', 'value': 'Urban'},
                            {'label': 'Rural', 'value': 'Rural'}
                        ],
                        value='Urban'
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("Extra Tutorials:"),
                    dcc.RadioItems(
                        id="tutorials-input",
                        options=[
                            {'label': 'Yes', 'value': 'Yes'},
                            {'label': 'No', 'value': 'No'}
                        ],
                        value='No'
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("Access to Learning Materials:"),
                    dcc.RadioItems(
                        id="materials-input",
                        options=[
                            {'label': 'Yes', 'value': 'Yes'},
                            {'label': 'No', 'value': 'No'}
                        ],
                        value='Yes'
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("Parent Involvement:"),
                    dcc.RadioItems(
                        id="parent-input",
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'}
                        ],
                        value='Medium'
                    )
                ], className="input-group"),
                
                html.Div([
                    html.Label("IT Knowledge:"),
                    dcc.RadioItems(
                        id="it-input",
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'}
                        ],
                        value='Medium'
                    )
                ], className="input-group"),
                
                html.Button("Predict Score", id="predict-button", className="predict-button"),
                
                html.Div(id="prediction-output", className="prediction-output")
            ], className="prediction-form")
        ], className="prediction-section")
    ], className="main-container"),
    
    html.Footer([
        html.P("Nigerian Educational Analytics Project | Created on May 9, 2025",
              style={'textAlign': 'center', 'color': '#666666'})
    ], className="footer")
])

# Define callback for overview statistics
@app.callback(
    Output("overview-stats", "children"),
    Input("overview-stats", "children")
)
def update_stats(n):
    if data.empty:
        return html.Div("No data available")
        
    avg_score = data['JAMB_Score'].mean()
    pass_rate = (data['JAMB_Score'] >= 200).mean() * 100
    top_performers = (data['JAMB_Score'] >= 250).mean() * 100
    
    stats = [
        html.Div([
            html.H4(f"{avg_score:.1f}"),
            html.P("Average JAMB Score")
        ], className="stat-card"),
        html.Div([
            html.H4(f"{pass_rate:.1f}%"),
            html.P("Pass Rate (≥200)")
        ], className="stat-card"),
        html.Div([
            html.H4(f"{top_performers:.1f}%"),
            html.P("Top Performers (≥250)")
        ], className="stat-card")
    ]
    
    return stats

# Define callback for score distribution
@app.callback(
    Output("score-distribution", "figure"),
    Input("score-distribution", "id")
)
def update_score_distribution(n):
    if data.empty:
        return {}
        
    fig = px.histogram(
        data, 
        x="JAMB_Score",
        nbins=30,
        color_discrete_sequence=['#1C4E80'],
        title="JAMB Score Distribution"
    )
    
    fig.add_vline(x=200, line_dash="dash", line_color="red",
                  annotation_text="Pass Threshold", annotation_position="top right")
    
    fig.update_layout(
        xaxis_title="JAMB Score",
        yaxis_title="Number of Students",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Define callback for factor analysis
@app.callback(
    Output("factor-analysis", "figure"),
    Input("factor-dropdown", "value")
)
def update_factor_analysis(factor):
    if data.empty:
        return {}
    
    if factor in ['Study_Hours_Per_Week', 'Teacher_Quality', 'Distance_To_School']:
        # For numeric factors, create scatter plot
        fig = px.scatter(
            data,
            x=factor,
            y="JAMB_Score",
            trendline="ols",
            color_discrete_sequence=['#1C4E80'],
            title=f"Relationship between {factor.replace('_', ' ')} and JAMB Score",
            opacity=0.7
        )
        
    else:
        # For categorical factors, create box plot
        fig = px.box(
            data,
            x=factor,
            y="JAMB_Score",
            color=factor,
            title=f"JAMB Score by {factor.replace('_', ' ')}",
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Define callback for correlation matrix
@app.callback(
    Output("correlation-matrix", "figure"),
    Input("correlation-matrix", "id")
)
def update_correlation_matrix(n):
    if data.empty:
        return {}
    
    # Select numeric columns for correlation
    numeric_cols = ['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate', 
                   'Teacher_Quality', 'Distance_To_School']
    
    corr = data[numeric_cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix of Key Factors"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Define callback for prediction tool - FIXED VERSION TO PREVENT STACK OVERFLOW
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("study-hours-input", "value"),
    State("teacher-quality-input", "value"),
    State("attendance-input", "value"),
    State("distance-input", "value"),
    State("school-type-input", "value"),
    State("location-input", "value"),
    State("tutorials-input", "value"),
    State("materials-input", "value"),
    State("parent-input", "value"),
    State("it-input", "value"),
    prevent_initial_call=True  # Prevent callback execution on initial load
)
def update_prediction(n_clicks, study_hours, teacher_quality, attendance, distance,
                      school_type, location, tutorials, materials, parent, it):
    if not n_clicks:
        return html.Div([
            html.P("Fill in the student details and click 'Predict Score'")
        ])
    
    try:
        if not models or 'jamb_xgb_regressor' not in models:
            return html.Div([
                html.P("Prediction model not available", style={'color': 'red'})
            ])
        
        # Create input features dataframe with explicit type conversion
        input_data = pd.DataFrame({
            'Study_Hours_Per_Week': [float(study_hours)],
            'Teacher_Quality': [float(teacher_quality)],
            'Attendance_Rate': [float(attendance)],
            'Distance_To_School': [float(distance)],
            'School_Type': [str(school_type)],
            'School_Location': [str(location)],
            'Extra_Tutorials': [str(tutorials)],
            'Access_To_Learning_Materials': [str(materials)],
            'Parent_Involvement': [str(parent)],
            'IT_Knowledge': [str(it)]
        })
        
        # Add engineered features - simplified to avoid potential issues
        input_data['School_Quality_Index'] = float(teacher_quality) * 0.6
        if materials == 'Yes':
            input_data.loc[0, 'School_Quality_Index'] += 0.4
            
        input_data['Engagement_Level'] = float(attendance) / 20.0
        if tutorials == 'Yes':
            input_data.loc[0, 'Engagement_Level'] += 1.0
        if parent == 'High':
            input_data.loc[0, 'Engagement_Level'] += 1.0
            
        input_data['Distance_Barrier'] = 1.0 / (1.0 + float(distance))
        
        # Make predictions - handle as simple float values
        predicted_score = float(models['jamb_xgb_regressor'].predict(input_data)[0])
        pass_probability = float(models['jamb_pass_classifier'].predict_proba(input_data)[0][1]) * 100
        
        # Determine risk level
        if predicted_score >= 250:
            risk_level = "Low"
            risk_color = "green"
        elif predicted_score >= 200:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"
        
        # Generate recommendations - simplified
        recommendations = []
        
        if study_hours < 20:
            recommendations.append("Increase study time to at least 20 hours per week")
        if attendance < 85:
            recommendations.append("Improve class attendance to at least 85%")
        if tutorials == 'No':
            recommendations.append("Consider enrolling in extra tutorial classes")
        if materials == 'No':
            recommendations.append("Ensure access to required learning materials")
        if distance > 10:
            recommendations.append("Consider finding accommodation closer to school")
        
        # Return simple HTML structure
        return html.Div([
            html.H4("Prediction Results"),
            html.Div([
                html.Div([
                    html.H2(f"{predicted_score:.1f}", style={'color': risk_color}),
                    html.P("Predicted JAMB Score")
                ], className="prediction-card"),
                html.Div([
                    html.H2(f"{pass_probability:.1f}%", style={'color': 'green' if pass_probability >= 70 else 'orange'}),
                    html.P("Chance of Scoring 200+")
                ], className="prediction-card"),
                html.Div([
                    html.H2(risk_level, style={'color': risk_color}),
                    html.P("Risk Level")
                ], className="prediction-card")
            ], className="prediction-results"),
            
            html.Div([
                html.H4("Recommendations:"),
                html.Ul([html.Li(rec) for rec in recommendations]) if recommendations else 
                    html.P("No specific recommendations at this time.")
            ], className="recommendations")
        ])
        
    except Exception as e:
        # Handle errors gracefully
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return html.Div([
            html.P(f"Error making prediction: {str(e)}", style={'color': 'red'})
        ])

# Run app
if __name__ == '__main__':
    try:
        # Print access instructions
        local_ip = "127.0.0.1"
        print("\n" + "="*60)
        print(f"Nigerian Educational Analytics Dashboard is running!")
        print(f"Access the dashboard at: http://localhost:8050 or http://127.0.0.1:8050")
        print("="*60 + "\n")
        
        # Run the app
        app.run(
            debug=True,
            host="localhost",  # Changed from 0.0.0.0 to localhost for better compatibility
            port=8050
        )
    except Exception as e:
        print(f"Error starting the dashboard: {e}")