import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
from pythermalcomfort.models import pmv_ppd_iso, utci
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from datetime import datetime
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Optional
import joblib

st.markdown("""
<style>
/* Make sidebar background and labels white */
[data-testid=stSidebar] {
  background-color: #4d85e5;
}

[data-testid=stSidebar] .stSelectbox label, 
[data-testid=stSidebar] .stMultiSelect label,
[data-testid=stSidebar] h1, 
[data-testid=stSidebar] h2, 
[data-testid=stSidebar] h3, 
[data-testid=stSidebar] h4,
[data-testid=stSidebar] p {
  color: white !important;
  font-weight: bold;
}

/* Make text inside dropdown/selectbox black */
[data-testid=stSidebar] .stSelectbox div[data-baseweb="select"] span,
[data-testid=stSidebar] .stMultiSelect div[data-baseweb="select"] span {
  color: black !important;
}

/* Make dropdown menu text black */
div[data-testid="stSidebar"] div[role="listbox"] ul li {
  color: black !important;
}

/* Style for the button text */
[data-testid=stSidebar] button {
  color: white !important;
  font-weight: bold;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div style='text-align: center; margin-bottom: 12px;'>
    <img src="https://raw.githubusercontent.com/LLM-AI-INDIA/GenAI-Bootcamp-FEB2025/main/Lab-4/image/default_logo.png"
         alt="App Logo"
         style="width:300px; height:150px; border-radius:10%;" />
</div>
<div style='text-align: center; color: #003366; font-size: 23px; font-weight: 500; margin-bottom: 12px;'>
    Our Multi-Agent Orchestration simplifies smart building maintenance
</div>
<div style='text-align: center; color: blue; font-size: 16px; margin-top: -10px;'>
    The agentic approach saves time, reduce costs, improves accuracy and enhances user experience
</div>
<hr style="border: 1px solid gray; height:2.5px; margin-top:0px; width:100%; background-color:gray;">
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password", value=st.session_state.get('openai_api_key', ''))
st.session_state['openai_api_key'] = openai_api_key
st.markdown("</div>", unsafe_allow_html=True)

# Set the API key as an environment variable
if openai_api_key:
    import os
    os.environ['OPENAI_API_KEY'] = openai_api_key

# Add after the header section and before the API configuration
st.markdown("<div style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# First table: Agent Configuration
#st.subheader("Agent Configuration")

# Create a static dataframe for the agent tools
tools_df = pd.DataFrame({
    "Tool": ["Thermal Comfort Calculator", "Energy Forecasting", "Space Analytics", "Layout Recommendation"],
    "Input": ["Indoor temperature, humidity, air speed, activity level, clothing insulation", "Power usage, HVAC status, lighting/appliance data, energy tariff", "Room-level data: occupancy, temperature, CO₂, light, humidity, energy usage", "Room usage metrics: power, occupancy, light, humidity, function patterns"],
    "Output": ["PMV, PPD, UTCI scores with thermal comfort classification", "Energy optimization actions with estimated savings & ROI", "Zone classification, underutilization alerts, consolidation suggestions", "Layout adjustments, multi-use zone identification, functional mapping"]
})

# Display the tools table
#st.table(tools_df)

# Second table: Agent Technology Stack


st.markdown("</div>", unsafe_allow_html=True)



# Initialize session state for storing data between agents
if 'thermal_data' not in st.session_state:
    st.session_state.thermal_data = None
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'results' not in st.session_state:
    st.session_state.results = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rooms_df" not in st.session_state:
    st.session_state.rooms_df = None
if "num_rooms" not in st.session_state:
    st.session_state.num_rooms = 5
if "city_name" not in st.session_state:
    st.session_state.city_name = "New York"
if 'use_case' not in st.session_state:
    st.session_state.use_case = None


with st.sidebar:
    st.header("Agent Configuration")
    
    # Industry selection
    industry = st.selectbox("Industry", [
    "",  # Empty/default
    "Energy Management in Buildings",
    "Public Sector"
], format_func=lambda x: "Select Industry" if x == "" else x)
    st.session_state.industry = industry
    
    # Use Case selection
    use_case = st.selectbox("Use Case", ["",
        "Thermal Comfort Analysis", "Energy Optimization", 
        "Space Optimization", "Layout Planning"
    ])
    st.session_state.use_case = use_case

    # Define agent tools, platforms, LLMs, frameworks and libraries for each use case
    agent_data = {
        "Thermal Comfort Analysis": {
            "tools": ["Thermal Comfort Calculator", "Weather API"],
            "platforms": ["CrewAI", "LangChain"],
            "llms": ["gpt-3.5-turbo", "gpt-4"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["pythermalcomfort", "pandas", "requests"]
        },
        "Energy Optimization": {
            "tools": ["Energy Forecasting", "Weather API"],
            "platforms": ["CrewAI", "LangChain"],
            "llms": ["gpt-4", "claude-3-opus"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["numpy", "scikit-learn", "pandas", "requests"]
        },
        "Space Optimization": {
            "tools": ["Space Analytics", "Occupancy Prediction"],
            "platforms": ["CrewAI", "LangChain"],
            "llms": ["gpt-4-turbo", "claude-3-sonnet"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["pandas", "statsmodels", "scikit-learn"]
        },
        "Layout Planning": {
            "tools": ["Layout Recommendation"],
            "platforms": ["CrewAI", "langChain"],
            "llms": ["gpt-3.5-turbo", "gpt-4-turbo"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["streamlit", "numpy", "plotly"]
        }
    }

    # Get the selected use case data
    tool_ui_map = {
    "Thermal Comfort Analysis": [
        "Get Building Summary",
        "Get Weather Summary"
    ],
    "Energy Optimization": [
        "Simulate Energy Reduction",
        "Explain Energy Forecast"
    ],
    "Space Optimization": [
        "Predict Occupancy Trend",
        "Suggest Room Consolidation",
        "Suggest Space Rezoning"
    ],
    "Layout Planning": [
        "Recommend Layout Plan",
        "Identify Multiuse Zones",
        "Room Function Mapping"
    ]
}

    selected_tool_labels = tool_ui_map.get(use_case, [])
    default_tool = selected_tool_labels[0] if selected_tool_labels else None
    st.multiselect("Agent Tools", options=selected_tool_labels, default=[default_tool] if default_tool else [])


    # Fallback if use_case is not found
    agent_info = agent_data.get(use_case, {
    "tools": [],
    "platforms": [],
    "llms": [],
    "frameworks": [],
    "libraries": []
})

    agent_platform = st.selectbox("Agent Platform", agent_info["platforms"])
    agent_llm = st.selectbox("Agent LLM", agent_info["llms"])
    agent_framework = st.selectbox("Agent Framework", agent_info["frameworks"])
    agent_libraries = st.multiselect("Agent Libraries", agent_info["libraries"])


    agent_tools_table_data = {
    "Thermal Comfort Analysis": [
        {
            "Agent": "Thermal Comfort Analyst",
            "Tool": "Get Building Summary",
            "Function": "Provides average environmental and occupancy stats"
        },
        {
            "Agent": "Thermal Comfort Analyst",
            "Tool": "Get Weather Summary",
            "Function": "Fetches current weather data for given city coordinates"
        }
    ],
    "Energy Optimization": [
        {
            "Agent": "Energy Optimization Engineer",
            "Tool": "Simulate Energy Reduction",
            "Function": "Estimates energy savings from optimized HVAC & lighting"
        },
        {
            "Agent": "Energy Optimization Engineer",
            "Tool": "Explain Energy Forecast",
            "Function": "LLM-based forecast based on room and weather conditions"
        }
    ],
    "Space Optimization": [
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Get Building Summary",
            "Function": "Summarizes average metrics from building data"
        },
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Predict Occupancy Trend",
            "Function": "Forecasts room occupancy using ARIMA"
        },
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Suggest Room Consolidation",
            "Function": "Identifies underused rooms for merging/closing"
        },
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Suggest Space Rezoning",
            "Function": "Clusters rooms into functional zones using KMeans"
        }
    ],
    "Layout Planning": [
        {
            "Agent": "Layout Planner",
            "Tool": "Recommend Layout Plan",
            "Function": "Suggests layout changes to optimize space utilization"
        },
        {
            "Agent": "Layout Planner",
            "Tool": "Identify Multiuse Zones",
            "Function": "Detects rooms suitable for multi-functional use"
        },
        {
            "Agent": "Layout Planner",
            "Tool": "Room Function Mapping",
            "Function": "Assigns room roles based on occupancy, light, and energy"
        }
    ]
}

    # Default table when no use case is selected
    default_llm_config = pd.DataFrame({
    "Agent": [
        "Thermal Comfort Analyst",
        "Energy Optimization Engineer",
        "Space Optimization Assistant",
        "Layout Planner"
    ],
    "LLM Used": [
        "gpt-3.5-turbo, gpt-4",
        "gpt-4, claude-3-opus",
        "gpt-4-turbo, claude-3-sonnet",
        "gpt-3.5-turbo, gpt-4-turbo"
    ],
    "Purpose": [
        "Thermal analysis based on ISO/ASHRAE standards",
        "Forecasts and cost-saving strategy generation",
        "Occupancy forecasting, clustering explanation",
        "Layout suggestions and room function mapping"
    ]
})

# Use-case-specific LLM mapping
    agent_llm_table_data = {
    "Thermal Comfort Analysis": [
        {
            "Agent": "Thermal Comfort Analyst",
            "LLM Used": "gpt-3.5-turbo, gpt-4",
            "Purpose": "Thermal analysis based on ISO/ASHRAE standards"
        }
    ],
    "Energy Optimization": [
        {
            "Agent": "Energy Optimization Engineer",
            "LLM Used": "gpt-4, claude-3-opus",
            "Purpose": "Forecasts and cost-saving strategy generation"
        }
    ],
    "Space Optimization": [
        {
            "Agent": "Space Optimization Assistant",
            "LLM Used": "gpt-4-turbo, claude-3-sonnet",
            "Purpose": "Occupancy forecasting, clustering explanation"
        }
    ],
    "Layout Planning": [
        {
            "Agent": "Layout Planner",
            "LLM Used": "gpt-3.5-turbo, gpt-4-turbo",
            "Purpose": "Layout suggestions and room function mapping"
        }
    ]
}



    # Add the reset chat button
    st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    if st.button("Clear/Reset", key="reset_chat_button"):
        # Clear the messages in session state
        st.session_state.messages = []
        # Clear the context information
        st.session_state.available_data = []
        st.rerun()


    # --- LOGO CSS for bottom placement and round/white style ---
    st.markdown("""
    <style>
    /* Make sidebar a flex column */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    /* Spacer grows to push logos down */
    .spacer {
        flex: 1 1 auto;
        height: 8px;
    }
    .sidebar-bottom-logos {
        width: 100%;
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 24px;
    }
    .sidebar-bottom-logos img {
        width: 45px;
        height: 45px;
        border-radius: 10%;
        background: #fff;
        padding: 0px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        object-fit: scale-down;
        border: 2px solid #fff;
    }
    [data-testid="stButton"] button {
    background-color: #33cc33;
    color: white;
    font-weight: bold;
    width: 100%;
    margin-bottom: 15px;
}
    </style>
    """, unsafe_allow_html=True)

    # --- SPACER to push logos to bottom ---
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # --- LOGOS PLACEMENT ---
    st.markdown('''
<div class="sidebar-bottom-logos">
    <img src="https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png" alt="AWS Logo">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/googlecloud/googlecloud-original.svg" alt="GCP Logo">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg" alt="Azure Logo">
</div>
''', unsafe_allow_html=True)




@st.cache_resource
def get_openai_client(api_key):
    if api_key and api_key.startswith('sk-'):
        return OpenAI(api_key=api_key)
    return None

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['current']
    return None

@st.cache_data
def get_sample_data():
    sample_data = {
        "current_temperature": [23.5],
        "humidity_level": [45.0],
        "ambient_light_level": [500],
        "current_power_consumption": [45.5],
        "energy_tariff_rate": [0.15],
        "hvac_status": ["On - Cooling"],
        "lighting_status": ["Dimmed"],
        "appliance_status": ["Essential Only"],
        "lighting_power_usage": [12.3],
        "appliance_power_usage": [18.7]
    }
    return pd.DataFrame(sample_data)

def calculate_thermal_metrics(inputs):
    pmv_result = pmv_ppd_iso(tdb=inputs['tdb'], tr=inputs['tr'], vr=inputs['vr'],
                           rh=inputs['rh'], met=inputs['met'], clo=inputs['clo'])
    utci_result = utci(tdb=inputs['tdb'], tr=inputs['tr'], v=inputs['vr'], rh=inputs['rh'])
    return {
        'pmv': round(pmv_result.pmv, 2),
        'ppd': round(pmv_result.ppd, 1),
        'utci': round(utci_result.utci, 1),
        'utci_category': utci_result.stress_category
    }

def _get_building_summary_internal() -> str:
    try:
        df = pd.read_csv("update.csv")
        df.columns = [col.strip() for col in df.columns]
        if 'date' in df.columns:
            df.drop(columns=['date'], inplace=True)
        return (
            f"Average Temperature: {df['Temperature'].mean():.2f} °C\n"
            f"Average CO₂ Level: {df['CO2'].mean():.2f} ppm\n"
            f"Average Light Level: {df['Light'].mean():.2f} Lux\n"
            f"Average Humidity: {df['Humidity'].mean():.2f} %\n"
            f"Average Occupancy: {df['Occupancy'].mean():.2f}"
        )
    except Exception as e:
        # Fallback if file doesn't exist
        if st.session_state.rooms_df is not None:
            df = st.session_state.rooms_df
            return (
                f"Average Temperature: {df['Temperature'].mean():.2f} °C\n"
                f"Average CO₂ Level: {df['CO2'].mean():.2f} ppm\n"
                f"Average Light Level: {df['Light'].mean():.2f} Lux\n"
                f"Average Humidity: {df['Humidity'].mean():.2f} %\n"
                f"Average Occupancy: {df['Occupancy'].mean():.2f}"
            )
        return "Building summary data not available."

def _get_weather_summary_internal(city: str) -> str:
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_resp = requests.get(geo_url).json()
        results = geo_resp.get("results", [])
        if not results:
            return f"City '{city}' not found."

        lat = results[0]['latitude']
        lon = results[0]['longitude']

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m&timezone=auto"
        weather_resp = requests.get(weather_url).json()
        data = weather_resp.get("current", {})

        return (
            f"Weather for {city} (lat: {lat}, lon: {lon}):\n"
            f"Outside Temp: {data.get('temperature_2m', 'N/A')} °C\n"
            f"Humidity: {data.get('relative_humidity_2m', 'N/A')}%\n"
            f"Cloud Cover: {data.get('cloud_cover', 'N/A')}%\n"
            f"Wind Speed: {data.get('wind_speed_10m', 'N/A')} km/h"
        )
    except Exception as e:
        return f"Error fetching weather data: {e}"

def call_openai_fallback(prompt: str, api_key: str, model: str, label: str) -> str:
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a focused assistant specialized in {label}. Do not provide overlapping content from other domains."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[{label}] OpenAI fallback failed: {e}"

def process_chat_query(query):
    if not st.session_state.openai_api_key:
        return "Please enter a valid OpenAI API key to use the chat feature."
    
    # Build context from available data and reports
    context = "You are an expert in building optimization. Here is the relevant context:\n\n"
    
    # Add thermal data if available
    if st.session_state.thermal_data:
        thermal_data = st.session_state.thermal_data
        context += f"THERMAL DATA:\n"
        context += f"Building Type: {thermal_data['building_type']}\n"
        context += f"Season: {thermal_data['season']}\n"
        context += f"Indoor Temperature: {thermal_data['tdb']}°C\n"
        context += f"Relative Humidity: {thermal_data['rh']}%\n"
        context += f"PMV: {thermal_data['pmv']}\n"
        context += f"PPD: {thermal_data['ppd']}%\n"
        context += f"UTCI: {thermal_data['utci']}°C\n"
        context += f"UTCI Category: {thermal_data['utci_category']}\n\n"
    
    # Add generated reports if available
    if "thermal_analysis" in st.session_state.results:
        context += f"THERMAL ANALYSIS REPORT:\n{st.session_state.results['thermal_analysis']}\n\n"
    
    if "energy_optimization" in st.session_state.results:
        context += f"ENERGY OPTIMIZATION REPORT:\n{st.session_state.results['energy_optimization']}\n\n"
    
    if "space_optimization" in st.session_state.results:
        context += f"SPACE OPTIMIZATION REPORT:\n{st.session_state.results['space_optimization']}\n\n"
    
    if "layout_recommendation" in st.session_state.results:
        context += f"LAYOUT RECOMMENDATIONS:\n{st.session_state.results['layout_recommendation']}\n\n"
    
    # Add building summary if available
    try:
        building_summary = _get_building_summary_internal()
        context += f"BUILDING SUMMARY:\n{building_summary}\n\n"
    except:
        pass
    
    # Add weather data if available
    try:
        weather_summary = _get_weather_summary_internal(st.session_state.city_name)
        context += f"WEATHER SUMMARY:\n{weather_summary}\n\n"
    except:
        pass
    
    context += "Answer the user's question based on this information."
    
    try:
        client = OpenAI(api_key=st.session_state.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"



def create_thermal_agent():
    selected_labels = [
        "Get Building Summary",
        "Get Weather Summary"
    ]
    return Agent(
        role="Thermal Comfort Analyst",
        goal="Analyze indoor environmental parameters and generate technical thermal comfort reports",
        backstory="Expert in building science and ISO/ASHRAE thermal comfort standards",
        tools=[tool_function_map[label] for label in selected_labels],
        verbose=True,
        allow_delegation=False
    )


def create_energy_agent():
    selected_labels = [
        "Simulate Energy Reduction",
        "Explain Energy Forecast"
    ]
    return Agent(
        role="Energy Optimization Engineer",
        goal="Recommend energy-saving actions while maintaining thermal comfort",
        backstory="Specialist in building energy systems and cost-benefit analysis",
        tools=[tool_function_map[label] for label in selected_labels],
        verbose=True,
        allow_delegation=False
    )


def create_space_optimization_agent():
    selected_labels = [
        "Get Building Summary",
        "Predict Occupancy Trend",
        "Suggest Room Consolidation",
        "Suggest Space Rezoning"
    ]
    return Agent(
        role="Space Optimization Assistant",
        goal="Optimize room allocation using occupancy and environment data",
        backstory="Expert in spatial optimization for sustainable building operations",
        tools=[tool_function_map[label] for label in selected_labels],
        verbose=True,
        allow_delegation=False
    )



def create_layout_recommendation_agent():
    selected_labels = [
        "Recommend Layout Plan",
        "Identify Multiuse Zones",
        "Room Function Mapping"
    ]
    return Agent(
        role="Layout Planner",
        goal="Analyze room data and suggest long-term layout changes for flexibility and efficiency",
        backstory="Specialist in spatial design and dynamic reconfiguration based on usage metrics",
        tools=[tool_function_map[label] for label in selected_labels],
        verbose=True,
        allow_delegation=False
    )



def create_thermal_analysis_task(inputs, agent):
    return Task(
        description=f"""Calculate thermal comfort metrics and generate a report for a
         {inputs['building_type']} building with: Air temp {inputs['tdb']}°C, Mean radiant temp {inputs['tr']}°C,
          Relative humidity {inputs['rh']}%, Air velocity {inputs['vr']} m/s, Activity level {inputs['met']} met,
           Clothing insulation {inputs['clo']} clo , Do not include any header with date, location, or building type.
Do not include any signature or prepared by section at the end.  """,
        expected_output="Thermal comfort metrics with ASHRAE compliance analysis and a detailed report",
        agent=agent
    )

def create_energy_optimization_task(thermal_data, energy_inputs, agent):
    thermal_data_str = ""
    if thermal_data:
        thermal_data_str = f"""
        Building Type: {thermal_data['building_type']}
        Season: {thermal_data['season']}
        Indoor Temperature: {thermal_data['tdb']}°C
        Relative Humidity: {thermal_data['rh']}%
        PMV: {thermal_data['pmv']}
        PPD: {thermal_data['ppd']}%
        UTCI: {thermal_data['utci']}°C
        UTCI Category: {thermal_data['utci_category']}
        Do not include any header with date, location, or building type.
        Do not include any signature or prepared by section at the end.
        """
    return Task(
        description=f"""Generate energy optimization recommendations based on the following:

        {thermal_data_str if thermal_data else "No thermal comfort data available."}

        Energy inputs:
        - Total power: {energy_inputs['current_power_consumption']} kW
        - HVAC status: {energy_inputs['hvac_status']}
        - Lighting power: {energy_inputs['lighting_power_usage']} kW
        - Appliance power: {energy_inputs['appliance_power_usage']} kW
        - Energy rate: ${energy_inputs['energy_tariff_rate']}/kWh
        """,
        expected_output="Detailed energy optimization report with ROI analysis",
        agent=agent
    )

def create_space_optimization_task(rooms_data, agent):
    return Task(
        description=f"""Analyze the following room data and provide space optimization recommendations:
        
        {rooms_data}
        
        Focus on:
        1. Identifying underutilized spaces
        2. Suggesting room consolidation opportunities
        3. Optimizing space allocation based on occupancy patterns
        4. Recommending zoning improvements
        """,
        expected_output="Detailed space optimization recommendations with specific actions",
        agent=agent
    )

def create_layout_recommendation_task(rooms_data, agent):
    return Task(
        description=f"""Analyze the following room data and provide layout recommendations:
        
        {rooms_data}
        
        Focus on:
        1. Identifying optimal room functions based on usage patterns
        2. Suggesting multi-functional space opportunities
        3. Recommending layout changes to improve efficiency
        4. Providing a functional map for the building
        """,
        expected_output="Detailed layout recommendations with specific actions",
        agent=agent
    )

def run_thermal_analysis(env_params):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate the report.")
        return None
    with st.spinner("Running thermal comfort analysis with CrewAI..."):
        thermal_agent = create_thermal_agent()
        thermal_task = create_thermal_analysis_task(env_params, thermal_agent)
        thermal_crew = Crew(
            agents=[thermal_agent],
            tasks=[thermal_task],
            verbose=True,
            process=Process.sequential
        )
        try:
            thermal_result = thermal_crew.kickoff()
            st.session_state.messages.append({"role": "assistant", "content": thermal_result})
            return thermal_result
            
        except Exception as e:
            st.error(f"Error running thermal analysis: {str(e)}")
            return None

def run_energy_optimization(thermal_data, energy_inputs):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None
    with st.spinner("Running energy optimization with CrewAI..."):
        energy_agent = create_energy_agent()
        energy_task = create_energy_optimization_task(thermal_data, energy_inputs, energy_agent)
        energy_crew = Crew(
            agents=[energy_agent],
            tasks=[energy_task],
            verbose=True,
            process=Process.sequential
        )
        try:
            energy_result = energy_crew.kickoff()
            st.session_state.messages.append({"role": "assistant", "content": energy_result})
            return energy_result
        except Exception as e:
            st.error(f"Error running energy optimization: {str(e)}")
            return None

def run_space_optimization(rooms_df):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None
    
    # Convert dataframe to string representation for the task
    rooms_data = rooms_df.to_string()
    
    with st.spinner("Running space optimization analysis with CrewAI..."):
        space_agent = create_space_optimization_agent()
        space_task = create_space_optimization_task(rooms_data, space_agent)
        
        try:
            # Set environment variable for LiteLLM
            if 'OPENAI_API_KEY' not in os.environ and st.session_state.openai_api_key:
                os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                
            space_crew = Crew(
                agents=[space_agent],
                tasks=[space_task],
                verbose=True,
                process=Process.sequential
            )
            space_result = space_crew.kickoff()
            
            # Add to message history
            st.session_state.messages.append({"role": "assistant", "content": space_result})
            
            return space_result
        except Exception as e:
            error_msg = f"Error running space optimization: {str(e)}"
            st.error(error_msg)
            
            # Add error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return None


def run_layout_recommendation(rooms_df):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None
    
    # Convert dataframe to string representation for the task
    rooms_data = rooms_df.to_string()
    
    with st.spinner("Running layout recommendation analysis with CrewAI..."):
        layout_agent = create_layout_recommendation_agent()
        layout_task = create_layout_recommendation_task(rooms_data, layout_agent)
        layout_crew = Crew(
            agents=[layout_agent],
            tasks=[layout_task],
            verbose=True,
            process=Process.sequential
        )
        try:
            layout_result = layout_crew.kickoff()
            st.session_state.messages.append({"role": "assistant", "content": layout_result})
            return layout_result
        except Exception as e:
            st.error(f"Error running layout recommendation: {str(e)}")
            return None

def chat_ui():
    st.divider()
    st.subheader("💬 Building Intelligence Chat")
    if st.session_state.thermal_data or st.session_state.results:
        available_data = []
        if st.session_state.thermal_data:
            available_data.append("Thermal comfort data")
        if "thermal_analysis" in st.session_state.results:
            available_data.append("Thermal analysis report")
        if "energy_optimization" in st.session_state.results:
            available_data.append("Energy optimization recommendations")
        if "space_optimization" in st.session_state.results:
            available_data.append("Space optimization recommendations")
        if "layout_recommendation" in st.session_state.results:
            available_data.append("Layout recommendations")
        if available_data:
            st.caption(f"Available context: {', '.join(available_data)}")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Interact with LLM"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_chat_query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def thermal_comfort_agent_ui():
    st.header("Thermal Comfort Analyst")
    st.caption("**Goal:** Analyze indoor environmental parameters and generate technical thermal comfort reports")
    st.subheader("Enter Location Coordinates")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.0, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.0, format="%.6f")
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
    weather_data = get_weather_data(lat, lon)
    if weather_data:
        st.success(f"Fetched weather: {weather_data['temperature_2m']}°C, RH {weather_data['relative_humidity_2m']}%, Wind {weather_data['wind_speed_10m']} m/s")
    st.header("Environmental Parameters")
    col1, col2 = st.columns(2)
    with col1:
        building_type = st.selectbox("Building Type", ["Office", "Residential", "Educational"])
        season = st.selectbox("Season", ["Summer", "Winter"])
        tdb = st.number_input("Air Temperature (°C)",
                             value=float(weather_data['temperature_2m']) if weather_data else 23.0)
        tr = st.number_input("Mean Radiant Temperature (°C)", value=tdb)
        rh = st.slider("Relative Humidity (%)", 0, 100,
                     value=int(weather_data['relative_humidity_2m']) if weather_data else 45)
    with col2:
        min_vr = 0.0
        max_vr = 1.0
        default_vr = 0.1
        if weather_data:
            weather_vr = float(weather_data['wind_speed_10m'])
            default_vr = min(max(weather_vr, min_vr), max_vr)
        vr = st.number_input("Air Velocity (m/s)", min_value=min_vr, max_value=max_vr, value=default_vr)
        met = st.select_slider("Activity Level (met)",
                             options=[1.0, 1.2, 1.4, 1.6, 2.0, 2.4], value=1.4)
        clo = st.select_slider("Clothing Insulation (clo)",
                              options=[0.5, 0.7, 1.0, 1.5, 2.0, 2.5], value=0.5)
    if st.button("Execute Thermal Analysis"):
        if not st.session_state.openai_api_key:
            st.error("Please enter a valid OpenAI API key to generate the report.")
        else:
            inputs = {
                'tdb': tdb, 'tr': tr, 'rh': rh, 'vr': vr,
                'met': met, 'clo': clo, 'building_type': building_type,
                'season': season
            }
            metrics = calculate_thermal_metrics(inputs)
            st.session_state.thermal_data = {
                'building_type': building_type,
                'season': season,
                'tdb': tdb,
                'tr': tr,
                'rh': rh,
                'vr': vr,
                'met': met,
                'clo': clo,
                'pmv': metrics['pmv'],
                'ppd': metrics['ppd'],
                'utci': metrics['utci'],
                'utci_category': metrics['utci_category']
            }
            thermal_result = run_thermal_analysis(inputs)
            if thermal_result:
                st.session_state.results["thermal_analysis"] = thermal_result

                st.subheader("Key Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("PMV", f"{metrics['pmv']}", "Neutral (0)" if -0.5 < metrics['pmv'] < 0.5 else "Needs Adjustment")
                col2.metric("PPD", f"{metrics['ppd']}%", help="Predicted Percentage Dissatisfied")
                col3.metric("UTCI", f"{metrics['utci']}°C", metrics['utci_category'])
                st.success("Thermal comfort analysis complete!")
    chat_ui()

def energy_optimization_agent_ui():
    st.header("Energy Optimization Engineer")
    st.caption("**Goal:** Recommend energy-saving actions while maintaining thermal comfort")
    if st.session_state.thermal_data is not None:
        st.subheader("Thermal Comfort Analysis Summary")
        thermal_data = st.session_state.thermal_data
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Building Type", thermal_data['building_type'])
        col2.metric("Indoor Temperature", f"{thermal_data['tdb']}°C")
        col3.metric("PMV", f"{thermal_data['pmv']}")
        col4.metric("UTCI", f"{thermal_data['utci']}°C", thermal_data['utci_category'])
    else:
        st.info("No thermal comfort data available. Consider running the Thermal Comfort Agent first for more comprehensive recommendations.")
    st.subheader("Upload Energy Data")
    sample_df = get_sample_data()
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_df.to_csv(index=False),
        file_name="energy_data_template.csv",
        mime="text/csv"
    )
    uploaded_file = st.file_uploader("Upload CSV file with energy data", type="csv")
    current_temperature = 23.5
    humidity_level = 45.0
    ambient_light_level = 500
    current_power_consumption = 45.5
    energy_tariff_rate = 0.15
    hvac_status = "On - Cooling"
    lighting_status = "Dimmed"
    appliance_status = "Essential Only"
    lighting_power_usage = 12.3
    appliance_power_usage = 18.7
    if st.session_state.thermal_data:
        current_temperature = float(st.session_state.thermal_data['tdb'])
        humidity_level = float(st.session_state.thermal_data['rh'])
        hvac_status = "On - Cooling" if current_temperature > 24 else "On - Heating"
    if uploaded_file is not None:
        try:
            energy_data = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded data from {uploaded_file.name}")
            st.dataframe(energy_data)
            if 'current_temperature' in energy_data.columns:
                current_temperature = float(energy_data['current_temperature'].iloc[0])
            if 'humidity_level' in energy_data.columns:
                humidity_level = float(energy_data['humidity_level'].iloc[0])
            if 'ambient_light_level' in energy_data.columns:
                ambient_light_level = float(energy_data['ambient_light_level'].iloc[0])
            if 'current_power_consumption' in energy_data.columns:
                current_power_consumption = float(energy_data['current_power_consumption'].iloc[0])
            if 'energy_tariff_rate' in energy_data.columns:
                energy_tariff_rate = float(energy_data['energy_tariff_rate'].iloc[0])
            if 'hvac_status' in energy_data.columns:
                hvac_status = str(energy_data['hvac_status'].iloc[0])
            if 'lighting_status' in energy_data.columns:
                lighting_status = str(energy_data['lighting_status'].iloc[0])
            if 'appliance_status' in energy_data.columns:
                appliance_status = str(energy_data['appliance_status'].iloc[0])
            if 'lighting_power_usage' in energy_data.columns:
                lighting_power_usage = float(energy_data['lighting_power_usage'].iloc[0])
            if 'appliance_power_usage' in energy_data.columns:
                appliance_power_usage = float(energy_data['appliance_power_usage'].iloc[0])
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
    st.subheader("Building Energy System Status")
    col1, col2 = st.columns(2)
    with col1:
        current_temperature = st.number_input("Current Indoor Temperature (°C)",
                                            value=current_temperature, step=0.1)
        humidity_level = st.number_input("Humidity Level (%)",
                                       value=humidity_level, step=1.0)
        ambient_light_level = st.number_input("Ambient Light Level (lux)",
                                            value=ambient_light_level, min_value=0, max_value=2000)
        current_power_consumption = st.number_input("Current Power Consumption (kW)",
                                                  value=current_power_consumption, min_value=0.0, step=0.1)
        energy_tariff_rate = st.number_input("Energy Tariff Rate ($/kWh)",
                                            value=energy_tariff_rate, min_value=0.01, step=0.01)
    with col2:
        hvac_status = st.selectbox("HVAC Status",
                                 ["On - Cooling", "On - Heating", "On - Fan Only", "Off"],
                                 index=["On - Cooling", "On - Heating", "On - Fan Only", "Off"].index(hvac_status) if hvac_status in ["On - Cooling", "On - Heating", "On - Fan Only", "Off"] else 0)
        lighting_status = st.selectbox("Lighting Status",
                                     ["Full Brightness", "Dimmed", "Partial (Zone Control)", "Off"],
                                     index=["Full Brightness", "Dimmed", "Partial (Zone Control)", "Off"].index(lighting_status) if lighting_status in ["Full Brightness", "Dimmed", "Partial (Zone Control)", "Off"] else 0)
        appliance_status = st.selectbox("Appliance Status",
                                      ["All Operating", "Essential Only", "Low Power Mode", "Standby"],
                                      index=["All Operating", "Essential Only", "Low Power Mode", "Standby"].index(appliance_status) if appliance_status in ["All Operating", "Essential Only", "Low Power Mode", "Standby"] else 0)
        lighting_power_usage = st.number_input("Lighting Power Usage (kW)",
                                             value=lighting_power_usage, min_value=0.0, step=0.1)
        appliance_power_usage = st.number_input("Appliance Power Usage (kW)",
                                              value=appliance_power_usage, min_value=0.0, step=0.1)
    if st.button("Run Energy Optimization"):
        if not st.session_state.openai_api_key:
            st.error("Please enter a valid OpenAI API key to generate recommendations.")
        else:
            energy_inputs = {
                'current_temperature': current_temperature,
                'humidity_level': humidity_level,
                'ambient_light_level': ambient_light_level,
                'current_power_consumption': current_power_consumption,
                'energy_tariff_rate': energy_tariff_rate,
                'hvac_status': hvac_status,
                'lighting_status': lighting_status,
                'appliance_status': appliance_status,
                'lighting_power_usage': lighting_power_usage,
                'appliance_power_usage': appliance_power_usage
            }
            energy_result = run_energy_optimization(st.session_state.thermal_data, energy_inputs)
            if energy_result:
                st.session_state.results["energy_optimization"] = energy_result

    chat_ui()

def space_optimization_agent_ui():
    st.header("Space Optimization Assistant")
    st.caption("**Goal:** Optimize room allocation using occupancy and environment data")
    
    # Generate live room data if not already in session state
    if "rooms_df" not in st.session_state or st.session_state.get("num_rooms") != st.session_state.num_rooms:
        st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)
    
    rooms_df = st.session_state.rooms_df
    
    st.subheader("Building Overview")
    building_summary = _get_building_summary_internal()
    st.info(building_summary)
    
    st.subheader("Room Data")
    st.dataframe(rooms_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Room Data"):
            st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)
            st.rerun()
    with col2:
        if st.button("Run Space Optimization Analysis"):
            if not st.session_state.openai_api_key:
                st.error("Please enter a valid OpenAI API key to generate recommendations.")
            else:
                space_result = run_space_optimization(st.session_state.rooms_df)
                if space_result:
                    st.session_state.results["space_optimization"] = space_result
                    st.subheader("🤖 AI Space Optimization Recommendations")
                    st.markdown(str(space_result))
    
    # Display visualizations
    st.subheader("Room Occupancy Visualization")
    fig = {
        'data': [
            {
                'x': rooms_df['RoomID'],
                'y': rooms_df['Occupancy'],
                'type': 'bar',
                'marker': {'color': rooms_df['Occupancy'].apply(lambda x: 'red' if x > 0.7 else 'orange' if x > 0.4 else 'green')}
            }
        ],
        'layout': {
            'title': 'Room Occupancy Levels',
            'xaxis': {'title': 'Room ID'},
            'yaxis': {'title': 'Occupancy Rate (0-1)'}
        }
    }
    st.plotly_chart(fig, use_container_width=True)

    
    chat_ui()

def layout_recommendation_agent_ui():
    st.header("Layout Planner")
    st.caption("**Goal:** Analyze room data and suggest long-term layout changes for flexibility and efficiency")
    
    # Generate live room data if not already in session state
    if "rooms_df" not in st.session_state or st.session_state.get("num_rooms") != st.session_state.num_rooms:
        st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)
    
    rooms_df = st.session_state.rooms_df
    
    st.subheader("Current Layout Data")
    st.dataframe(rooms_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Layout Data"):
            st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)
            st.rerun()
    with col2:
        if st.button("Generate Layout Recommendations"):
            if not st.session_state.openai_api_key:
                st.error("Please enter a valid OpenAI API key to generate recommendations.")
            else:
                layout_result = run_layout_recommendation(st.session_state.rooms_df)
                if layout_result:
                    st.session_state.results["layout_recommendation"] = layout_result

    
    # Display visualizations
    st.subheader("Room Usage Analysis")
    
    # Power usage by room
    fig = {
        'data': [
            {
                'x': rooms_df['RoomID'],
                'y': rooms_df['Power_kWh'],
                'type': 'bar',
                'name': 'Power Usage (kWh)',
                'marker': {'color': 'blue'}
            }
        ],
        'layout': {
            'title': 'Power Usage by Room',
            'xaxis': {'title': 'Room ID'},
            'yaxis': {'title': 'Power (kWh)'}
        }
    }
    st.plotly_chart(fig, use_container_width=True)
    
    # Room metrics comparison
    fig2 = {
        'data': [
            {
                'type': 'scatterpolar',
                'r': [row['Occupancy'], row['Temperature']/30, row['CO2']/1200, row['Light']/600, row['Humidity']/100, row['Power_kWh']/4],
                'theta': ['Occupancy', 'Temperature', 'CO2', 'Light', 'Humidity', 'Power'],
                'fill': 'toself',
                'name': row['RoomID']
            } for _, row in rooms_df.iterrows()
        ],
        'layout': {
            'title': 'Room Metrics Comparison',
            'polar': {
                'radialaxis': {'visible': True, 'range': [0, 1]}
            }
        }
    }
    st.plotly_chart(fig2, use_container_width=True)
    
    chat_ui()

def show_agent_tools_table(use_case: Optional[str] = None):
    if not use_case or use_case == "Select Agent":
        st.subheader("Agent Configuration")
        st.table(tools_df)
    else:
        table_data = agent_tools_table_data.get(use_case, [])
        if table_data:
            df = pd.DataFrame(table_data)
            st.subheader(f"{use_case}: Agents and Tools Used")
            st.table(df)

def show_agent_llm_table(use_case: Optional[str] = None):
    if not use_case or use_case == "Select Agent":
        st.subheader("LLM used in this Agents")
        st.table(default_llm_config)
    else:
        table_data = agent_llm_table_data.get(use_case, [])
        if table_data:
            df = pd.DataFrame(table_data)
            st.subheader(f"LLM used in {use_case}")
            st.table(df)




def get_live_room_data(num_rooms: int = 5) -> pd.DataFrame:
    rooms = []
    room_numbers = random.sample(range(1, 101), num_rooms)
    for room_num in room_numbers:
        light = random.randint(100, 600)
        lighting_status = random.choice(["Full", "Dimmed", "Off"])
        temperature = round(random.uniform(18.0, 26.0), 1)
        hvac_status = random.choice(["Heating", "Cooling", "Off"])
        if hvac_status == "Heating" and temperature > 24:
            hvac_action = f"Turn off heating in Room_{room_num}"
        elif hvac_status == "Cooling" and temperature < 20:
            hvac_action = f"Turn off cooling in Room_{room_num}"
        elif hvac_status == "Off" and (temperature < 18 or temperature > 26):
            hvac_action = f"Enable HVAC in Room_{room_num} for comfort"
        else:
            hvac_action = f"Maintain Room_{room_num} HVAC"

        if lighting_status != "Off" and light > 400:
            lighting_action = f"Turn off lights in Room_{room_num}"
        elif lighting_status == "Full" and light > 250:
            lighting_action = f"Dim lights in Room_{room_num}"
        else:
            lighting_action = f"Maintain Room_{room_num} lighting"

        rooms.append({
            "RoomID": f"Room_{room_num}",
            "Occupancy": round(random.uniform(0.0, 1.0), 2),
            "Temperature": temperature,
            "CO2": random.randint(400, 1200),
            "Light": light,
            "Humidity": round(random.uniform(20.0, 60.0), 1),
            "Power_kWh": round(random.uniform(0.2, 3.5), 2),
            "HVAC_Status": hvac_status,
            "Lighting_Status": lighting_status,
            "Auto_Lighting_Action": lighting_action,
            "Auto_HVAC_Action": hvac_action
        })
    return pd.DataFrame(rooms)

# Define tools for agents
@tool
def get_building_summary() -> str:
    """Return average environmental and occupancy statistics for the building."""
    return _get_building_summary_internal()




@tool
def get_weather_summary(city: str) -> str:
    """Return current weather conditions for a given city using Open-Meteo API."""
    return _get_weather_summary_internal(city)



@tool
def predict_occupancy_trend(room_id: Optional[str] = None) -> str:
    """Forecast future occupancy for a given room using ARIMA. Defaults to all rooms if not specified."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty:
            return "⚠️ No live room data available."

        room_ids = [room_id] if room_id else rooms_df["RoomID"].unique()
        forecast_results = []

        now = pd.Timestamp.now()

        for rid in room_ids:
            # Generate synthetic 48-point historical occupancy data (last 48 hours)
            history = [
                round(random.uniform(0.0, 1.0), 2)
                for _ in range(48)
            ]

            try:
                model = ARIMA(history, order=(2, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=6)  # next 6 hours
                forecast_values = [round(val, 2) for val in forecast.tolist()]

                avg = sum(forecast_values) / len(forecast_values)
                max_occ = max(forecast_values)
                min_occ = min(forecast_values)

                forecast_results.append(
                    f"📊 Occupancy forecast for {rid} (next 6 hours):\n"
                    f"• Avg: {avg:.2f}, Max: {max_occ:.2f}, Min: {min_occ:.2f}\n"
                    f"• Trend: {', '.join(map(str, forecast_values))}"
                )
            except:
                forecast_results.append(f"⚠️ Forecast model failed for {rid}.")

        return "\n\n".join(forecast_results)

    except Exception as e:
        return f"❌ Occupancy trend forecast failed: {e}"
    



@tool
def simulate_energy_reduction(room: dict) -> str:
    """Simulate energy reduction by optimizing HVAC and lighting settings based on model predictions."""
    try:
        # Since we don't have the actual model file, we'll simulate the prediction
        current_energy = room.get('Power_kWh', 2.0)
        
        # Calculate optimized energy based on room parameters
        optimized_energy = current_energy * 0.7  # Assume 30% reduction
        
        # Calculate savings
        reduction = current_energy - optimized_energy
        percent = (reduction / current_energy) * 100

        return (
            f"🔋 Current Energy Usage: {current_energy:.2f} kWh\n"
            f"⚙️ Optimized Energy Usage: {optimized_energy:.2f} kWh\n"
            f"📉 Estimated Reduction: {reduction:.2f} kWh ({percent:.1f}%)"
        )
    except Exception as e:
        return f"❌ Failed to simulate energy reduction: {e}"
    


@tool
def suggest_room_consolidation() -> str:
    """Suggest consolidation of underutilized rooms based on occupancy threshold."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty:
            return "⚠️ No live room data available."

        underused = rooms_df[rooms_df["Occupancy"] < 0.3]
        if underused.empty:
            return "✅ All rooms have adequate occupancy. No consolidation needed."

        suggestions = []
        for _, row in underused.iterrows():
            suggestions.append(
                f"🔄 Room {row['RoomID']} has low occupancy ({row['Occupancy']*100:.1f}%). Consider merging or closing."
            )
        return "\n".join(suggestions)
    except Exception as e:
        return f"❌ Error during consolidation suggestion: {e}"
    



@tool
def suggest_space_rezoning() -> str:
    """Cluster rooms into zones based on occupancy, temperature, CO2, light, humidity, and energy usage."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty:
            return "⚠️ No live room data available."

        features = ["Occupancy", "Temperature", "CO2", "Light", "Humidity", "Power_kWh"]
        data = rooms_df[features]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        rooms_df["Zone"] = kmeans.fit_predict(X_scaled)
        st.session_state["rooms_df"] = rooms_df  # update stored df

        # Format result
        cluster_counts = rooms_df["Zone"].value_counts().sort_index()
        summary = "\n".join([f"Zone {z}: {count} room(s)" for z, count in cluster_counts.items()])
        return f"🧠 Re-zoning complete using KMeans clustering:\n{summary}"
    except Exception as e:
        return f"❌ Error during re-zoning: {e}"
    




@tool
def explain_energy_forecast_llm(room_id: str) -> str:
    """LLM-based forecast summary for a room using current room and weather data."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty or room_id not in rooms_df["RoomID"].values:
            return f"⚠️ Room {room_id} not found in current data."

        room = rooms_df[rooms_df["RoomID"] == room_id].iloc[0].to_dict()
        weather = _get_weather_summary_internal(st.session_state.get("city_name", "New York"))

        prompt = (
            f"Given the following room and outside weather conditions, generate a detailed energy usage forecast.\n\n"
            f"Room: {room_id}\n"
            f"Occupancy: {room['Occupancy']}\n"
            f"Temperature: {room['Temperature']}°C\n"
            f"CO₂: {room['CO2']}ppm\n"
            f"Light: {room['Light']} Lux\n"
            f"Humidity: {room['Humidity']}%\n"
            f"Power: {room['Power_kWh']} kWh\n"
            f"HVAC: {room['HVAC_Status']}\n"
            f"Lighting: {room['Lighting_Status']}\n\n"
            f"Outside Weather:\n{weather}\n\n"
            f"Please explain:\n"
            f"1. Heating or cooling demand\n"
            f"2. Lighting adjustments and their effect\n"
            f"3. Occupancy impact\n"
            f"4. Forecasted energy usage (range) for the next few hours\n"
            f"5. Actionable recommendations"
        )


        client = OpenAI(api_key=st.session_state.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Energy forecast failed: {e}"




@tool
def recommend_layout_plan() -> str:
    """Suggest layout changes based on occupancy and usage patterns."""
    try:
        df = st.session_state.get("rooms_df")
        if df is None or df.empty:
            return "⚠️ No live room data available."

        underused = df[df["Occupancy"] < 0.2]
        overused = df[df["Occupancy"] > 0.75]
        suggestions = []

        for _, row in underused.iterrows():
            suggestions.append(
                f"🔄 Room {row['RoomID']} is underused (Occupancy {row['Occupancy']:.2f}). Consider repurposing it as a quiet zone, storage, or merging."
            )

        for _, row in overused.iterrows():
            suggestions.append(
                f"📈 Room {row['RoomID']} is heavily used (Occupancy {row['Occupancy']:.2f}). Consider creating more spaces like this or redistributing usage."
            )

        return "\n".join(suggestions) if suggestions else "✅ Current layout appears optimal."

    except Exception as e:
        return f"❌ Failed to generate layout plan: {e}"




@tool
def identify_multiuse_zones() -> str:
    """Identify rooms suitable for multi-functional use based on usage and environment."""
    try:
        df = st.session_state.get("rooms_df")
        if df is None or df.empty:
            return "⚠️ No room data available."

        flexible = df[
            (df["Occupancy"] > 0.3) &
            (df["Power_kWh"] < 2.5) &
            (df["Light"] > 200) &
            (df["Humidity"] < 50)
        ]

        if flexible.empty:
            return "No rooms currently suitable for multi-functional use."

        return "\n".join([
            f"🌀 Room {row['RoomID']} could serve multiple purposes (e.g., meeting + focus work)."
            for _, row in flexible.iterrows()
        ])

    except Exception as e:
        return f"❌ Error identifying flexible rooms: {e}"




@tool
def recommend_room_function_map() -> str:
    """
    Recommend a function for each room based on occupancy, power usage, and lighting.
    Logic:
    - High occupancy + high power → Meeting Room
    - Low occupancy + high light → Focus Pod
    - Low occupancy + low power → Storage/Flex Space
    """
    try:
        df = st.session_state.get("rooms_df")
        if df is None or df.empty:
            return "⚠️ No room data available."

        suggestions = []
        for _, row in df.iterrows():
            room_id = row["RoomID"]
            occ = row["Occupancy"]
            power = row["Power_kWh"]
            light = row["Light"]

            if occ > 0.6 and power > 2.5:
                function = "🧑‍💼 Meeting Room"
            elif occ < 0.3 and light > 300:
                function = "🔕 Focus Pod"
            elif occ < 0.3 and power < 1.5:
                function = "📦 Storage or Flex Space"
            else:
                function = "🔄 Multi-use Room"

            suggestions.append(f"{room_id}: Recommended use → {function}")

        return "\n".join(suggestions)

    except Exception as e:
        return f"❌ Failed to generate room function map: {e}"

tool_function_map = {
    "Get Building Summary": get_building_summary,
    "Get Weather Summary": get_weather_summary,
    "Simulate Energy Reduction": simulate_energy_reduction,
    "Explain Energy Forecast": explain_energy_forecast_llm,
    "Predict Occupancy Trend": predict_occupancy_trend,
    "Suggest Room Consolidation": suggest_room_consolidation,
    "Suggest Space Rezoning": suggest_space_rezoning,
    "Recommend Layout Plan": recommend_layout_plan,
    "Identify Multiuse Zones": identify_multiuse_zones,
    "Room Function Mapping": recommend_room_function_map
}

def space_optimization_agent_ui():
    st.header("Space Optimization Assistant")
    st.caption("**Goal:** Optimize room allocation using occupancy and environment data")
    
    # Generate live room data if not already in session state
    if (
    "rooms_df" not in st.session_state
    or st.session_state.rooms_df is None
    or st.session_state.get("num_rooms") != st.session_state.num_rooms
):
        st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)

    
    rooms_df = st.session_state.rooms_df
    
    st.subheader("Building Overview")
    try:
        building_summary = _get_building_summary_internal()
        st.info(building_summary)
    except Exception as e:
        st.info("Building summary not available. Using generated room data.")
    
    st.subheader("Room Data")
    st.dataframe(rooms_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Room Data"):
            st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)
            st.rerun()
    with col2:
        if st.button("Run Space Optimization Analysis"):
            if not st.session_state.openai_api_key:
                st.error("Please enter a valid OpenAI API key to generate recommendations.")
            else:
                space_result = run_space_optimization(st.session_state.rooms_df)
                if space_result:
                    st.session_state.results["space_optimization"] = space_result
                    
    
    # Display visualizations
    st.subheader("Room Occupancy Visualization")
    fig = {
        'data': [
            {
                'x': rooms_df['RoomID'],
                'y': rooms_df['Occupancy'],
                'type': 'bar',
                'marker': {'color': rooms_df['Occupancy'].apply(lambda x: 'red' if x > 0.7 else 'orange' if x > 0.4 else 'green')}
            }
        ],
        'layout': {
            'title': 'Room Occupancy Levels',
            'xaxis': {'title': 'Room ID'},
            'yaxis': {'title': 'Occupancy Rate (0-1)'}
        }
    }
    st.plotly_chart(fig, use_container_width=True)
    

    
    chat_ui()

def layout_recommendation_agent_ui():
    st.header("Layout Planner")
    st.caption("**Goal:** Analyze room data and suggest long-term layout changes for flexibility and efficiency")
    
    # Generate live room data if not already in session state
    if (
    "rooms_df" not in st.session_state or
    st.session_state.rooms_df is None or
    st.session_state.get("num_rooms") != st.session_state.num_rooms
):
        st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)

    
    rooms_df = st.session_state.rooms_df
    
    st.subheader("Current Layout Data")
    st.dataframe(rooms_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Layout Data"):
            st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)
            st.rerun()
    with col2:
        if st.button("Generate Layout Recommendations"):
            if not st.session_state.openai_api_key:
                st.error("Please enter a valid OpenAI API key to generate recommendations.")
            else:
                layout_result = run_layout_recommendation(st.session_state.rooms_df)
                if layout_result:
                    st.session_state.results["layout_recommendation"] = layout_result

    # Display visualizations
    st.subheader("Room Usage Analysis")
    
    # Power usage by room
    fig = {
        'data': [
            {
                'x': rooms_df['RoomID'],
                'y': rooms_df['Power_kWh'],
                'type': 'bar',
                'name': 'Power Usage (kWh)',
                'marker': {'color': 'blue'}
            }
        ],
        'layout': {
            'title': 'Power Usage by Room',
            'xaxis': {'title': 'Room ID'},
            'yaxis': {'title': 'Power (kWh)'}
        }
    }
    st.plotly_chart(fig, use_container_width=True)
    
    # Room metrics comparison
    fig2 = {
        'data': [
            {
                'type': 'scatterpolar',
                'r': [row['Occupancy'], row['Temperature']/30, row['CO2']/1200, row['Light']/600, row['Humidity']/100, row['Power_kWh']/4],
                'theta': ['Occupancy', 'Temperature', 'CO2', 'Light', 'Humidity', 'Power'],
                'fill': 'toself',
                'name': row['RoomID']
            } for _, row in rooms_df.iterrows()
        ],
        'layout': {
            'title': 'Room Metrics Comparison',
            'polar': {
                'radialaxis': {'visible': True, 'range': [0, 1]}
            }
        }
    }
    st.plotly_chart(fig2, use_container_width=True)
    
    chat_ui()

def main():
    industry = st.session_state.get('industry', '')
    use_case = st.session_state.get('use_case', '')

    if industry == "Energy Management in Buildings":
        # ✅ Show default system (tables + agent UI)
        show_agent_tools_table(use_case)
        show_agent_llm_table(use_case)

        if use_case == "Thermal Comfort Analysis":
            thermal_comfort_agent_ui()
        elif use_case == "Energy Optimization":
            energy_optimization_agent_ui()
        elif use_case == "Space Optimization":
            space_optimization_agent_ui()
        elif use_case == "Layout Planning":
            layout_recommendation_agent_ui()
    
    elif industry == "Public Sector":
        # 🚧 Show in-progress message
        st.warning("🚧 The 'Public Sector' use case is under development. Stay tuned!")

    else:
        # 📭 Default selection prompt
        st.info("")


if __name__ == "__main__":
    main()
