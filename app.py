

import sys
import os

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
from pythermalcomfort.models import pmv_ppd_iso, utci
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from datetime import datetime


# Set page config
st.set_page_config(page_title="Building Intelligence System (CrewAI)", layout="wide")

# Add this code to make sidebar blue and text bolder
st.markdown("""
<style>
[data-testid=stSidebar] {
  background-color: #60B5FF;
}
/* Make all text in sidebar bold */
[data-testid=stSidebar] p, 
[data-testid=stSidebar] .st-bq, 
[data-testid=stSidebar] .st-ae, 
[data-testid=stSidebar] .st-af,
[data-testid=stSidebar] span,
[data-testid=stSidebar] label,
[data-testid=stSidebar] div {
  font-weight: bold !important;
}
/* Make sidebar headers extra bold */
[data-testid=stSidebar] h1,
[data-testid=stSidebar] h2,
[data-testid=stSidebar] h3,
[data-testid=stSidebar] h4 {
  font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state for storing data between agents
if 'thermal_data' not in st.session_state:
    st.session_state.thermal_data = None
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'results' not in st.session_state:
    st.session_state.results = {}
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for API key and agent selection
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API key:",
                                 type="password",
                                 value=st.session_state.openai_api_key)
    st.session_state.openai_api_key = openai_api_key

    # Set the API key for CrewAI to use
    if openai_api_key and openai_api_key.startswith('sk-'):
        os.environ["OPENAI_API_KEY"] = openai_api_key

    st.header("🤖 Agent Selection")
    agent_selection = st.selectbox("Select Agent:",
                              ["Thermal Comfort Agent", "Energy Optimization Agent"])

# Helper functions
@st.cache_resource
def get_openai_client(api_key):
    if api_key and api_key.startswith('sk-'):
        return OpenAI(api_key=api_key)
    return None

def get_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo API (no auth required)"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['current']
    return None

@st.cache_data
def get_sample_data():
    """Generate sample energy data template"""
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
    """Calculate thermal comfort metrics"""
    pmv_result = pmv_ppd_iso(tdb=inputs['tdb'], tr=inputs['tr'], vr=inputs['vr'],
                           rh=inputs['rh'], met=inputs['met'], clo=inputs['clo'])
    utci_result = utci(tdb=inputs['tdb'], tr=inputs['tr'], v=inputs['vr'], rh=inputs['rh'])

    return {
        'pmv': round(pmv_result.pmv, 2),
        'ppd': round(pmv_result.ppd, 1),
        'utci': round(utci_result.utci, 1),
        'utci_category': utci_result.stress_category
    }

# Chat function to process user queries
def process_chat_query(query):
    if not st.session_state.openai_api_key:
        return "Please enter a valid OpenAI API key to use the chat feature."

    client = get_openai_client(st.session_state.openai_api_key)
    if not client:
        return "Invalid OpenAI API key. Please check your API key and try again."

    # Prepare context from available reports
    context = "You are a building intelligence assistant that helps with thermal comfort and energy optimization questions."

    if st.session_state.thermal_data:
        context += f"\n\nThermal data available: Building type: {st.session_state.thermal_data['building_type']}, "
        context += f"Temperature: {st.session_state.thermal_data['tdb']}°C, "
        context += f"Humidity: {st.session_state.thermal_data['rh']}%, "
        context += f"PMV: {st.session_state.thermal_data['pmv']}, "
        context += f"PPD: {st.session_state.thermal_data['ppd']}%, "
        context += f"UTCI: {st.session_state.thermal_data['utci']}°C ({st.session_state.thermal_data['utci_category']})"

    if "thermal_analysis" in st.session_state.results:
        # Convert CrewOutput to string
        thermal_analysis = str(st.session_state.results["thermal_analysis"])
        context += "\n\nThermal Analysis Report:\n" + thermal_analysis

    if "energy_optimization" in st.session_state.results:
        # Convert CrewOutput to string
        energy_optimization = str(st.session_state.results["energy_optimization"])
        context += "\n\nEnergy Optimization Report:\n" + energy_optimization

    try:
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

# Define CrewAI agents
def create_thermal_agent():
    return Agent(
        role="Thermal Comfort Analyst",
        goal="Analyze indoor environmental parameters and generate technical thermal comfort reports",
        backstory="Expert in building science and ISO/ASHRAE thermal comfort standards",
        verbose=True,
        allow_delegation=False
    )

def create_energy_agent():
    return Agent(
        role="Energy Optimization Engineer",
        goal="Recommend energy-saving actions while maintaining thermal comfort",
        backstory="Specialist in building energy systems and cost-benefit analysis",
        verbose=True,
        allow_delegation=False
    )

# Define tasks
def create_thermal_analysis_task(inputs, agent):
    return Task(
        description=f"Calculate thermal comfort metrics and generate a report for a {inputs['building_type']} building with: Air temp {inputs['tdb']}°C, Mean radiant temp {inputs['tr']}°C, Relative humidity {inputs['rh']}%, Air velocity {inputs['vr']} m/s, Activity level {inputs['met']} met, Clothing insulation {inputs['clo']} clo",
        expected_output="Thermal comfort metrics with ASHRAE compliance analysis and a detailed report",
        agent=agent
    )

def create_energy_optimization_task(thermal_data, energy_inputs, agent):
    # Include thermal data directly in the task description
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

# Run thermal comfort analysis
def run_thermal_analysis(env_params):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate the report.")
        return None

    with st.spinner("Running thermal comfort analysis with CrewAI..."):
        # Create agent
        thermal_agent = create_thermal_agent()

        # Create task
        thermal_task = create_thermal_analysis_task(env_params, thermal_agent)

        # Create crew
        thermal_crew = Crew(
            agents=[thermal_agent],
            tasks=[thermal_task],
            verbose=True,
            process=Process.sequential
        )

        # Run analysis
        try:
            thermal_result = thermal_crew.kickoff()
            return thermal_result
        except Exception as e:
            st.error(f"Error running thermal analysis: {str(e)}")
            return None

# Run energy optimization
def run_energy_optimization(thermal_data, energy_inputs):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None

    with st.spinner("Running energy optimization with CrewAI..."):
        # Create agent
        energy_agent = create_energy_agent()

        # Create task
        energy_task = create_energy_optimization_task(thermal_data, energy_inputs, energy_agent)

        # Create crew
        energy_crew = Crew(
            agents=[energy_agent],
            tasks=[energy_task],
            verbose=True,
            process=Process.sequential
        )

        # Run optimization
        try:
            energy_result = energy_crew.kickoff()
            return energy_result
        except Exception as e:
            st.error(f"Error running energy optimization: {str(e)}")
            return None

# Chat UI component that can be added to any page
def chat_ui():
    st.divider()
    st.subheader("💬 Building Intelligence Chat")

    # Display available data summary if any
    if st.session_state.thermal_data or st.session_state.results:
        available_data = []
        if st.session_state.thermal_data:
            available_data.append("Thermal comfort data")
        if "thermal_analysis" in st.session_state.results:
            available_data.append("Thermal analysis report")
        if "energy_optimization" in st.session_state.results:
            available_data.append("Energy optimization recommendations")

        if available_data:
            st.caption(f"Available context: {', '.join(available_data)}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about building comfort or energy..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_chat_query(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# THERMAL COMFORT AGENT
def thermal_comfort_agent_ui():
    st.header("Thermal Comfort Analyst")
    st.caption("**Goal:** Analyze indoor environmental parameters and generate technical thermal comfort reports")

    # Location input
    st.subheader("📍 Enter Location Coordinates")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.0, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.0, format="%.6f")

    # Map display
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))

    # Weather data
    weather_data = get_weather_data(lat, lon)
    if weather_data:
        st.success(f"Fetched weather: {weather_data['temperature_2m']}°C, RH {weather_data['relative_humidity_2m']}%, Wind {weather_data['wind_speed_10m']} m/s")

    # Environmental parameters - moved from sidebar to main page
    st.header("⚙️ Environmental Parameters")
    col1, col2 = st.columns(2)

    with col1:
        building_type = st.selectbox("Building Type", ["Office", "Residential", "Educational"])
        season = st.selectbox("Season", ["Summer", "Winter"])

        tdb = st.number_input("Air Temperature (°C)",
                             value=float(weather_data['temperature_2m']))
        tr = st.number_input("Mean Radiant Temperature (°C)", value=tdb)
        rh = st.slider("Relative Humidity (%)", 0, 100,
                     value=int(weather_data['relative_humidity_2m']))

    with col2:
        # Clamp wind speed to valid range for PMV
        min_vr = 0.0
        max_vr = 1.0
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

            # Calculate metrics locally
            metrics = calculate_thermal_metrics(inputs)

            # Store data for energy optimization agent
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

            # Run CrewAI thermal analysis
            thermal_result = run_thermal_analysis(inputs)

            if thermal_result:
                st.session_state.results["thermal_analysis"] = thermal_result

                # Display results
                st.subheader("📊 Thermal Comfort Analysis Report")
                st.markdown(str(thermal_result))

                st.subheader("🌡️ Key Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("PMV", f"{metrics['pmv']}", "Neutral (0)" if -0.5 < metrics['pmv'] < 0.5 else "Needs Adjustment")
                col2.metric("PPD", f"{metrics['ppd']}%", help="Predicted Percentage Dissatisfied")
                col3.metric("UTCI", f"{metrics['utci']}°C", metrics['utci_category'])

                st.success("Thermal comfort analysis complete!")

    # Add chat UI at the bottom of the page
    chat_ui()

# ENERGY OPTIMIZATION AGENT
def energy_optimization_agent_ui():
    st.header("Energy Optimization Engineer")
    st.caption("**Goal:** Recommend energy-saving actions while maintaining thermal comfort")

    # Display thermal comfort data summary if available
    if st.session_state.thermal_data is not None:
        st.subheader("📋 Thermal Comfort Analysis Summary")
        thermal_data = st.session_state.thermal_data

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Building Type", thermal_data['building_type'])
        col2.metric("Indoor Temperature", f"{thermal_data['tdb']}°C")
        col3.metric("PMV", f"{thermal_data['pmv']}")
        col4.metric("UTCI", f"{thermal_data['utci']}°C", thermal_data['utci_category'])
    else:
        st.info("No thermal comfort data available. You can still run energy optimization without it.")

    # CSV Upload Section
    st.subheader("📤 Upload Energy Data")

    # Provide sample template for download
    sample_df = get_sample_data()
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_df.to_csv(index=False),
        file_name="energy_data_template.csv",
        mime="text/csv"
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file with energy data", type="csv")

    # Default values
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

    # Update with thermal data if available
    if st.session_state.thermal_data:
        current_temperature = float(st.session_state.thermal_data['tdb'])
        humidity_level = float(st.session_state.thermal_data['rh'])
        hvac_status = "On - Cooling" if current_temperature > 24 else "On - Heating"

    # Process uploaded file if available
    if uploaded_file is not None:
        try:
            energy_data = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded data from {uploaded_file.name}")
            st.dataframe(energy_data)

            # Update values from CSV if columns exist
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

    # Energy system inputs - moved from sidebar to main page
    st.subheader("🏢 Building Energy System Status")

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

    # AI Recommendations based on thermal comfort and energy data
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

            # Run CrewAI energy optimization
            energy_result = run_energy_optimization(st.session_state.thermal_data, energy_inputs)

            if energy_result:
                st.session_state.results["energy_optimization"] = energy_result

                # Display the recommendations
                st.subheader("🤖 AI Energy Optimization Recommendations")
                st.markdown(str(energy_result))

    # Add chat UI at the bottom of the page
    chat_ui()

# Main app logic
def main():
    if agent_selection == "Thermal Comfort Agent":
        thermal_comfort_agent_ui()
    else:
        energy_optimization_agent_ui()

if __name__ == "__main__":
    main()
