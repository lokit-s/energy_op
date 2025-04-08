
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pythermalcomfort.models import pmv_ppd_iso, utci

st.set_page_config(page_title="Thermal Comfort & Solar Calculator", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a tool:", ["Thermal Comfort Calculator", "Solar PV Calculator"])

# Thermal Comfort Calculator Page
if page == "Thermal Comfort Calculator":
    st.title("Thermal Comfort Calculator")
    
    st.markdown("""
    This tool calculates thermal comfort indices using the **PMV/PPD** model from ISO 7730 
    and the **UTCI** (Universal Thermal Climate Index).
    """)

    # Create two columns for inputs

    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader("Environmental Parameters")
        tdb = st.slider("Air temperature (°C)", 10.0, 40.0, 25.0, 0.1)
        tr = st.slider("Mean radiant temperature (°C)", 10.0, 40.0, 25.0, 0.1)
        vr = st.slider("Relative air velocity (m/s)", 0.0, 2.0, 0.1, 0.05)
        rh = st.slider("Relative humidity (%)", 0, 100, 50, 1)

    with col2:
        
        # Using fixed default values instead of inputs
        met = 1.4  # Default metabolic rate (office work)

        
        clo = 0.5  # Default clothing (Trousers and T-shirt)


    # Calculate button
    if st.button("Calculate Thermal Comfort Indices"):
        # Calculate PMV/PPD
        try:
            pmv_result = pmv_ppd_iso(tdb=tdb, tr=tr, vr=vr, rh=rh, met=1.4, clo=0.5, model='7730-2005')
            
            # Calculate UTCI
            utci_result = utci(tdb=tdb, tr=tr, v=vr, rh=rh)
            
        # Rest of the calculation code remains the same...
            # Display results in expanded sections
            with st.expander("PMV/PPD Results (ISO 7730)", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("PMV (Predicted Mean Vote)", f"{pmv_result.pmv:.2f}")
                    
                    # Interpret PMV
                    pmv_interp = ""
                    if pmv_result.pmv < -3:
                        pmv_interp = "Very Cold"
                    elif pmv_result.pmv < -2:
                        pmv_interp = "Cold"
                    elif pmv_result.pmv < -1:
                        pmv_interp = "Slightly Cool"
                    elif pmv_result.pmv < 1:
                        pmv_interp = "Neutral"
                    elif pmv_result.pmv < 2:
                        pmv_interp = "Slightly Warm"
                    elif pmv_result.pmv < 3:
                        pmv_interp = "Warm"
                    else:
                        pmv_interp = "Hot"
                    
                    st.write(f"Interpretation: **{pmv_interp}**")
                
                with col2:
                    st.metric("PPD (Predicted Percentage Dissatisfied)", f"{pmv_result.ppd:.1f}%")
                    
                    # PPD gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = pmv_result.ppd,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Dissatisfaction Level"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 10], 'color': "green"},
                                {'range': [10, 20], 'color': "yellow"},
                                {'range': [20, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': pmv_result.ppd
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("UTCI Results", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("UTCI", f"{utci_result.utci:.2f} °C")
                    st.write(f"Thermal stress category: **{utci_result.stress_category}**")
                
                with col2:
                    # UTCI stress categories visualization
                    categories = ["Extreme heat stress", "Very strong heat stress", "Strong heat stress", 
                                "Moderate heat stress", "No thermal stress", "Slight cold stress", 
                                "Moderate cold stress", "Strong cold stress", "Very strong cold stress", 
                                "Extreme cold stress"]
                    
                    # Determine which category we're in
                    highlight_cat = utci_result.stress_category
                    
                    # Create color array (grey for all except our category)
                    colors = ["lightgrey" if cat != highlight_cat else "red" for cat in categories]
                    
                    fig = go.Figure(go.Bar(
                        x=[1]*len(categories),
                        y=categories,
                        orientation='h',
                        marker_color=colors
                    ))
                    fig.update_layout(
                        title="UTCI Stress Category",
                        showlegend=False,
                        xaxis={'showticklabels': False},
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on results
            st.subheader("Recommendations")
            if pmv_result.ppd > 20:
                st.warning("The predicted percentage dissatisfied is high. Consider adjusting the following parameters:")
                
                if pmv_result.pmv > 0.5:
                    st.write("- Decrease room temperature")
                    st.write("- Increase air velocity")
                    st.write("- Reduce clothing insulation")
                elif pmv_result.pmv < -0.5:
                    st.write("- Increase room temperature")
                    st.write("- Decrease air velocity")
                    st.write("- Increase clothing insulation")
            else:
                st.success("The thermal environment is within comfortable ranges!")
                
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")

# Solar PV Calculator Page
elif page == "Solar PV Calculator":
    st.title("Solar PV Calculator")
    
    st.markdown("""
    This calculator uses the NREL PVWatts API to estimate solar electricity production 
    based on location and system parameters.
    """)
    
    # Initialize with default values
    if 'api_key' not in st.session_state:
        st.session_state.api_key = "k6pwYezldzUqdE8rJFuvxevkHs1IKbGQtNkn2iR7"  # Default API key
    
    # Two column layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location")
        location_type = st.radio("Input type", ["Coordinates", "Address"])
        
        if location_type == "Coordinates":
            lat = st.number_input("Latitude", -90.0, 90.0, 39.742, 0.001)
            lon = st.number_input("Longitude", -180.0, 180.0, -105.18, 0.001)
        else:
            address = st.text_input("Enter address", "Denver, CO")
            # Geocoding would be implemented here in a real application
            # For now, we'll use default coordinates
            lat, lon = 39.742, -105.18
    
    with col2:
        st.subheader("System Parameters")
        system_capacity = st.number_input("System Capacity (kW)", 0.5, 100.0, 4.0, 0.1)
        
        module_type = st.selectbox(
            "Module Type",
            ["Standard", "Premium", "Thin film"],
            index=0
        )
        # Map to PVWatts API values
        module_type_map = {"Standard": 0, "Premium": 1, "Thin film": 2}
        
        array_type = st.selectbox(
            "Array Type",
            ["Fixed (open rack)", "Fixed (roof mount)", "1-axis tracking", "2-axis tracking"],
            index=1
        )
        # Map to PVWatts API values
        array_type_map = {
            "Fixed (open rack)": 0, 
            "Fixed (roof mount)": 1, 
            "1-axis tracking": 2, 
            "2-axis tracking": 3
        }
        
        losses = st.slider("System Losses (%)", 0, 50, 14, 1)
    
    # Advanced parameters in expander
    with st.expander("Advanced Parameters"):
        azimuth = st.slider("Azimuth (degrees, 180° = South in Northern Hemisphere)", 0, 359, 180, 1)
        tilt = st.slider("Tilt (degrees)", 0, 90, 20, 1)
        dc_ac_ratio = st.number_input("DC to AC Ratio", 1.0, 2.0, 1.2, 0.1)
        gcr = st.number_input("Ground Coverage Ratio", 0.1, 0.9, 0.4, 0.01)
        inv_eff = st.number_input("Inverter Efficiency (%)", 90.0, 99.5, 96.0, 0.1)
    
    # Calculate button
    if st.button("Calculate Solar Production"):
        with st.spinner("Fetching solar data..."):
            try:
                # Prepare API request
                url = "https://developer.nrel.gov/api/pvwatts/v6.json"
                
                params = {
                    "api_key": st.session_state.api_key,
                    "lat": lat,
                    "lon": lon,
                    "system_capacity": system_capacity,
                    "azimuth": azimuth,
                    "tilt": tilt,
                    "array_type": array_type_map[array_type],
                    "module_type": module_type_map[module_type],
                    "losses": losses,
                    "dc_ac_ratio": dc_ac_ratio,
                    "gcr": gcr,
                    "inv_eff": inv_eff
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display results
                    st.success("Calculation successful!")
                    
                    # Main metrics
                    annual_output = data['outputs']['ac_annual']
                    st.metric("Annual AC Energy Output", f"{annual_output:,.0f} kWh")
                    
                    st.subheader("Financial Estimates")
                    avg_cost_per_kwh = 0.12  # Average US electricity cost
                    annual_savings = annual_output * avg_cost_per_kwh
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Estimated Annual Savings", f"${annual_savings:,.2f}")
                    col2.metric("Estimated System Cost", f"${system_capacity * 3000:,.2f}", 
                              help="Estimated at $3,000 per kW installed")
                    
                    payback_years = (system_capacity * 3000) / annual_savings
                    col3.metric("Simple Payback Period", f"{payback_years:.1f} years")
                    
                    # Monthly data visualization
                    if 'ac_monthly' in data['outputs']:
                        monthly_output = data['outputs']['ac_monthly']
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        # Create DataFrame for visualization
                        df = pd.DataFrame({
                            'Month': months,
                            'Energy (kWh)': monthly_output
                        })
                        
                        # Create visualizations
                        st.subheader("Monthly Production")
                        
                        # Bar chart
                        fig = px.bar(
                            df, 
                            x='Month', 
                            y='Energy (kWh)',
                            title="Monthly Solar Energy Production",
                            labels={'Energy (kWh)': 'Energy Production (kWh)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display tabular data
                        with st.expander("Show monthly data table"):
                            st.dataframe(df, hide_index=True)
                        
                        # System performance metrics
                        st.subheader("System Performance Metrics")
                        
                        metric_cols = st.columns(3)
                        metric_cols[0].metric("Capacity Factor", f"{data['outputs']['capacity_factor']:.1f}%")
                        
                        # Calculate approximate roof area required
                        if array_type == "Fixed (roof mount)":
                            roof_area = system_capacity * (1 / 0.17) * 10.76  # m² to ft²
                            metric_cols[1].metric("Estimated Roof Area Required", f"{roof_area:.0f} ft²")
                        
                        # Total solar radiation
                        avg_solar_radiation = sum(data['outputs']['poa_monthly']) / 12
                        metric_cols[2].metric("Avg Monthly Solar Radiation", f"{avg_solar_radiation:.0f} kWh/m²")
                        
                else:
                    st.error(f"Error with PVWatts API: {response.status_code}")
                    st.write("Error details:", response.text)
                    
            except Exception as e:
                st.error(f"An error occurred during calculation: {e}")
                st.write("Please check your inputs and try again.")

# Add footer
st.markdown("---")
st.markdown("### About This Tool")
st.markdown("""
This tool combines thermal comfort assessment with solar PV system analysis, helping users make informed decisions 
about building comfort and renewable energy potential. The thermal comfort calculations use the 
pythermalcomfort library, and solar analysis uses NREL's PVWatts API.
""")
