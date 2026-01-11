import os
import random
from sklearn.base import BaseEstimator
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import dice_ml

from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

OUTPUT_DIR = "outputs/"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models/")
DATA_SPLITS = os.path.join(OUTPUT_DIR, "splits/")

DATA_DIR = "data/"
DATA_FILE = "clean_data.csv"
INPUT_PATH  = os.path.join(DATA_DIR, DATA_FILE)

GLOBAL_SEED = 63

def set_global_seed():
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

set_global_seed()

st.set_page_config(page_title="Building Energy XAI Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(INPUT_PATH)
    return df

@st.cache_data
def load_data_splits():
    X_train = pd.read_csv(os.path.join(DATA_SPLITS,'X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_SPLITS, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_SPLITS, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(DATA_SPLITS, 'y_test.csv'))
    return X_train, X_test, y_train, y_test

@st.cache_data
def load_data_combined_loads():
    X_train, X_test, y_train, y_test = load_data_splits()

    y_train_combined = y_train.copy()
    y_test_combined = y_test.copy()

    y_train_combined['Combined Load'] = y_train_combined['Heating Load'] + y_train_combined['Cooling Load']
    y_train_combined.drop('Heating Load', axis=1, inplace=True)
    y_train_combined.drop('Cooling Load', axis=1, inplace=True)

    y_test_combined['Combined Load'] = y_test_combined['Heating Load'] + y_test_combined['Cooling Load']
    y_test_combined.drop('Heating Load', axis=1, inplace=True)
    y_test_combined.drop('Cooling Load', axis=1, inplace=True)

    return X_train, X_test, y_train_combined, y_test_combined

@st.cache_resource
def load_model():
    model_study = joblib.load(os.path.join(MODEL_DIR, "catboost_regressor.joblib"))
    best_params = model_study.best_params_
    best_params = {k.removeprefix("estimator__"): v for k, v in best_params.items()}

    best_model = CatBoostRegressor(**best_params, logging_level='Silent')
    wrapped_model = MultiOutputRegressor(estimator=best_model)

    X_train, X_test, y_train, y_test = load_data_splits()
    wrapped_model.fit(X_train, y_train)

    return wrapped_model

@st.cache_resource
def load_combined_model():
    # Load your trained model for combined loads
    model_study = joblib.load(os.path.join(MODEL_DIR, "catboost_regressor.joblib"))
    best_params = model_study.best_params_
    best_params = {k.removeprefix("estimator__"): v for k, v in best_params.items()}
    single_output_model = CatBoostRegressor(**best_params, logging_level='Silent')

    X_train, X_test, y_train, y_test = load_data_combined_loads()
    single_output_model.fit(X_train, y_train)

    return single_output_model

class MultiTargetWrapper(BaseEstimator):
    def __init__(self, model, heating_max, cooling_max):
        self.model = model
        self.y1_max = heating_max 
        self.y2_max = cooling_max

    def predict(self, X):
        preds = self.model.predict(X)
        y1 = preds[:, 0]
        y2 = preds[:, 1]
        
        y1_norm = y1 / self.y1_max
        y2_norm = y2 / self.y2_max
        
        composite_score = (y1_norm + y2_norm) / 2
        
        return composite_score

st.title("Building Energy Efficiency - XAI Dashboard")
st.markdown("""
This dashboard helps architects, end users, and regulators understand and optimize building energy consumption
using explainable AI techniques.
""")

st.sidebar.header("Select Your Role")
role = st.sidebar.selectbox(
    "I am a(an)...",
    ["Architect", "Building Owner", "Regulator"]
)

st.sidebar.markdown(f"**Dashboard customized for: {role}**")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Building Energy Predictor", 
    "Feature Importance (SHAP)", 
    "What-If Analysis (Counterfactuals)",
    "Portfolio Analysis"
])

# ============ TAB 1: BUILDING ENERGY PREDICTOR ============
with tab1:
    st.header("Building Energy Predictor")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Building Specifications")

        roof_area = st.slider("Roof Area", 100, 250, 150, 10)
        surface_area = st.slider("Surface Area", 500, 850, 650, 10)
        wall_area = st.slider("Wall Area", 200, 420, 300, 10)
        glazing_area = st.select_slider("Glazing Area Ratio", 
                                         options=[0.0, 0.1, 0.25, 0.4], 
                                         value=0.25)
        glazing_area_distribution = st.selectbox("Glazing Area Distribution", 
                                                 options=[0, 1, 2, 3, 4],
                                                 index=2)
        
        # Predict button
        if st.button("Predict Energy Consumption", type="primary"):
            model = load_model()
            input_data = pd.DataFrame({
                'Roof Area': [roof_area],
                'Surface Area': [surface_area],
                'Wall Area': [wall_area],
                'Glazing Area': [glazing_area],
                'Glazing Area Distribution': [glazing_area_distribution]
            })
            prediction = model.predict(input_data)
            heating_load, cooling_load = prediction[0]
            st.success(f"Predicted Heating Load: {heating_load:.2f}")
            st.success(f"Predicted Cooling Load: {cooling_load:.2f}")

            # Store in session state
            st.session_state['current_prediction'] = prediction
            st.session_state['current_input'] = input_data

    with col2:
        st.subheader("Energy Prediction Results")

        if 'current_prediction' in st.session_state:
            pred = st.session_state['current_prediction']
            total_energy = pred[0][0] + pred[0][1]
            
            # Display results with metrics
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Heating Load", f"{pred[0][0]:.2f}", delta=None)

            with col_b:
                st.metric("Cooling Load", f"{pred[0][1]:.2f}", delta=None)

            with col_c:
                st.metric("Total Load", f"{(pred[0][0] + pred[0][1]):.2f}", delta=None)

            # Benchmark comparison
            st.subheader("Benchmark Comparison")
            df = load_data()
            avg_total = df[['Heating Load', 'Cooling Load']].sum(axis=1).mean()

            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Your Building', 'Average Building']
            values = [total_energy, avg_total]

            ax.barh(categories, values, color=['skyblue', 'lightgreen'])
            ax.set_xlabel("Total Energy Load")
            ax.set_title("Energy Performance Comparison")
            st.pyplot(fig)
        else:
            st.info("Please input building specifications and click 'Predict Energy Consumption' to see results.")

# ============ TAB 2: SHAP ANALYSIS ============
with tab2:
    st.header("Feature Importance Analysis (SHAP)")
    # Role-specific guidance
    if role == "Architect":
        st.info("**For Architects**: This shows which design features have the biggest impact on energy consumption. Use this to prioritize design decisions.")
    elif role == "Building Owner":
        st.info("**For Building Owners**: This shows why your building's energy consumption is what it is, helping you understand potential savings opportunities.")
    elif role == "Regulator":
        st.info("**For Regulators**: This analysis helps identify key factors influencing building energy use across portfolios, guiding policy decisions.")

    # Load model and compute SHAP
    if st.button("Generate Feature Importance Explanations"):
        with st.spinner("Computing SHAP values..."):
            model = load_model()
            df = load_data()
            X = df.iloc[:, :-2]

            heating_model = model.estimators_[0]
            cooling_model = model.estimators_[1]

            heating_explainer = shap.TreeExplainer(heating_model)
            cooling_explainer = shap.TreeExplainer(cooling_model)

            heating_shap_values = heating_explainer.shap_values(X)
            cooling_shap_values = cooling_explainer.shap_values(X)

            # Store in session state
            st.session_state['heating_shap_values'] = heating_shap_values
            st.session_state['cooling_shap_values'] = cooling_shap_values
            st.session_state['shap_data'] = X

    if 'heating_shap_values' in st.session_state:
        heating_shap_values = st.session_state['heating_shap_values']
        cooling_shap_values = st.session_state['cooling_shap_values']
        X_shap = st.session_state['shap_data']

        # Choose output
        output_choice = st.radio("Analyze:", ["Heating Load", "Cooling Load"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Global Feature Importance")

            if output_choice == "Heating Load":
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(heating_shap_values, X_shap, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.clf()
            elif output_choice == "Cooling Load":
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(cooling_shap_values, X_shap, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.clf()
        
        with col2:
            st.subheader("Feature Impact Distribution")

            if output_choice == "Heating Load":
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(heating_shap_values, X_shap, show=False)
                st.pyplot(fig)
                plt.clf()
            elif output_choice == "Cooling Load":
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(cooling_shap_values, X_shap, show=False)
                st.pyplot(fig)
                plt.clf()

        # Local explanation for current building
        if 'current_input' in st.session_state:
            st.subheader("Your Building's Explanation")

            # Choose output
            output_choice_local = st.radio("Explain for:", ["Heating Load", "Cooling Load", "Both"], key="local_explain_choice")

            X_current = st.session_state['current_input']
            model = load_model()

            heating_model = model.estimators_[0]
            cooling_model = model.estimators_[1]

            heating_explainer = shap.TreeExplainer(heating_model)
            cooling_explainer = shap.TreeExplainer(cooling_model)

            heating_shap_current = heating_explainer.shap_values(X_current)
            cooling_shap_current = cooling_explainer.shap_values(X_current)

            if output_choice_local == "Heating Load":
                st.subheader("Heating Load Explanation")
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap.Explanation(values=heating_shap_current[0],
                                                      base_values=heating_explainer.expected_value,
                                                      data=X_current.iloc[0]), show=False)
                st.pyplot(fig)
                plt.clf()
            
            elif output_choice_local == "Cooling Load":
                st.subheader("Cooling Load Explanation")
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap.Explanation(values=cooling_shap_current[0],
                                                      base_values=cooling_explainer.expected_value,
                                                      data=X_current.iloc[0]), show=False)
                st.pyplot(fig)
                plt.clf()
            
            elif output_choice_local == "Both":
                # Show two waterfall plots side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Heating Load Explanation")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap.Explanation(values=heating_shap_current[0],
                                                        base_values=heating_explainer.expected_value,
                                                        data=X_current.iloc[0]), show=False)
                    st.pyplot(fig)
                    plt.clf()

                with col2:
                    st.subheader("Cooling Load Explanation")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap.Explanation(values=cooling_shap_current[0],
                                                        base_values=cooling_explainer.expected_value,
                                                        data=X_current.iloc[0]), show=False)
                    st.pyplot(fig)
                    plt.clf()

# ============ TAB 3: COUNTERFACTUAL ANALYSIS ============
with tab3:
    st.header("What-If Analysis: Counterfactual Scenarios")

    if role == "Architect":
        st.info("**For Architects**: Explore how changing design features can impact energy consumption. Use this to optimize your designs for better efficiency.")

    if 'current_prediction' not in st.session_state:
        st.warning("Please make a prediction in the 'Building Predictor' tab first.")
    else:
        current_total = st.session_state['current_prediction'].sum()

        # Input target variable - Total Load or separate Heating/Cooling
        target_type = st.radio("Optimize for:", ["Total Load", "Heating Load and Cooling Load"])

        if target_type == "Total Load":
            st.subheader("Set Energy Reduction Target")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Total Energy", f"{current_total:.2f}")

            with col2:
                reduction_pct = st.slider("Target Reduction %", 0, 50, 20, 5)
                target_energy = current_total * (1 - reduction_pct/100)
                st.metric("Target Total Energy", f"{target_energy:.2f}")

            with col3:
                st.metric("Required Reduction", f"{current_total - target_energy:.2f}")

            # Generate counterfactuals button
            if st.button("Generate Counterfactual Scenarios"):
                with st.spinner("Finding optimal design changes..."):
                    # Load data and model
                    X_train, X_test, y_train, y_test = load_data_combined_loads()
                    model = load_combined_model()

                    # Create DiCE data and model objects
                    combined_df = pd.concat([X_train, y_train], axis=1)

                    dice_data = dice_ml.Data(
                        dataframe=combined_df,
                        continuous_features=X_train.columns.tolist(),
                        outcome_name='Combined Load'
                    )

                    dice_model = dice_ml.Model(model=model, backend='sklearn', model_type='regressor')
                    dice_explainer = dice_ml.Dice(data_interface=dice_data, model_interface=dice_model, method='random')

                    # Current input
                    X_current = st.session_state['current_input']
                    
                    query_instance = X_current.iloc[0:1]

                    # Generate counterfactuals
                    counterfactual = dice_explainer.generate_counterfactuals(
                        query_instance,
                        total_CFs=3,
                        desired_range=[0, target_energy]
                    )

                    st.subheader("Counterfactual Scenarios")
                    cf_df = counterfactual.cf_examples_list[0].final_cfs_df
                    st.dataframe(cf_df)
        
        elif target_type == "Heating Load and Cooling Load":
            current_heating_load = st.session_state['current_prediction'][0][0]
            current_cooling_load = st.session_state['current_prediction'][0][1]

            X_train, X_test, y_train, y_test = load_data_splits()

            st.subheader("Set Energy Reduction Target")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Heating Load", f"{current_heating_load:.2f}")
                st.metric("Current Cooling Load", f"{current_cooling_load:.2f}")

            with col2:
                st.metric("Max Known Heating Load", f"{y_train.iloc[:, 0].max() :.2f}")
                st.metric("Max Known Cooling Load", f"{y_train.iloc[:, 1].max() :.2f}")

            with col3:
                percentage_of_max = st.slider("Target % of Max Known Loads", 0, 100, 80, 5)
                target_heating = y_train.iloc[:, 0].max() * (percentage_of_max / 100)
                target_cooling = y_train.iloc[:, 1].max() * (percentage_of_max / 100)

            with col4:
                st.metric("Target Heating Load", f"{target_heating:.2f}")
                st.metric("Target Cooling Load", f"{target_cooling:.2f}")

            # Generate counterfactuals button
            if st.button("Generate Counterfactual Scenarios"):
                with st.spinner("Finding optimal design changes..."):
                    model = load_model()

                    multi_object_counterfactual_wrapper = MultiTargetWrapper(
                        model=model,
                        heating_max=y_train.iloc[:, 0].max(),
                        cooling_max=y_train.iloc[:, 1].max()
                    )

                    multi_object_dice_data = dice_ml.Data(dataframe=X_train.assign(Composite_Score=0),
                    continuous_features=X_train.columns.tolist(),
                    outcome_name='Composite_Score')

                    multi_object_dice_model = dice_ml.Model(model=multi_object_counterfactual_wrapper, backend="sklearn", model_type='regressor')
                    multi_object_dice_explainer = dice_ml.Dice(multi_object_dice_data, multi_object_dice_model, method="random")

                    # Current input
                    X_current = st.session_state['current_input']
                    query_instance = X_current.iloc[0:1]

                    desired_range = [0, percentage_of_max / 100]

                    multi_target_counterfactuals = multi_object_dice_explainer.generate_counterfactuals(
                        query_instance, 
                        total_CFs=3, 
                        desired_range=desired_range
                    )

                    # Decompose counterfactuals back to Heating and Cooling Loads
                    cf_df = multi_target_counterfactuals.cf_examples_list[0].final_cfs_df.drop('Composite_Score', axis=1)

                    predictions = model.predict(cf_df)
                    cf_df['Predicted Heating Load'] = predictions[:, 0]
                    cf_df['Predicted Cooling Load'] = predictions[:, 1]

                    # Display counterfactuals
                    st.subheader("Counterfactual Scenarios")
                    st.dataframe(cf_df)

# ============ TAB 4: PORTFOLIO ANALYSIS ============
with tab4:
    st.header("Building Portfolio Analysis")

    if role == "Regulator":
        st.info("**For Regulators**: Analyze energy performance across a portfolio of buildings to identify trends and set efficiency standards.")

    df = load_data()
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Buildings", f"{df.shape[0]}")

    with col2:
        avg_heating = df['Heating Load'].mean()
        st.metric("Avg Heating Load", f"{avg_heating:.2f}")
    
    with col3:
        avg_cooling = df['Cooling Load'].mean()
        st.metric("Avg Cooling Load", f"{avg_cooling:.2f}")

    with col4:
        avg_total = (df['Heating Load'] + df['Cooling Load']).mean()
        st.metric("Avg Total Load", f"{avg_total:.2f}")

    # Distribution plots
    st.subheader("Energy Distribution Analysis")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df['Heating Load'], bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title("Heating Load Distribution")
    axes[0].set_xlabel("Heating Load")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df['Cooling Load'], bins=20, color='lightgreen', edgecolor='black')
    axes[1].set_title("Cooling Load Distribution")
    axes[1].set_xlabel("Cooling Load")
    axes[1].set_ylabel("Frequency")

    total_load = df['Heating Load'] + df['Cooling Load']
    axes[2].hist(total_load, bins=20, color='salmon', edgecolor='black')
    axes[2].set_title("Total Load Distribution")
    axes[2].set_xlabel("Total Load")
    axes[2].set_ylabel("Frequency")

    st.pyplot(fig)

    # Feature correlation with energy
    st.subheader("Feature Impact Across Portfolio")

    feature_choice = st.selectbox("Select Feature to Analyze", 
                                  df.columns[:-2].tolist())
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df[feature_choice], df['Heating Load'] + df['Cooling Load'], alpha=0.6)
    ax.set_xlabel(feature_choice)
    ax.set_ylabel("Total Energy Load")
    ax.set_title(f"{feature_choice} vs Total Energy Load")
    
    # Add trend line
    z = np.polyfit(df[feature_choice], df['Heating Load'] + df['Cooling Load'], 1)
    p = np.poly1d(z)
    ax.plot(df[feature_choice], p(df[feature_choice]), "r--")

    st.pyplot(fig)

# Footer
st.markdown("---")
st.sidebar.markdown("""
- [Data Source: UCI Machine Learning Repository - Energy Efficiency Data Set](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
""")
st.sidebar.markdown("---")
st.sidebar.info("This is a prototype dashboard for demonstration purposes.")
