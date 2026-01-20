import warnings
warnings.filterwarnings('ignore')

import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import dice_ml

from catboost import CatBoostRegressor

# Constants
OUTPUT_DIR = "outputs/"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models/")
DATA_SPLITS = os.path.join(OUTPUT_DIR, "splits/")

DATA_DIR = "data/"
DATA_FILE = "clean_data.csv"
INPUT_PATH  = os.path.join(DATA_DIR, DATA_FILE)

GLOBAL_SEED = 63
categorical_features = ['Orientation', 'Glazing Area Distribution']

# Page configuration
st.set_page_config(page_title="Energy Efficiency XAI Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #8dff55;
        margin-top: 1rem !important;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    div[data-baseweb="select"]>div {
        cursor: pointer !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'interaction_values' not in st.session_state:
    st.session_state.interaction_values = None

def set_global_seed():
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

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

@st.cache_resource
def load_model_and_explainer():
    model_study = joblib.load(os.path.join(MODEL_DIR, "catboost_regressor.joblib"))
    best_params = model_study.best_params_

    best_model = CatBoostRegressor(**best_params, logging_level='Silent')

    X_train, X_test, y_train, _ = load_data_splits()
    best_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(best_model)
    interaction_values = explainer.shap_interaction_values(X_train)

    return best_model, explainer, X_train, y_train, interaction_values

set_global_seed()

# Stakeholder - Question mapping
STAKEHOLDER_QUESTION_MAPPING = {
    "Building Owners": {
        "questions": [
            "Why is my building predicted to have this energy load?",
            "What small changes can reduce energy usage while keeping some features the same?",
            "Is my building's energy performance typical for its design?"
        ],
        "explanation_type": ["local_shap", "counterfactual", "local_comparison"]
    },
    "Architects": {
        "questions": [
            "What design features generally impact energy requirements the most?",
            "How does this feature tend to affect the total energy load?",
            "How does the increase/decrease of features affect the total energy load?",
            "Are there non-linear or interaction effects among the features?"
        ],
        "explanation_type": ["global_shap", "feature_dependence", "global_beeswarm", "interaction"]
    },
    "Energy Consultants": {
        "questions": [
            "Which features are making this design efficient/inefficient?",
            "What overall changes can improve the efficiency of this building?",
            "If we cannot change some specific features, what can we change to optimize the energy efficiency?"
        ],
        "explanation_type": ["local_shap", "counterfactual", "counterfactual_restrict"]
    },
    "Regulators": {
        "questions": [
            "Are the model decisions consistent and reliable?",
            "How much do the factors contribute to energy efficiency predictions overall?",
            "How do predictions vary across different building features?"
        ],
        "explanation_type": ["model_performance", "global_shap", "distribution"]
    }
}

def plot_local_shap(explainer, instance, feature_names, role="Building Owners"):
    shap_values = explainer.shap_values(instance)

    if role == "Building Owners":
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.force(
            explainer.expected_value,
            shap_values[0], 
            feature_names=feature_names, 
            show=False, 
            matplotlib=True
        )
        fig = plt.gcf() 
        st.pyplot(fig)
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=instance.values[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()

def plot_global_shap(explainer, X_train, feature_names):
    shap_values = explainer.shap_values(X_train)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    st.pyplot(fig)
    plt.close()

def plot_feature_dependence(explainer, X_train, feature_names, feature):
    shap_values = explainer(X_train)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.scatter(shap_values[:, feature], color="#1E88E5", show=False)
    fig = plt.gcf() 
    st.pyplot(fig)
    plt.close(fig)

def generate_counterfactual_owners(model, instance, feature_names, features_to_fix, desired_range, X_train):
    current_prediction = model.predict(instance)[0]
    
    st.write(f"**Current Prediction:** {current_prediction:.2f} units")
    st.write(f"**Desired Range:** {desired_range[0]:.2f} - {desired_range[1]:.2f} units")
    
    counterfactuals = []
    
    for feature_idx, feature in enumerate(feature_names):
        if feature in features_to_fix:
            continue

        if feature in categorical_features:
            # Only consider the neighboring categories for categorical features
            unique_values = sorted(X_train[feature].unique())
            current_value = instance.iloc[0, feature_idx]
            current_index = unique_values.index(current_value)

            neighbor_indices = []
            if current_index == 0:
                neighbor_indices.append(1)
            elif current_index == len(unique_values) - 1:
                neighbor_indices.append(len(unique_values) - 2)
            else:
                neighbor_indices.extend([current_index - 1, current_index + 1])

            for idx in neighbor_indices:
                cf_instance = instance.copy()
                cf_instance.iloc[0, feature_idx] = unique_values[idx]
                
                cf_prediction = model.predict(cf_instance)[0]
                
                if desired_range[0] <= cf_prediction <= desired_range[1]:
                    counterfactuals.append({
                        'Feature': feature,
                        'Original': instance.iloc[0, feature_idx],
                        'Modified': unique_values[idx],
                        'Change (%)': 'N/A',
                        'New Prediction': cf_prediction
                    })
            continue
            
        for delta in [-0.2, -0.1, 0.1, 0.2]:
            cf_instance = instance.copy()
            cf_instance.iloc[0, feature_idx] *= (1 + delta)
            
            cf_prediction = model.predict(cf_instance)[0]
            
            if desired_range[0] <= cf_prediction <= desired_range[1]:
                change_pct = delta * 100
                counterfactuals.append({
                    'Feature': feature,
                    'Original': instance.iloc[0, feature_idx],
                    'Modified': cf_instance.iloc[0, feature_idx],
                    'Change (%)': change_pct,
                    'New Prediction': cf_prediction
                })
    
    if counterfactuals:
        cf_df = pd.DataFrame(counterfactuals)
        cf_df = cf_df.sort_values('Change (%)',
            key=lambda x: pd.to_numeric(x, errors='coerce').abs(),
            na_position='last'
        ).reset_index(drop=True)
        cf_df.index += 1
        st.write("**Possible Counterfactual Changes:**")
        st.dataframe(cf_df, use_container_width=True)
        st.info("**Interpretation:** Shows how a feature with small changes (maximum 20%) can push the energy load to the desired range.")
    else:
        st.warning("No simple counterfactuals found. Try adjusting the desired range.")

def generate_counterfactuals(model, X_train, y_train, instance, desired_range, features=None):
    current_prediction = model.predict(instance)[0]
    
    st.write(f"**Current Prediction:** {current_prediction:.2f} units")
    st.write(f"**Desired Range:** {desired_range[0]:.2f} - {desired_range[1]:.2f} units")

    combined_df = pd.concat([X_train, y_train], axis=1)
    if features is None or features == []:
        features = X_train.columns.tolist()

    dice_data = dice_ml.Data(
        dataframe=combined_df,
        continuous_features=X_train.columns.tolist(),
        outcome_name='Combined Load'
    )

    dice_model = dice_ml.Model(model=model, backend='sklearn', model_type='regressor')
    dice_explainer = dice_ml.Dice(data_interface=dice_data, model_interface=dice_model, method='random')
    
    query_instance = instance.iloc[0:1]

    # Generate counterfactuals
    try:
        counterfactual = dice_explainer.generate_counterfactuals(
            query_instance,
            total_CFs=5,
            desired_range=desired_range,
            features_to_vary=features
        )

        if counterfactual:
            cf_df = counterfactual.cf_examples_list[0].final_cfs_df
            cf_df.index += 1
            st.write("**Possible Counterfactual Changes:**")
            st.dataframe(cf_df, use_container_width=True)
            st.info("**Interpretation:** Shows alternative designs that can push the energy load within the desired range.")
        else:
            st.warning("No counterfactuals found for this range.")
    except Exception as e:
        st.warning("Could not generate counterfactuals for this configuration.")

def plot_interaction(feature_one, feature_two, interaction_values, X_train):
    if feature_one != feature_two:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            (feature_one, feature_two),
            interaction_values, 
            X_train,
            show=False
        )
        fig = plt.gcf() 
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Please input two different features.")

def main():
    st.markdown('<p class="main-header">Energy Efficiency - XAI Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard helps architects, owners, energy consultants, and regulators understand and optimize building energy consumption
    using explainable AI techniques.
    """)

    # Load data
    df = load_data()
    X = df.drop('Combined Load', axis=1)
    y = df['Combined Load']
    feature_names = X.columns.tolist()

    # Train model
    if st.session_state.model is None:
        with st.spinner("Training model..."):
            model, explainer, X_train, y_train, interaction_values = load_model_and_explainer()
            st.session_state.model = model
            st.session_state.explainer = explainer
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.feature_names = feature_names
            st.session_state.interaction_values = interaction_values

    model = st.session_state.model
    explainer = st.session_state.explainer
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    interaction_values = st.session_state.interaction_values

    st.success("Model loaded and ready!")
    st.markdown("---")

    # Stakeholder selection
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown('<p class="sub-header">Select Role</p>', unsafe_allow_html=True)
        selected_role = st.radio(
            "Stakeholder",
            options=list(STAKEHOLDER_QUESTION_MAPPING.keys()),
            label_visibility="collapsed"
        )

    with col2:
        st.markdown(f'<p class="sub-header">{selected_role}</p>', unsafe_allow_html=True)

        questions = STAKEHOLDER_QUESTION_MAPPING[selected_role]['questions']
        selected_question = st.selectbox("Select your question:", questions)

        question_idx = questions.index(selected_question)
        explanation_type = STAKEHOLDER_QUESTION_MAPPING[selected_role]["explanation_type"][question_idx]
        st.markdown("---")

        # Create layout for inputs and visualization
        needs_local = explanation_type in ["local_shap", "counterfactual", "counterfactual_restrict", "local_comparison"]
        needs_counterfactual = explanation_type == "counterfactual"
        
        if needs_local or needs_counterfactual:
            input_col, viz_col = st.columns([1, 2])
        else:
            viz_col = st.container()
            input_col = None

        # Input section
        instance_data = {}
        features_to_fix = None
        desired_range = None

        if needs_local and input_col:
            with input_col:
                st.markdown("### Building Specifications")

                for feature in feature_names:
                    if feature in categorical_features:
                        options = sorted(X[feature].unique())
                        default_option = options[1] if len(options) > 1 else options[0]
                        instance_data[feature] = st.selectbox(
                            feature.replace('_', ' '),
                            options=options,
                            index=options.index(default_option)
                        )
                        continue

                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    default_val = float(X[feature].median())

                    instance_data[feature] = st.slider(
                        feature.replace('_', ' '),
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=(max_val - min_val) / 100
                    )

        # Visualization section
        with viz_col:
            st.markdown("### Explanation")
            
            if explanation_type == "local_shap":
                instance = pd.DataFrame([instance_data])
                prediction = model.predict(instance)[0]
                st.metric("Predicted Energy Load", f"{prediction:.2f} units")
                plot_local_shap(explainer, instance, feature_names, selected_role)
                
                st.info("**Interpretation:** Red bars push the prediction higher, blue bars push it lower. Longer bars indicate stronger effects.")

            elif explanation_type == "global_shap":
                shap_values = explainer.shap_values(X_train)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.plots.bar(shap.Explanation(
                    values=shap_values,
                    base_values=explainer.expected_value,
                    data=X_train,
                    feature_names=feature_names
                ), show=False)

                st.pyplot(fig)
                plt.close()
                
                st.info("**Interpretation:** This shows the absolute impact of each feature on model predictions across all samples.")
                
            elif explanation_type == "feature_dependence":
                feature_of_interest = st.selectbox(
                    "Select feature to analyze:",
                    options=feature_names,
                    index=feature_names.index("Orientation") if "Orientation" in feature_names else 0
                )

                plot_feature_dependence(explainer, X_train, feature_names, feature_of_interest)
                st.info(f"**Interpretation:** Shows how {feature_of_interest} affects energy load predictions.")

            elif explanation_type == "counterfactual":
                st.markdown("### Counterfactual Settings")
                    
                if selected_role == "Building Owners":
                    features_to_fix = st.multiselect(
                        "Keep these features unchanged:",
                        options=feature_names,
                        default=[]
                    )
                
                current_pred = model.predict(pd.DataFrame([instance_data]))[0]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    min_desired = st.number_input(
                        "Min desired load:",
                        value=max(0.0, current_pred - 10),
                        step=1.0
                    )
                with col_b:
                    max_desired = st.number_input(
                        "Max desired load:",
                        value=current_pred - 5,
                        step=1.0
                    )
                desired_range = (min_desired, max_desired)

                instance = pd.DataFrame([instance_data])
                if selected_role == "Building Owners":
                    generate_counterfactual_owners(model, instance, feature_names, features_to_fix, desired_range, X_train)
                else:
                    generate_counterfactuals(model, X_train, y_train, instance, desired_range)
            
            elif explanation_type == "model_performance":
                shap_values = explainer.shap_values(X_train)
                
                # Global feature importance
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                feature_importance = np.abs(shap_values).mean(axis=0)
                sorted_idx = np.argsort(feature_importance)
                
                ax1.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                ax1.set_yticks(range(len(sorted_idx)))
                ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
                ax1.set_xlabel('Mean |SHAP value|')
                ax1.set_title('Global Feature Importance')
                
                # Prediction distribution
                predictions = model.predict(X_train)
                ax2.hist(predictions, bins=30, edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Predicted Energy Load (units)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Prediction Distribution')
                
                st.pyplot(fig)
                plt.close()
                
                st.info("**Interpretation:** Left plot shows which features are most important globally. Right plot shows the distribution of model predictions.")

            elif explanation_type == "global_beeswarm":
                plot_global_shap(explainer, X_train, feature_names)
                st.info("**Interpretation:** Shows how the increase and decrease of features affect the total energy load.")
            
            elif explanation_type == "local_comparison":
                features_to_fix_for_comparison = st.multiselect(
                    "Buildings with similar:",
                    options=feature_names,
                    default=feature_names[:2]
                )

                instance = pd.DataFrame([instance_data])
                prediction = model.predict(instance)[0]

                print(type(X_train))
                print(type(instance_data))
                
                eps = 1e-8

                instance_subset = pd.Series(
                    {f: instance_data[f] for f in features_to_fix_for_comparison}
                )

                X_subset = X_train[features_to_fix_for_comparison]

                relative_diff = np.abs(X_subset - instance_subset) / (
                    np.abs(instance_subset) + eps
                )

                similar_mask = (relative_diff <= 0.10).all(axis=1)

                similar_predictions = (
                    model.predict(X_train.loc[similar_mask])
                    if similar_mask.any()
                    else []
                )
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Your Building", f"{prediction:.2f} units")
                with col_b:
                    if len(similar_predictions) > 0:
                        st.metric("Similar Buildings (Avg)", f"{similar_predictions.mean():.2f} units")
                with col_c:
                    st.metric("Dataset Average", f"{model.predict(X_train).mean():.2f} units")
                
                # plot_local_shap(model, explainer, X_train, instance, feature_names)
                # Plot bar chart comparing predictions
                labels = ['Your Building', 'Similar Buildings (Avg)', 'Dataset Average']
                values = [
                    prediction,
                    similar_predictions.mean() if len(similar_predictions) > 0 else 0,
                    model.predict(X_train).mean()
                ]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels, values, color=['blue', 'green', 'red'])
                ax.set_ylabel('Energy Load (units)')
                ax.set_title('Prediction Comparison')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

                st.info("**Interpretation:** Compares the energy load of other buildings with similar features to that of your building.")

            elif explanation_type == "distribution":
                predictions = model.predict(X_train)

                feature_to_plot = st.selectbox(
                    "Select feature to segment by:",
                    options=feature_names,
                    index=feature_names.index("Overall Height") if "Overall Height" in feature_names else 0
                )
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Overall distribution
                axes[0, 0].hist(predictions, bins=30, edgecolor='black', alpha=0.7)
                axes[0, 0].set_xlabel('Energy Load (units)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Overall Prediction Distribution')

                # By selected feature
                for value in sorted(X_train[feature_to_plot].unique()):
                    mask = X_train[feature_to_plot] == value
                    axes[0, 1].hist(predictions[mask], alpha=0.5, label=f'{feature_to_plot}={value}', bins=20)

                axes[0, 1].set_xlabel('Energy Load (units)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title(f'Distribution by {feature_to_plot}')
                axes[0, 1].legend()
                
                # Scatter: Actual features vs prediction
                axes[1, 0].scatter(X_train[feature_to_plot], predictions, alpha=0.5)
                axes[1, 0].set_xlabel(f'{feature_to_plot}')
                axes[1, 0].set_ylabel('Predicted Energy Load (units)')
                axes[1, 0].set_title(f'{feature_to_plot} vs Energy Load')

                axes[1, 1].axis('off')  # Empty plot
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info("**Interpretation:** These plots show how predictions vary across a building characteristic.")

            elif explanation_type == "interaction":
                col_a, col_b = st.columns(2)
                with col_a:
                    feature_one = st.selectbox("Select first feature:", feature_names, index=0)
                with col_b:
                    feature_two = st.selectbox("Select second feature:", feature_names, index=1)
                
                plot_interaction(feature_one, feature_two, interaction_values, X_train)

                st.info("**Interpretation:** Shows how a feature interacts with another feature, can reveal non-linear relationships.")

            elif explanation_type == "counterfactual_restrict":
                st.markdown("### Counterfactual Settings")

                features_to_change = st.multiselect(
                    "Change these features:",
                    options=feature_names,
                    default=feature_names[:2]
                )

                current_pred = model.predict(pd.DataFrame([instance_data]))[0]

                col_a, col_b = st.columns(2)
                with col_a:
                    min_desired = st.number_input(
                        "Min desired load:",
                        value=max(0.0, current_pred - 10),
                        step=1.0
                    )
                with col_b:
                    max_desired = st.number_input(
                        "Max desired load:",
                        value=current_pred - 5,
                        step=1.0
                    )
                desired_range = (min_desired, max_desired)

                instance = pd.DataFrame([instance_data])

                generate_counterfactuals(model, X_train, y_train, instance, desired_range, features_to_change)

    # Footer
    st.markdown("---")
    st.sidebar.markdown("""
    - [Data Source: UCI Machine Learning Repository - Energy Efficiency Data Set](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("This is a prototype dashboard for demonstration purposes.")

if __name__ == "__main__":
    main()