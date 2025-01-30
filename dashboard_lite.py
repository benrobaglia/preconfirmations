import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from utils import *
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import zipfile

st.set_page_config(page_title="Chorus One Preconf Pricing",
                   page_icon="img/c1_img.png", 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

# Change the bin legend of the histogram (start to end instead of middle)
# Mention the Sample size in the text
# Fix the target for all models
# change RF parameters in the dashboard
# Show training set vs testing set


st.markdown(
    """
    <style>
    /* Target the sidebar */
    [data-testid="stSidebarContent"] * {
        font-size: 25px !important; 
    }

    /* Increase the sidebar title size */
    [data-testid="stSidebarContent"] h1 {
        font-size: 34px !important; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.image("img/logo.svg", width=200)
st.sidebar.title("Navigation")
pages = ["Overview", "Data Settings", "Simulation"]
page = st.sidebar.radio("", pages)

# Landing page
if page == "Overview":
    
    st.title("📈 Preconfirmation Pricing by Chorus One")
    
    # st.subheader("Developed by Chorus One")

    st.subheader("How to use this dashboard?")
    st.markdown("""
                1. **Navigate** through the sidebar to access different sections.
                2. **Adjust Parameters** in the Data Settings and Simulation pages to customize the analysis.
                3. **View Results** and insights to understand the impact of your configurations.
                4. **Download Data** to conduct your own analysis.

                """
                )

    
    st.markdown("""
                ### 📘 **Introduction**

                A **preconfirmation** (or preconf) is a credible, signed commitment from an Ethereum block proposer guaranteeing that a specific transaction will be included in a future block.

                Chorus One is the leading node operator on preconfirmations, with operational experience extending to the earliest private testnets (ZuBerlin 2024).

                If priced correctly, preconfirmations can unlock new revenue for delegators. The purpose of this dashboard is provide a hands-on view of how this can be done. 

                Preconf pricing is a complex problem. One critical element in this process is the ability to estimate the future priority fee (PF).
                If a transaction's PF is too low compared to future fees, preconfirming it may result in a lower expected value for the builder and proposer.

                The exact pricing mechanism for preconfirmations is an active exploration, pending complete information on slashing- and penalty-mechanisms. Due to this limitation, building an exact P&L model is not yet possible. 

                Therefore, the primary focus of this dashboard is to provide accurate predictions for the priority fee per gas unit, which serves as a key input for any potential preconfirmation pricing strategy.
                """)

    st.markdown("""
                ### 🔍 **Key Features**
                - 🛠️ **Estimator Bias Reducer**: Not all transactions are equal. Some can bias the estimators. 
                With this feature, you can select the approach for **choosing which transactions** to include in the model, ensuring a more robust estimator with minimal bias.

                - 📉 **Dynamic Fee Predictions**: Use machine learning models to predict **future priority fees**. 
                These predictions help determine which transactions are worth preconfirming.

                - 📊 **Customizable Estimators**: Explore preconfirmation pricing using multiple model approaches, 
                including a **Quantile Heuristic**, **Linear Regression**, and **Machine Learning model**.

                - ⚙️ **Interactive Configurations**: Adjust model parameters to see how they affect **priority fees, and preconfirmations**.

                We invite you to explore the dashboard and leverage its features to enhance your understanding of preconfirmation pricing. Enjoy your journey!

                """)

    # st.markdown("""
    #     ### 📘 **Introduction**
    #     **Preconfirmations** are a hot topic in Ethereum research aimed at improving the user experience.  
    #     A **preconfirmation** (or **preconf**) is a credible, signed commitment from an Ethereum block proposer guaranteeing that a specific transaction will be included in a future block.

    #     However, **pricing preconfirmations is a complex problem**. One critical element in this process is the ability to estimate the **future priority fee (PF)**.  
    #     If a transaction's PF is too low compared to future fees, preconfirming it may result in **a lower expected value for the builder and proposer**.  

    #     Additionally, the **pricing mechanism for preconfirmations is still under discussion**, as the conditions for **slashing and penalties** have not yet been fully defined.  
    #     This uncertainty makes it challenging to establish a reliable **P&L model** and conduct a realistic **backtest** since **preconfirmations are not live on Ethereum**.  
    #     Therefore, the primary focus of this dashboard is to provide accurate predictions for the **priority fee per gas unit**, which serves as a key input for any potential preconfirmation pricing strategy.
    #     """)



elif page == "Data Settings":
    st.title('Data Settings')
    st.subheader("Reducing the bias")
    st.markdown("""
                The Priority Fee per Gas Unit is highly influenced by a transaction's position within the block (as shown in the interactive figure below).
                To reduce estimator bias, we exclude outliers, especially those caused by "toxic flow" (cite paper link).
                
                This section offers an interactive tool to explore and customize the selection of transaction positions, enabling users to build a more robust and unbiased estimator.
                """)

    # tx_data = pd.read_csv(f'{DATA_PATH}txs_sample.csv')
    tx_data = pd.read_parquet('txs_sample.parquet')
    aggregated = agg_pf_per_gas_per_position(tx_data)

    # Reduce the plot
    # aggregated = aggregated[:300]

    st.subheader("Priority Fee Metrics for Each Position in the Block")

    positions = aggregated['position']
    metrics = {
        'Mean': aggregated['mean_priority_fee'],
        'Min': aggregated['min_priority_fee'],
        'Max': aggregated['max_priority_fee'],
        'Median': aggregated['median_priority_fee'],
        '25th Percentile': aggregated['quantile_25_priority_fee'],
        '75th Percentile': aggregated['quantile_75_priority_fee'],
        '95th Percentile': aggregated['quantile_95_priority_fee']
    }

    st.markdown(f"""
                The transaction dataset contains all transactions between block {tx_data.block_number.min():,} and block {tx_data.block_number.max():,} totaling {len(tx_data):,} transactions.
                """)

    # Create Traces
    traces = []
    for metric, values in metrics.items():
        if metric in ['Max', '95th Percentile']:
            traces.append(go.Scatter(mode='lines', name=metric,
                          x=positions, y=values, visible='legendonly'))

        else:
            traces.append(go.Scatter(
                mode='lines', name=metric, x=positions, y=values))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Aggregated Metrics for Priority Fee per Gas by Position",
            x=0,  # Aligns to the left
            xanchor='left'  # Ensures the text is aligned from the left
        ),
        xaxis_title="Position",
        yaxis_title="Priority Fee per Gas (Gwei)",
        barmode='group',
        bargap=0.2,
        height=600
    )

    # Display Plot
    st.plotly_chart(fig)

    st.subheader("Selecting Transactions based on the Block Position")

    default_start_position = 10
    default_end_position = 150
    eligible_txs = tx_data.loc[(tx_data['position'] > default_start_position) & (
        tx_data['position'] < default_end_position)]

    with st.form(key="eligibility_form"):
        start_position = st.slider('Start Position', min_value=1, max_value=300, value=default_start_position,
                                   step=1, help="Set the starting position for eligible transactions.")
        end_position = st.slider('End Position', min_value=50, max_value=400, value=default_end_position,
                                 step=1, help="Set the ending position for eligible transactions.")
        submitted = st.form_submit_button("Run")

    if submitted:
        eligible_txs = tx_data.loc[(tx_data['position'] > start_position) & (
            tx_data['position'] < end_position)]

    st.header("Results")
    percentage_of_eligible_txs = len(
        eligible_txs) / len(tx_data) if len(tx_data) > 0 else 0
    st.metric(
        label="Percentage of Transactions Left to Price Preconfirmations",
        value=f"{percentage_of_eligible_txs:.2%}",
        help="This shows the percentage of transactions that are in the position range. This is the percentage of transactions that are left to price preconfirmations."
    )

    st.subheader("Priority Fee Distribution of the Selected Set")

    # Convert to Gwei
    filtered_priority_fees = eligible_txs['priority_fee_per_gas'] * 1e9

    # filtered_priority_fees.loc[filtered_priority_fees<0] = 0.1

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=filtered_priority_fees,
        nbinsx=300,  # Number of bins
        marker_color='rgb(26, 118, 255)',
        opacity=0.75
    ))

    fig.update_layout(
        title="Distribution of Priority Fee per Gas",
        xaxis=dict(
            tickmode='auto',
            title="Priority Fee per Gas (in Gwei)",
            range=[0.1, None]
        ),
        yaxis_title="Count",
        title_x=0.,
        bargap=0.2,  # Space between bars
        bargroupgap=0.1  # Space between groups (for multiple histograms)
    )

    # Streamlit app
    st.plotly_chart(fig)

    st.markdown("""
                Want to conduct your own analysis? Click the button below to download the filtered dataset!
                """)
    csv = eligible_txs.to_csv(index=False)
    st.download_button(
        label="Download Filtered Transactions as CSV",
        data=csv,
        file_name='filtered_transactions.csv',
        mime='text/csv',
    )


# Model Configuration Page
elif page == "Simulation":
    st.title("Simulation")

    st.markdown(r"""
        This section allows testing of different model configurations to predict a **priority fee per gas unit (PF/GU) threshold** for pre-confirming Ethereum transactions. Simulations are pre-computed to ensure a smooth user experience.

        ### **Problem Formulation**
        We aim to predict a threshold $T$ for **PF/GU**. A transaction $tx$ is pre-confirmed if: $\frac{PF_{tx}}{GU_{tx}} > T$
        
        We denote $\rho_{tx}(T) = \frac{PF_{tx}}{GU_{tx}}- T$ the preconfirmation error. The goal is to accurately estimate $T$ to minimize pre-confirmation errors while maximizing the number of preconfirmed transactions.

        ### **Models**
        - **Quantile Heuristic (QH)**: Uses a quantile of past PF/GU values as the threshold.
        - **Linear Regression (LR)**: Predicts $T$ using a linear model of historical transaction data.
        - **Random Forest (RF)**: Uses an ensemble of decision trees to capture non-linear patterns.
        - **Upper Bound (UB)**: An oracle model that uses the true future PF/GU values. This baseline gives an upper limit on what a model can acheive.
        ### **Metrics**
        - **Mean Absolute Error (MAE)**: Measures the average deviation between predicted $\hat{T}$ and actual $T$:

        $\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| T_i - \hat{T}_i \right| $
        - **Average Preconfirmed Error**: Captures losses from incorrect pre-confirmations:
        $\text{Error} = \frac{1}{N} \sum_{i=1}^N \rho_{i}(\hat{T}_i)$
        - **Percentage of Eligible Transactions**: Tracks the share of transactions that meet the pre-confirmation condition.

        """)

    st.subheader("Model Configuration")

    # Possible parameter values
    quantile_values = np.arange(10, 100, 10)
    quantile_lag_values = [10, 20, 32]
    training_threshold_values = [21331500, 21332079]
    lr_lags_values = [5, 10, 15, 32]
    lr_features = ['block_gas_used_lag_1']
    rf_features = [
        'block_gas_used_lag_1',
        'tx_count_lag_1',
        'mean_priority_fee_per_gas',
        'min_priority_fee_per_gas',
        'max_priority_fee_per_gas',
        'skew_priority_fee_per_gas'
    ]
    n_estimators_values = [50, 100, 300]
    max_depth_values = [5, 10, 20]

    # Initialize session state for model configuration
    session_defaults = {
        'quantile': 50,
        'max_window': 10,
        'lr_features': ['block_gas_used_lag_1'],
        'rf_features': ['block_gas_used_lag_1', 'tx_count_lag_1', 'mean_priority_fee_per_gas', 'min_priority_fee_per_gas', 'max_priority_fee_per_gas', 'skew_priority_fee_per_gas'],
        'lr_lags': 10,
        'training_threshold': 21331500,
        'n_estimators': 100,
        'max_depth': 10
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    tabs = st.tabs(["Quantile Heuristic", "Linear Regression",
                   "Random Forest Regression"])

    # Quantile Heuristic Configuration
    with tabs[0]:
        st.subheader("Configure Quantile Heuristic")
        quantile = st.selectbox(
            "Select Quantile", quantile_values, index=quantile_values.tolist().index(50))
        max_window = st.selectbox("Select Maximum Window Size", quantile_lag_values, index=quantile_lag_values.index(10))

        if st.button("Save Configuration for Quantile Heuristic"):
            st.session_state['quantile'] = quantile
            st.session_state['max_window'] = max_window
            st.success(f"Configuration saved! Quantile: {quantile}, Max Window: {max_window}")

    # Linear Regression Configuration
    with tabs[1]:
        st.subheader("Configure Linear Regression Model")
        lr_lags = st.selectbox("Select Target Lags",
                               lr_lags_values, index=lr_lags_values.index(10))
        training_threshold = st.selectbox(
            "Select Training Threshold", training_threshold_values, index=0)

        if st.button("Save Configuration for Linear Regression"):
            st.session_state['lr_lags'] = lr_lags
            st.session_state['training_threshold'] = training_threshold
            st.success(f"Configuration saved! LR Lags: {lr_lags}, Threshold: {training_threshold}")

    # Random Forest Configuration
    with tabs[2]:
        st.subheader("Configure Random Forest Model")
        n_estimators = st.slider(
            "Number of Trees (n_estimators)", min_value=100, max_value=300, value=100, step=100)
        max_depth = st.slider("Maximum Depth (max_depth)",
                              min_value=5, max_value=20, value=10, step=5)

        with st.expander("Parameter Details"):
            st.write(
                """
                - **n_estimators**: Specifies the number of trees in the ensemble. 
                A higher number of trees can improve accuracy but increases computation time.

                - **max_depth**: Defines the maximum depth of each tree. 
                Limiting the depth can prevent overfitting, but may reduce model complexity.

                For more information, visit the [official documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
                """
            )

        if st.button("Save Configuration for Random Forest"):
            st.session_state['n_estimators'] = n_estimators
            st.session_state['max_depth'] = max_depth
            st.success(f"Configuration saved! RF Lags: Trees: {n_estimators}, Depth: {max_depth}")

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.header("Simulation Results")


    # QH
    qh_pointer = f"{st.session_state['quantile']}_{st.session_state['max_window']}_{st.session_state['training_threshold']}.p"
    # global_quantile_metrics = pickle.load(
    #     open(f'precomputed_results/qh/global_qh_metrics/{qh_pointer}', 'rb'))
    # groups_quantile_metrics = pickle.load(
    #     open(f'precomputed_results/qh/groups_qh_metrics/{qh_pointer}', 'rb'))

    # LR
    lr_pointer = f"{st.session_state['quantile']}_{st.session_state['lr_lags']}_{st.session_state['training_threshold']}.p"
    # global_lr_metrics = pickle.load(
    #     open(f'precomputed_results/lr/global_lr_metrics/{lr_pointer}', 'rb'))
    # groups_lr_metrics = pickle.load(
    #     open(f'precomputed_results/lr/groups_lr_metrics/{lr_pointer}', 'rb'))
    # lr_results_train = pickle.load(
    #     open(f'precomputed_results/lr/lr_results_train/{lr_pointer}', 'rb'))

    # RF
    rf_pointer = f"{st.session_state['quantile']}_{st.session_state['training_threshold']}_{st.session_state['n_estimators']}_{st.session_state['max_depth']}.p"
    # global_rf_metrics = pickle.load(
    #     open(f'precomputed_results/rf/global_rf_metrics/{rf_pointer}', 'rb'))
    # groups_rf_metrics = pickle.load(
    #     open(f'precomputed_results/rf/groups_rf_metrics/{rf_pointer}', 'rb'))
    # rf_results_train = pickle.load(
    #     open(f'precomputed_results/rf/rf_results_train/{rf_pointer}', 'rb'))
    # rf_models = pickle.load(
    #     open(f'precomputed_results/rf/rf_models/{rf_pointer}', 'rb'))

    # UB
    ub_pointer = f"{st.session_state['quantile']}.p"
    # global_ub_metrics = pickle.load(
    #     open(f'precomputed_results/ub/global_ub_metrics/{ub_pointer}', 'rb'))
    # groups_ub_metrics = pickle.load(
    #     open(f'precomputed_results/ub/groups_ub_metrics/{ub_pointer}', 'rb'))


    @st.cache_data
    def load_all_results(zip_path, file_paths):
        results = {}
        with zipfile.ZipFile(zip_path, 'r') as z:
            for key, file_path in file_paths.items():
                if file_path in z.namelist():
                    results[key] = pickle.load(z.open(file_path))
                else:
                    print(f"File not found in archive: {file_path}")
        return results

    zip_path = "precomputed_results.zip"
    file_paths = {
        'global_quantile_metrics': f"precomputed_results/qh/global_qh_metrics/{qh_pointer}",
        'groups_quantile_metrics': f"precomputed_results/qh/groups_qh_metrics/{qh_pointer}",
        'global_lr_metrics': f"precomputed_results/lr/global_lr_metrics/{lr_pointer}",
        'groups_lr_metrics': f"precomputed_results/lr/groups_lr_metrics/{lr_pointer}",
        'lr_results_train': f"precomputed_results/lr/lr_results_train/{lr_pointer}",
        'global_rf_metrics': f"precomputed_results/rf/global_rf_metrics/{rf_pointer}",
        'groups_rf_metrics': f"precomputed_results/rf/groups_rf_metrics/{rf_pointer}",
        'rf_results_train': f"precomputed_results/rf/rf_results_train/{rf_pointer}",
        'rf_models': f"precomputed_results/rf/rf_models/{rf_pointer}",
        'global_ub_metrics': f"precomputed_results/ub/global_ub_metrics/{ub_pointer}",
        'groups_ub_metrics': f"precomputed_results/ub/groups_ub_metrics/{ub_pointer}"
    }
    results = load_all_results(zip_path, file_paths)

    global_quantile_metrics = results.get("global_quantile_metrics")
    groups_quantile_metrics = results.get("groups_quantile_metrics")
    global_lr_metrics = results.get("global_lr_metrics")
    groups_lr_metrics = results.get("groups_lr_metrics")
    lr_results_train = results.get("lr_results_train")
    global_rf_metrics = results.get("global_rf_metrics")
    groups_rf_metrics = results.get("groups_rf_metrics")
    rf_results_train = results.get("rf_results_train")
    rf_models = results.get("rf_models")
    global_ub_metrics = results.get("global_ub_metrics")
    groups_ub_metrics = results.get("groups_ub_metrics")


    global_results = [global_quantile_metrics,
                      global_lr_metrics, global_rf_metrics, global_ub_metrics]


    # print(len(groups_quantile_metrics))
    # print(len(groups_lr_metrics))
    # print(len(groups_rf_metrics))
    # print(len(groups_ub_metrics))

    global_results = pd.DataFrame(global_results, index=[
                                  'Quantile Heuristic', 'Linear Regression', 'Random Forest', 'Upper Bound (Oracle)'])
    global_results.drop(columns=['preconf_value'], inplace=True)
    global_results.rename(columns={'mae': 'Mean Absolute Error', "avg_preconfirmed_errors": 'Avg Preconfirmed Error',
                          'preconfirmations_eligible': "Eligible Transactions (%)"}, inplace=True)

    st.dataframe(global_results)

    st.header("Results per Transaction Topology")
    value_to_plot = st.selectbox('Select a value to plot', [
                                 'Avg Preconfirmed Error', 'Eligible Transactions (%)'])
    idxer = {'Avg Preconfirmed Error': 'avg_preconfirmed_errors',
             'Eligible Transactions (%)': 'preconfirmations_eligible'}
    fig = plot_grouped_bar(
        idxer[value_to_plot], groups_quantile_metrics, groups_lr_metrics, groups_rf_metrics, groups_ub_metrics)
    st.plotly_chart(fig)

    st.header("Model Insights")

    lr_train_mae = mean_absolute_error(
        lr_results_train['y_true'], lr_results_train['y_pred']) * 1e9
    lr_train_r2 = r2_score(
        lr_results_train['y_true'], lr_results_train['y_pred'])

    rf_train_mae = mean_absolute_error(
        rf_results_train['y_true'], rf_results_train['y_pred']) * 1e9
    rf_train_r2 = r2_score(
        rf_results_train['y_true'], rf_results_train['y_pred'])

    performance_table = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "Train MAE": [lr_train_mae, rf_train_mae],
        "Train R²": [lr_train_r2, rf_train_r2],
        "Test MAE": [global_lr_metrics['mae'], global_rf_metrics['mae']]
    })

    st.markdown("##### Model Performance on training set")
    st.table(performance_table.set_index('Model'))

    feature_name_mapping = {
        'block_gas_used_mean_5': 'Block Gas Used (average, 5 prev blocks)',
        'tx_count_lag_1': 'Nb Transactions (prev block)',
        f"q{st.session_state['quantile']}_priority_fee_per_gas_lag_1": f"Q{st.session_state['quantile']} PF/GU (prev block)",
        f"q{st.session_state['quantile']}_priority_fee_per_gas_mean_5": f"Q{st.session_state['quantile']} PF/GU (average, 5 prev blocks)",
        'mean_priority_fee_per_gas_mean_5': 'Mean PF/GU (average, 5 prev blocks)',
        'skew_priority_fee_per_gas_mean_5': 'Skew PF/GU (average, 5 prev blocks)'
    }

    label_feature_importance = [feature_name_mapping[feature]
                                for feature in rf_models.feature_names_in_]

    feature_importances = pd.DataFrame({
        "Feature": label_feature_importance,
        "Importance": rf_models.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=feature_importances["Importance"],
        y=feature_importances["Feature"],
        orientation='h',  # Horizontal bar chart
        marker=dict(color='blue'),  # Optional: Set bar color
        name="Feature Importance"
    ))

    # Update layout for better visualization
    fig.update_layout(
        title="Random Forest Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),  # Ensure descending order
        height=600  # Adjust height for better readability
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    print("finished")
