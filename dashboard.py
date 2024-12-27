import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from utils import *
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/"

# Change the bin legend of the histogram (start to end instead of middle)
# Mention the Sample size in the text 
# Fix the target for all models
# change RF parameters in the dashboard
# Show training set vs testing set

# Sidebar Navigation
st.sidebar.image("logo.svg", width=200)
st.sidebar.title("Navigation")
pages = ["Presentation", "Bias Reducer", "Simulations"]
page = st.sidebar.radio("", pages)

# Landing page
if page == "Presentation":
    st.title("📈 Preconfirmation Pricing Dashboard")
    st.subheader("Developed by Chorus One")

    st.markdown("""
        ### 📘 **Introduction**
        **Preconfirmations** are a hot topic in Ethereum research aimed at improving the user experience.  
        A **preconfirmation** (or **preconf**) is a credible, signed commitment from an Ethereum block proposer guaranteeing that a specific transaction will be included in a future block.

        However, **pricing preconfirmations is a complex problem**. One critical element in this process is the ability to estimate the **future priority fee (PF)**.  
        If a transaction's PF is too low compared to future fees, preconfirming it may result in **a lower expected value for the builder and proposer**.  

        Additionally, the **pricing mechanism for preconfirmations is still under discussion**, as the conditions for **slashing and penalties** have not yet been fully defined.  
        This uncertainty makes it challenging to establish a reliable **P&L model** and conduct a realistic **backtest** since **preconfirmations are not live on Ethereum**.  
        Therefore, the primary focus of this dashboard is to provide accurate predictions for the **priority fee per gas unit**, which serves as a key input for any potential preconfirmation pricing strategy.
        """)

    # Key Features
    st.markdown("""
    ### 🔍 **Key Features**
    - 🛠️ **Estimator Bias Reducer**: Not all transactions are equal. Some can bias the estimators. 
    With this feature, you can select the approach for **choosing which transactions** to include in the model, ensuring a more robust estimator with minimal bias.

    - 📉 **Dynamic Fee Predictions**: Use machine learning models to predict **future priority fees**. 
    These predictions help determine which transactions are worth preconfirming.

    - 📊 **Customizable Estimators**: Explore preconfirmation pricing using multiple model approaches, 
    including a **Quantile Heuristic**, **Linear Regression**, and **Machine Learning model**.

    - ⚙️ **Interactive Configurations**: Adjust model parameters to see how they affect **priority fees, and preconfirmations**.

    """)


    
elif page == "Bias Reducer":
    st.title('Estimator Bias Reducer')
    
    st.markdown("""
                The Priority Fee per Gas Unit is highly influenced by a transaction's position within the block (as shown in the interactive figure below).
                To reduce estimator bias, we exclude outliers, especially those caused by "toxic flow" (cite paper link).
                
                This section offers an interactive tool to explore and customize the selection of transaction positions, enabling users to build a more robust and unbiased estimator.
                """)

    
    tx_data = pd.read_csv(f'{DATA_PATH}txs_sample.csv')
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
            traces.append(go.Scatter(mode='lines', name=metric, x=positions, y=values, visible='legendonly'))
            
        else:
            traces.append(go.Scatter(mode='lines', name=metric, x=positions, y=values))

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
    eligible_txs = tx_data.loc[(tx_data['position'] > default_start_position) & (tx_data['position'] < default_end_position)]


    with st.form(key="eligibility_form"):
        start_position = st.slider('Start Position', min_value=1, max_value=300, value=default_start_position, step=1, help="Set the starting position for eligible transactions.")
        end_position = st.slider('End Position', min_value=50, max_value=400, value=default_end_position, step=1, help="Set the ending position for eligible transactions.")
        submitted = st.form_submit_button("Run")
    
    if submitted:
        eligible_txs = tx_data.loc[(tx_data['position'] > start_position) & (tx_data['position'] < end_position)]
                
    st.header("Results")
    percentage_of_eligible_txs = len(eligible_txs) / len(tx_data) if len(tx_data) > 0 else 0
    st.metric(
        label="Percentage of Transactions Left to Price Preconfirmations", 
        value=f"{percentage_of_eligible_txs:.2%}", 
        help="This shows the percentage of transactions that are in the position range. This is the percentage of transactions that are left to price preconfirmations."
    )

    st.subheader("Priority Fee Distribution of the Selected Set")

    filtered_priority_fees = eligible_txs['priority_fee_per_gas'] * 1e9  # Convert to Gwei

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
elif page == "Simulations":
    st.title("Simulations")
    
    st.markdown(r"""
        This section allows testing of different model configurations to predict a **priority fee per gas unit (PF/GU) threshold** for pre-confirming Ethereum transactions.

        ### **Problem Formulation**
        We aim to predict a threshold $T$ for **PF/GU**. A transaction $tx$ is pre-confirmed if: $\frac{PF_{tx}}{GU_{tx}} > T$
        
        We denote $\rho_{tx}(T) = \frac{PF_{tx}}{GU_{tx}}- T$ the preconfirmation error. The goal is to accurately estimate $T$ to minimize pre-confirmation errors while maximizing the number of preconfirmed transactions.

        ### **Models**
        - **Quantile Heuristic (QH)**: Uses a quantile of past PF/GU values as the threshold.
        - **Linear Regression (LR)**: Predicts $T$ using a linear model of historical transaction data.
        - **Random Forest (RF)**: Uses an ensemble of decision trees to capture non-linear patterns.

        ### **Metrics**
        - **Mean Absolute Error (MAE)**: Measures the average deviation between predicted $\hat{T}$ and actual $T$:

        $\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| T_i - \hat{T}_i \right| $
        - **Average Preconfirmed Error**: Captures losses from incorrect pre-confirmations:
        $\text{Error} = \frac{1}{N} \sum_{i=1}^N \rho_{i}(\hat{T}_i)$
        - **Percentage of Eligible Transactions**: Tracks the share of transactions that meet the pre-confirmation condition.

        """)

    st.subheader("Model Configuration")
    eligible_txs = pd.read_csv(f'{DATA_PATH}eligible_txs_sample.csv')
    block_features = pd.read_csv(f'{DATA_PATH}block_features.csv')
    agg_df = build_agg_features(eligible_txs, np.arange(1, 33, 1)).merge(block_features, on='block_number')
    final_df = eligible_txs.merge(agg_df, on=['block_number', 'tx_topology'])
    
    col_choices = [c for c in agg_df.columns if 'lag' not in c]
    col_choices.remove('tx_count')
    col_choices.remove('block_gas_used')
    col_choices.remove('block_number')
    col_choices.remove('tx_topology')
    col_choices.append('tx_count_lag_1')
    col_choices.append('block_gas_used_lag_1')

    # Initialize session state for model configuration
    session_defaults = {
        'quantile': 50,
        'max_window': 10,
        'lr_features': ['block_gas_used_lag_1'],
        'rf_features': ['block_gas_used_lag_1', 'tx_count_lag_1', 'mean_priority_fee_per_gas', 'min_priority_fee_per_gas', 'max_priority_fee_per_gas', 'skew_priority_fee_per_gas'],
        'lr_lags': 10,
        'rf_lags': 5,
        'training_threshold': 21331500,
        'n_estimators': 100,
        'max_depth': 10
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    tabs = st.tabs(["Quantile Heuristic", "Linear Regression", "Random Forest Regression"])

    # Quantile Heuristic Configuration
    with tabs[0]:  
        st.subheader("Configure Quantile Heuristic")
        
        temp_quantile = st.selectbox(
            "Select Quantile", 
            [i for i in range(10, 100, 10)], 
            index=[i for i in range(10, 100, 10)].index(st.session_state['quantile']),
        )

        temp_max_window = st.slider(
            "Select Maximum Window Size", 
            min_value=1, 
            max_value=32, 
            value=st.session_state['max_window']
        )

        if st.button("Save Configuration for Quantile Heuristic"):
            st.session_state['quantile'] = temp_quantile
            st.session_state['max_window'] = temp_max_window
            st.success(f"Configuration for Quantile Heuristic model saved! (Quantile: {st.session_state['quantile']}, Max Window: {st.session_state['max_window']})")

    # Linear Regression Configuration
    with tabs[1]:  
        st.subheader("Configure Linear Regression Model")

        temp_lr_features = st.multiselect(
            "Select Features", 
            col_choices, 
            default=st.session_state['lr_features']
        )

        temp_lr_lags = st.selectbox(
            'Target lags to include as features for regression:', 
            options=np.arange(1, 33, 1),
            index=st.session_state['lr_lags'] - 1,
        )

        temp_training_threshold = st.slider(
            "Select Training Threshold", 
            min_value=21330584, 
            max_value=21332578, 
            value=st.session_state['training_threshold']
        )

        if st.button("Save Configuration for Linear Regression"):
            st.session_state['lr_features'] = temp_lr_features
            st.session_state['lr_lags'] = temp_lr_lags
            st.session_state['training_threshold'] = temp_training_threshold
            st.success("Configuration for Linear Regression model saved!")

    # Random Forest Configuration
    with tabs[2]: 
        st.subheader("Configure Machine Learning Model")

        temp_rf_features = st.multiselect(
            "Target lags to include as features for regression", 
            col_choices, 
            default=st.session_state['rf_features']
        )
    
        temp_rf_lags = st.selectbox(
            'Select a lag from the list below:', 
            options=np.arange(1, 33, 1),
            index=st.session_state['rf_lags']-1,
        )
        
        temp_n_estimators = st.slider(
            "Number of trees (n_estimators):", 
            min_value=10, 
            max_value=500, 
            value=st.session_state['n_estimators'], 
            step=10
        )

        temp_max_depth = st.slider(
            "Maximum tree depth (max_depth):", 
            min_value=1, 
            max_value=50, 
            value=st.session_state['max_depth'], 
            step=1
        )

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
            st.session_state['rf_features'] = temp_rf_features
            st.session_state['rf_lags'] = temp_rf_lags
            st.session_state['n_estimators'] = temp_n_estimators
            st.session_state['max_depth'] = temp_max_depth
            st.success("Configuration for Random Forest model saved!")


    print(f"Quantiles: {st.session_state['quantile']}")
    print(f"Max Window: {st.session_state['max_window']}")
    print(f"lr_features: {st.session_state['lr_features']}")
    print(f"rf_features: {st.session_state['rf_features']}")
    print(f"lr_lags: {st.session_state['lr_lags']}")
    print(f"rf_lags: {st.session_state['rf_lags']}")
    print(f"training_threshold: {st.session_state['training_threshold']}")
    print(f"n_estimators: {st.session_state['n_estimators']}")
    print(f"max_depth: {st.session_state['max_depth']}")
    
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.header("Simulation Results")
    
    quantile_lags = np.arange(1, st.session_state['max_window']+1)
    quantile_estimator = rolling_mean_block_pf_estimator(final_df.loc[final_df.block_number > st.session_state['training_threshold']],
                                                         f"q{st.session_state['quantile']}",
                                                         quantile_lags)
    global_quantile_metrics, groups_quantile_metrics = compute_metrics(quantile_estimator)
    # global_rolling_quantile_metrics = compute_rolling_metrics(y_results=quantile_estimator)
    
    # LR
    target = f'q{st.session_state["quantile"]}_priority_fee_per_gas'
    target_lr_features = build_lag_features(target, np.arange(1, st.session_state['lr_lags']+1))
    total_lr_features = target_lr_features + st.session_state['lr_features']
    lr_results, lr_results_train, lr_models = train_linear_regression_one_hot(agg_df,
                                                      total_lr_features,
                                                      st.session_state['training_threshold'],
                                                      target=target)
    
    lr_estimator = final_df[['block_number', 'tx_topology', 'gas_used', 'priority_fee_per_gas']].merge(lr_results, on=['block_number', 'tx_topology'])
    print(len(lr_estimator.block_number.unique()))
    global_lr_metrics, groups_lr_metrics = compute_metrics(lr_estimator)
    # global_rolling_lr_metrics = compute_rolling_metrics(y_results=lr_estimator)

    # RF
    target_rf_features = build_lag_features(target, np.arange(1, st.session_state['rf_lags']+1))
    total_rf_features = target_rf_features + st.session_state['rf_features']

    params = {"n_estimators": st.session_state['n_estimators'],
              "max_depth": st.session_state['max_depth']}

    rf_results, rf_results_train, rf_models = train_random_forest_one_hot(agg_df,
                                                        total_rf_features,
                                                        st.session_state['training_threshold'],
                                                        target=target,
                                                        params=params
                                                        )

    print(f"Here are the targets and features \n\n")
    print(f"Target: {target}")
    print(f"RF Features: {total_rf_features}")
    print(f"LR Features: {total_lr_features}")
    
    rf_estimator = final_df[['block_number', 'tx_topology', 'gas_used', 'priority_fee_per_gas']].merge(rf_results, on=['block_number', 'tx_topology'])
    global_rf_metrics, groups_rf_metrics = compute_metrics(rf_estimator)
    # global_rolling_rf_metrics = compute_rolling_metrics(y_results=rf_estimator)

    global_results = [global_quantile_metrics, global_lr_metrics, global_rf_metrics]
    
    global_results = pd.DataFrame(global_results, index=['Quantile Heuristic', 'Linear Regression', 'Random Forest'])
    global_results.drop(columns=['preconf_value'], inplace=True)
    global_results.rename(columns={'mae': 'Mean Absolute Error', "avg_preconfirmed_errors": 'Avg Preconfirmed Error', 'preconfirmations_eligible': "Eligible Transactions (%)"}, inplace=True)
    
    st.dataframe(global_results)
    
    # Plot results
    # Plot global rolling metrics
    
    # metric_to_plot = st.selectbox(
    #     'Select a metric to plot', 
    #     options=['expanding_mae', 'expanding_preconf_value', 'expanding_preconfirmations_eligible'], 
    #     index=0
    # )

    # fig = go.Figure()

    # fig.add_trace(go.Scatter(
    #     x=global_rolling_quantile_metrics['block_number'], 
    #     y=global_rolling_quantile_metrics[metric_to_plot], 
    #     mode='lines', 
    #     name=f'Quantile - {metric_to_plot}'
    # ))

    # fig.add_trace(go.Scatter(
    #     x=global_rolling_lr_metrics['block_number'], 
    #     y=global_rolling_lr_metrics[metric_to_plot], 
    #     mode='lines', 
    #     name=f'LR - {metric_to_plot}',
    #     line=dict(dash='dash')
    # ))

    # fig.add_trace(go.Scatter(
    #     x=global_rolling_rf_metrics['block_number'], 
    #     y=global_rolling_rf_metrics[metric_to_plot], 
    #     mode='lines', 
    #     name=f'RF - {metric_to_plot}',
    #     line=dict(dash='dot')
    # ))

    # fig.update_layout(
    #     title=f"Plot of {metric_to_plot} Over Blocks",
    #     xaxis_title="Block Number",
    #     yaxis_title=metric_to_plot.replace('_', ' ').title(),
    #     title_x=0.0,
    #     height=600
    # )

    # st.plotly_chart(fig)


    st.header("Results per Tranaction Topology")
    value_to_plot = st.selectbox('Select a value to plot', ['Avg Preconfirmed Error', 'Eligible Transactions (%)'])
    # value_to_plot = st.selectbox('Select a value to plot', ['avg_preconfirmed_errors', 'preconfirmations_eligible', 'preconf_value'])
    # st.write('You selected:', value_to_plot)
    idxer = {'Avg Preconfirmed Error': 'avg_preconfirmed_errors', 'Eligible Transactions (%)': 'preconfirmations_eligible'}
    fig = plot_grouped_bar(idxer[value_to_plot], groups_quantile_metrics, groups_lr_metrics, groups_rf_metrics)
    st.plotly_chart(fig)
        
    st.header("Model Insights")

    lr_train_mae = mean_absolute_error(lr_results_train['y_true'], lr_results_train['y_pred'])
    lr_train_r2 = r2_score(lr_results_train['y_true'], lr_results_train['y_pred'])

    rf_train_mae = mean_absolute_error(rf_results_train['y_true'], rf_results_train['y_pred'])
    rf_train_r2 = r2_score(rf_results_train['y_true'], rf_results_train['y_pred'])

    performance_table = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "Train MAE": [lr_train_mae, rf_train_mae],
        "Train R²": [lr_train_r2, rf_train_r2],
        "Test MAE": [global_lr_metrics['mae'], global_rf_metrics['mae']]
    })

    st.markdown("##### Model Performance on training set")
    st.table(performance_table.set_index('Model'))


    feature_importances = pd.DataFrame({
        "Feature": rf_models.feature_names_in_,
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
