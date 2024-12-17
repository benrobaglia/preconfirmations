import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from utils import *

DATA_PATH = "data/"

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Presentation", "Eligible Transactions", "Model Configuration"]
page = st.sidebar.radio("", pages)

# Landing page
if page == "Presentation":
    st.title("Chorus One's Preconfirmation Pricing Model")
    st.markdown("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas pulvinar sapien quam, at vestibulum lacus egestas ut. Maecenas in blandit nisl. Nullam tincidunt, nisi vel blandit varius, nisi tortor hendrerit felis, ac condimentum mauris orci ut magna. In et mattis est. Suspendisse convallis in magna non bibendum. Praesent id nisi blandit, hendrerit lectus nec, ornare odio. Sed velit sapien, eleifend eget nibh eu, aliquam dignissim diam. Morbi viverra diam id est posuere sagittis. Mauris euismod massa sed magna euismod, quis scelerisque dolor vehicula. Aliquam semper lectus sem, quis ullamcorper nisl finibus nec. Quisque vel semper orci, ut viverra tortor. Phasellus varius enim non pharetra viverra. In vehicula vehicula nunc in facilisis. Sed tortor libero, egestas dapibus ullamcorper ut, porttitor et nisi. In eget tellus pharetra odio luctus gravida.
    """)
    

elif page == "Eligible Transactions":
    st.header('Preconfirmable Transactions')
    
    tx_data = pd.read_csv(f'{DATA_PATH}txs_sample.csv')
    aggregated = agg_pf_per_gas_per_position(tx_data)
    
    # Reduce the plot
    # aggregated = aggregated[:300]
        
    st.subheader("Priority Fee Metrics for Each Position")

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

    st.subheader("Preconfirmation Eligibility based on the Block Position")

    default_start_position = 10
    default_end_position = 150
    eligible_txs = tx_data.loc[(tx_data['position'] > default_start_position) & (tx_data['position'] < default_end_position)]


    with st.form(key="eligibility_form"):
        start_position = st.slider('Start Position', min_value=1, max_value=200, value=default_start_position, step=1, help="Set the starting position for eligible transactions.")
        end_position = st.slider('End Position', min_value=1, max_value=200, value=default_end_position, step=1, help="Set the ending position for eligible transactions.")
        submitted = st.form_submit_button("Run")
    
    if submitted:
        eligible_txs = tx_data.loc[(tx_data['position'] > start_position) & (tx_data['position'] < end_position)]
                
    st.header("Key Metrics")
    percentage_of_eligible_txs = len(eligible_txs) / len(tx_data) if len(tx_data) > 0 else 0
    st.metric(
        label="Percentage of Eligible Transactions for Preconfirmations", 
        value=f"{percentage_of_eligible_txs:.2%}", 
        help="This shows the percentage of transactions that are eligible for preconfirmations based on the position range."
    )

    st.header("Priority Fee Distribution for Eligible Transactions")

    filtered_priority_fees = eligible_txs['priority_fee_per_gas'] * 1e9  # Convert to Gwei

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=filtered_priority_fees,
        nbinsx=200,  # Number of bins
        marker_color='rgb(26, 118, 255)',
        opacity=0.75
    ))

    fig.update_layout(
        title="Distribution of Priority Fee per Gas",
        xaxis_title="Priority Fee per Gas (in Gwei)",
        yaxis_title="Count",
        title_x=0.5,  # Center the title
        bargap=0.2,  # Space between bars
        bargroupgap=0.1  # Space between groups (for multiple histograms)
    )

    # Streamlit app
    st.plotly_chart(fig)


    csv = eligible_txs.to_csv(index=False)
    st.download_button(
        label="Download Eligible Transactions as CSV",
        data=csv,
        file_name='eligible_transactions.csv',
        mime='text/csv',
    )

    

# 3️⃣ Model Configuration Page
elif page == "Model Configuration":
    st.header("3. Model Configurations")
    
    lags = np.arange(1, 33, 1)
    eligible_txs = pd.read_csv(f'{DATA_PATH}eligible_txs_sample.csv')
    block_features = pd.read_csv(f'{DATA_PATH}block_features.csv')
    agg_df = build_agg_features(eligible_txs, lags)
    final_df = eligible_txs.merge(agg_df, on=['block_number', 'tx_topology']).merge(block_features, on='block_number')

    # Initialize session state for model configuration
    if 'quantile' not in st.session_state:
        st.session_state['quantile'] = 50
    if 'max_window' not in st.session_state:
        st.session_state['max_window'] = 10
    if 'lr_features' not in st.session_state:
        st.session_state['lr_features'] = ['q50_priority_fee_per_gas']
    if 'max_window_lr' not in st.session_state:
        st.session_state['max_window_lr'] = (1, 10)
    if 'lr_training_threshold' not in st.session_state:
        st.session_state['lr_training_threshold'] = 21331500

    tabs = st.tabs(["Quantile Heuristic", "Linear Regression", "Neural Network"])
    
    with tabs[0]:  # Quantile Heuristic Configuration
        st.subheader("Configure Quantile Heuristic Model")
        
        # Define the widgets and their callbacks
        quantile = st.selectbox(
            "Select Quantile", 
            [i for i in range(10, 100, 10)], 
            index=[i for i in range(10, 100, 10)].index(st.session_state['quantile']),
            key="quantile",
            on_change=lambda: st.session_state.update({'quantile': st.session_state['quantile']})
        )

        max_window = st.slider(
            "Select Maximum Window Size", 
            min_value=1, 
            max_value=32, 
            value=st.session_state['max_window'], 
            key="max_window", 
            on_change=lambda: st.session_state.update({'max_window': st.session_state['max_window']})
        )

        if st.button("Save Configuration for Quantile Heuristic"):
            st.success(f"Configuration for Quantile Heuristic model saved! (Quantile: {st.session_state['quantile']}, Max Window: {st.session_state['max_window']})")
            
    with tabs[1]:  # Linear Regression Configuration
        st.subheader("Configure Linear Regression Model")

        # Define the widgets and their callbacks
        feats = [f'q{q}_priority_fee_per_gas' for q in range(10, 100, 10)]
        feats2 = ['mean_priority_fee_per_gas', 'min_priority_fee_per_gas', 'max_priority_fee_per_gas', 'skew_priority_fee_per_gas']
        indiv = ['gas_used']
        blocks = ['block_gas_used_log_1']
        
        features = st.multiselect(
            "Select Features", 
            feats+feats2+indiv+blocks, 
            default=st.session_state['lr_features'], 
            key='lr_features', 
            on_change=lambda: st.session_state.update({'lr_features': st.session_state['lr_features']})
        )

        lags = st.slider(
            "Select Max Window Size for the Features", 
            min_value=1, 
            max_value=32,
            value=st.session_state['max_window_lr'],
            key='max_window_lr', 
            on_change=lambda: st.session_state.update({'max_window_lr': st.session_state['max_window_lr']})
        )


        training_threshold = st.slider(
            "Select Training Threshold", 
            min_value=21330584, 
            max_value=21332578, 
            value=st.session_state['lr_training_threshold'], 
            key='lr_training_threshold', 
            on_change=lambda: st.session_state.update({'lr_training_threshold': st.session_state['lr_training_threshold']})
        )

        if st.button("Save Configuration for Linear Regression"):
            st.success("Configuration for Linear Regression model saved!")

    with tabs[2]:  # Quantile Heuristic Configuration
        st.subheader("Configure Neural Network Model")

    print(f"Quantiles: {st.session_state['quantile']}")
    print(f"Max Window: {st.session_state['max_window']}")
    print(f"lr_features: {st.session_state['lr_features']}")
    print(f"max_window_lr: {st.session_state['max_window_lr']}")
    print(f"lr_training_threshold: {st.session_state['lr_training_threshold']}")
    
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.header("Simulation Results")
    
    quantile_lags = np.arange(1, st.session_state['max_window'])
    quantile_estimator = rolling_mean_block_pf_estimator(final_df, f"q{st.session_state['quantile']}", quantile_lags)
    global_quantile_metrics, groups_quantile_metrics = compute_metrics(quantile_estimator)

    global_lr_metrics = {
        'mae': np.random.uniform(0.5, 2.5),
        'avg_preconfirmed_errors': np.random.uniform(0.5, 2.5),
        'preconfirmations_eligible': np.random.rand(),
        'preconf_value': np.random.uniform(10, 50)
    }
    global_nn_metrics = {
        'mae': np.random.uniform(0.5, 2.5),
        'avg_preconfirmed_errors': np.random.uniform(0.5, 2.5),
        'preconfirmations_eligible': np.random.rand(),
        'preconf_value': np.random.uniform(10, 50)
    }
    global_results = [global_quantile_metrics, global_lr_metrics, global_nn_metrics]
    
    global_results = pd.DataFrame(global_results, index=['Quantile Heuristic', 'Linear Regression', 'Neural Network'])
    
    st.dataframe(global_results)
    
    # Plot results

    # Extract data for plotting
    lr_pl = generate_placeholder()
    ml_pl = generate_placeholder()
    
    # Streamlit app
    st.title('Grouped Bar Chart')
    value_to_plot = st.selectbox('Select a value to plot', ['avg_preconfirmed_errors', 'preconfirmations_eligible', 'preconf_value'])
    st.write('You selected:', value_to_plot)

    fig = plot_grouped_bar(value_to_plot, groups_quantile_metrics, lr_pl, ml_pl)
    st.plotly_chart(fig)
    print("finished")