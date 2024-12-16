import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

DATA_PATH = "data/"

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Presentation", "Eligibility Transactions", "Model Configuration"]
page = st.sidebar.radio("", pages)

# Landing page
if page == "Presentation":
    st.title("Chorus One's Preconfirmation Pricing Model")
    st.markdown("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas pulvinar sapien quam, at vestibulum lacus egestas ut. Maecenas in blandit nisl. Nullam tincidunt, nisi vel blandit varius, nisi tortor hendrerit felis, ac condimentum mauris orci ut magna. In et mattis est. Suspendisse convallis in magna non bibendum. Praesent id nisi blandit, hendrerit lectus nec, ornare odio. Sed velit sapien, eleifend eget nibh eu, aliquam dignissim diam. Morbi viverra diam id est posuere sagittis. Mauris euismod massa sed magna euismod, quis scelerisque dolor vehicula. Aliquam semper lectus sem, quis ullamcorper nisl finibus nec. Quisque vel semper orci, ut viverra tortor. Phasellus varius enim non pharetra viverra. In vehicula vehicula nunc in facilisis. Sed tortor libero, egestas dapibus ullamcorper ut, porttitor et nisi. In eget tellus pharetra odio luctus gravida.
    """)

elif page == "Eligibility Transactions":
    st.header('Preconfirmable Transactions')
    
    with st.form(key="eligibility_form"):
        start_position = st.slider('Start Position', min_value=1, max_value=200, value=10, step=1, help="Set the starting position for eligible transactions.")
        end_position = st.slider('End Position', min_value=1, max_value=200, value=150, step=1, help="Set the ending position for eligible transactions.")
        submitted = st.form_submit_button("Run")
    
    if submitted:
        tx_data = pd.read_csv(f'{DATA_PATH}txs_sample.csv')
        eligible_txs = tx_data.loc[(tx_data['position'] > start_position) & (tx_data['position'] < end_position)]
        
        st.header("Key Metrics")
        percentage_of_eligible_txs = len(eligible_txs) / len(tx_data) if len(tx_data) > 0 else 0
        st.metric(label="Percentage of Eligible Transactions for Preconfirmations", value=f"{percentage_of_eligible_txs:.2%}", help="This shows the percentage of transactions that are eligible for preconfirmations based on the position range.")
        
        st.header("Priority Fee Distribution for Eligible Transactions")
        fig, ax = plt.subplots()
        sns.histplot(eligible_txs['priority_fee_per_gas'].loc[eligible_txs['priority_fee_per_gas'] < 10e-9] * 1e9, bins=20, ax=ax)
        ax.set_xlabel("Priority Fee per Gas (in Gwei)")
        ax.set_title("Distribution of Priority Fee per Gas")
        st.pyplot(fig)
        
        csv = eligible_txs.to_csv(index=False)
        st.download_button(
            label="Download Eligible Transactions as CSV",
            data=csv,
            file_name='eligible_transactions.csv',
            mime='text/csv',
        )


# 3️⃣ Model Configuration Page
elif page == "Model Configuration":
    st.header("3. Models Configuration")
    
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
    