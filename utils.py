import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go


def compute_metrics(y_results):
    """
    Columns in y_results (already filtered by eligible txs):
    y_true | y_pred | block_number | eligible | tx_topology | gas_used
    """
    # Preconf txs
    preconf_txs = y_results.loc[y_results['y_true'] > y_results['y_pred']]
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_results['y_true'] - y_results['y_pred']))
    
    # MSE (Mean Squared Error)
    mse = np.mean((y_results['y_true'] - y_results['y_pred']) ** 2)

    # Avg of differences (y_true - y_pred)
    avg_preconfirmed_errors = (y_results['y_true'] - y_results['y_pred']).clip(lower=0).mean()

    # Avg nb of preconfirmations (when y_true > y_pred)
    preconfirmations_eligible = (y_results['y_true'] > y_results['y_pred']).mean()

    # Total nb of preconf
    preconfirmations = len(preconf_txs)
    
    # Preconfirmable value
    preconf_value = (preconf_txs['y_true'] * preconf_txs['gas_used']).sum()
    
    # Compute the metrics for each tx_topology group
    grouped_metrics = y_results.groupby('tx_topology').apply(lambda group: {
        'avg_preconfirmed_errors': (group['y_true'] - group['y_pred']).clip(lower=0).mean() * 1e9,
        'preconfirmations_eligible': (group['y_true'] > group['y_pred']).mean(),
        'preconf_value': group.loc[group.y_true > group.y_pred][['y_true', 'gas_used']].prod(axis=1).sum()
    }).to_dict()
    
    metrics = {
        'mae': mae * 1e9,
        'avg_preconfirmed_errors': avg_preconfirmed_errors * 1e9,
        'preconfirmations_eligible': preconfirmations_eligible,
        'preconf_value': preconf_value
    }

    return metrics, grouped_metrics


def compute_lags_stat(df, feature='priority_fee_per_gas', statistic = 'mean', lags = [1, 2, 3, 4, 5]):
    # Validate statistic argument
    valid_statistics = ['mean', 'median', 'q25', 'q75', 'q60']
    
    if statistic not in valid_statistics:
        raise ValueError(f"Invalid statistic '{statistic}'. Choose from {valid_statistics}.")
    
    # Group by block_number and tx_topology to calculate the chosen statistic for priority_fee_per_gas
    if statistic == 'mean':
        agg = (
            df.groupby(['tx_topology', 'block_number'])[feature]
            .mean().to_frame(f'{statistic}_{feature}').sort_index(level=0)
        )
    elif statistic == 'median':
        agg = (
            df.groupby(['tx_topology', 'block_number'])[feature]
            .median()
            .to_frame(f'{statistic}_{feature}').sort_index(level=0)
        )
    elif statistic == 'q25':
        agg = (
            df.groupby(['tx_topology', 'block_number'])[feature]
            .quantile(0.25)
            .to_frame(f'{statistic}_{feature}').sort_index(level=0)
        )
    elif statistic == 'q75':
        agg = (
            df.groupby(['tx_topology', 'block_number'])[feature]
            .quantile(0.75)
            .to_frame(f'{statistic}_{feature}').sort_index(level=0)
        )
    elif statistic == 'q60':
        agg = (
            df.groupby(['tx_topology', 'block_number'])[feature]
            .quantile(0.60)
            .to_frame(f'{statistic}_{feature}').sort_index(level=0)
        )
    
    for lag in lags:
        agg[f'{statistic}_{feature}_lag_{lag}'] = agg.groupby(level='tx_topology')[f'{statistic}_{feature}'].shift(lag)
    
    return agg


def last_block_pf_estimator(df, stat):
    df = df.rename(columns={f'{stat}_priority_fee_per_gas_lag_1': 'y_pred', 'priority_fee_per_gas': 'y_true'})
    return df[['block_number', 'y_true', 'y_pred', 'tx_topology', 'gas_used']].dropna().reset_index(drop=True)


def rolling_mean_block_pf_estimator(df, stat, lags):
    df[f'{stat}_pfpg_rolling_{len(lags)}'] = df[[f'{stat}_priority_fee_per_gas_lag_{l}' for l in lags]].mean(1)
    df = df.rename(columns={f'{stat}_pfpg_rolling_{len(lags)}': 'y_pred', 'priority_fee_per_gas': 'y_true'})
    return df[['block_number', 'y_true', 'y_pred', 'tx_topology', 'gas_used']].dropna().reset_index(drop=True)


def build_block_features(df):
    block_metrics = df.groupby('block_number').agg(
        block_gas_used=('gas_used', 'sum'),
        tx_count=('gas_used', 'size')
    ).reset_index()
    block_metrics['block_gas_used_log_1'] = block_metrics['block_gas_used'].shift(1)
    block_metrics['tx_count_log_1'] = block_metrics['tx_count'].shift(1)
    return block_metrics


def build_agg_features(df, lags=[1, 2, 3, 4, 5]):
    """
    Builds aggregated features for the priority fee per gas for each block and each tx topology.
    
    Args:
        df (DataFrame): A DataFrame containing transaction data with at least 'block_number', 'tx_topology', and 'priority_fee_per_gas' columns.
        lags (list): A list of lags to apply to the aggregated features.
    
    Returns:
        DataFrame: A DataFrame with the original data and the new aggregated features and their respective lags.
    """
    
    # 1️⃣ Compute Quantile Features (all at once)
    quantiles = np.arange(0.1, 1, 0.1)
    quantile_agg = (
        df.groupby(['tx_topology', 'block_number'])['priority_fee_per_gas']
        .quantile(quantiles)
        .unstack(level=-1)
    )
    # Rename quantile columns to reflect quantile names
    quantile_agg.columns = [f'q{int(quantile * 100)}_priority_fee_per_gas' for quantile in quantiles]
    
    # 2️⃣ Compute Additional Aggregate Features (mean, min, max, skew)
    additional_agg = df.groupby(['tx_topology', 'block_number'])['priority_fee_per_gas'].agg(
        mean_priority_fee_per_gas='mean',
        min_priority_fee_per_gas='min',
        max_priority_fee_per_gas='max',
        skew_priority_fee_per_gas='skew'
    )
    
    # 3️⃣ Combine all aggregate features
    agg = pd.concat([quantile_agg, additional_agg], axis=1)
    
    # 4️⃣ Generate Lags for All Columns
    lagged_features = []
    for lag in lags:
        shifted_agg = agg.groupby(level='tx_topology').shift(lag)
        shifted_agg.columns = [f'{col}_lag_{lag}' for col in shifted_agg.columns]
        lagged_features.append(shifted_agg)
    
    # 5️⃣ Concatenate the original features with their lags
    final_agg = pd.concat([agg] + lagged_features, axis=1)
    
    # Reset index to flatten the index into columns
    return final_agg.reset_index()



def train_linear_regression(df, features, features_lag, lags, training_threshold, target='priority_fee_per_gas'):
    """
    Trains a Linear Regression model using lagged features.
    
    Args:
        df (DataFrame): The DataFrame containing all data with features and lags.
        features (list): List of base features to be used as predictors.
        lags (list): List of lags to be applied to each feature.
        training_threshold (int): The block number threshold to split the train and test data.
        target (str): The name of the target variable column (default is 'priority_fee_per_gas').
        
    Returns:
        dict: A dictionary containing the trained model, predictions, and metrics.
    """
    # 1️⃣ Generate the feature columns from features + lags
    feature_columns = [f'{feature}_lag_{lag}' for feature in features_lag for lag in lags]
    feature_columns += features
    
    # Ensure that the required feature columns exist in the DataFrame
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the DataFrame: {missing_features}")
    
    # 2️⃣ Split the data into training and testing sets
    train_data = df[df['block_number'] <= training_threshold]
    test_data = df[df['block_number'] > training_threshold]
    
    # Remove any rows with NaN values (caused by shift() operations)
    train_data = train_data.dropna(subset=feature_columns + [target])
    test_data = test_data.dropna(subset=feature_columns + [target])
    
    # 3️⃣ Extract X (features) and y (target) for training and testing
    X_train = train_data[feature_columns]
    y_train = train_data[target]
    X_test = test_data[feature_columns]
    y_test = test_data[target]
    
    # 4️⃣ Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5️⃣ Make predictions on the test data
    y_pred = model.predict(X_test)
    
    r2 = model.score(X_test, y_test)
    
    # 7️⃣ Return the model, predictions, and metrics
    results = pd.DataFrame()
    results['y_pred'] = y_pred
    results['y_true'] = y_test.reset_index(drop=True)
    results['block_number'] = test_data['block_number'].reset_index(drop=True)
    results['tx_topology'] = test_data['tx_topology'].reset_index(drop=True)
    results['gas_used'] = test_data['gas_used'].reset_index(drop=True)
    return results, r2

def generate_placeholder():
    
    def random_value():
        return np.random.uniform(0.01, 30)

    # Dictionary with random values
    randomized_dict = {
        'Approve': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        },
        'EthTransfer': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        },
        'Other': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        },
        'SetApprovalForAll': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        },
        'Transfer': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        },
        'TransferFrom': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        },
        'TransformERC20': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        },
        'Withdraw': {
            'avg_preconfirmed_errors': random_value(),
            'preconfirmations_eligible': np.random.uniform(0, 1),
            'preconf_value': random_value()
        }
    }

    return randomized_dict

# Function to plot grouped bar chart
def plot_grouped_bar(value_to_plot, q_pl, lr_pl, ml_pl):
    tx_topologies = list(q_pl.keys())
    q_values = [q_pl[tx][value_to_plot] for tx in tx_topologies]
    lr_values = [lr_pl[tx][value_to_plot] for tx in tx_topologies]
    ml_values = [ml_pl[tx][value_to_plot] for tx in tx_topologies]

    trace_q = go.Bar(x=tx_topologies, y=q_values, name='Q PL')
    trace_lr = go.Bar(x=tx_topologies, y=lr_values, name='LR PL')
    trace_ml = go.Bar(x=tx_topologies, y=ml_values, name='ML PL')

    fig = go.Figure(data=[trace_q, trace_lr, trace_ml])
    fig.update_layout(barmode='group')

    return fig

def agg_pf_per_gas_per_position(tx_data):
    aggregated = tx_data.groupby('position').agg(
        mean_priority_fee=('priority_fee_per_gas', 'mean'),
        min_priority_fee=('priority_fee_per_gas', 'min'),
        max_priority_fee=('priority_fee_per_gas', 'max'),
        median_priority_fee=('priority_fee_per_gas', 'median'),
        quantile_25_priority_fee=('priority_fee_per_gas', lambda x: x.quantile(0.25)),
        quantile_75_priority_fee=('priority_fee_per_gas', lambda x: x.quantile(0.75)),
        quantile_95_priority_fee=('priority_fee_per_gas', lambda x: x.quantile(0.95))
    )  
    aggregated *= 1e9
    return aggregated.reset_index()
