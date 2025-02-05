import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
# from tqdm import tqdm
# import zipfile
# import pickle


def compute_metrics(y_results):
    """
    Columns in y_results (already filtered by eligible txs):
    priority_fee_per_gas | y_true | y_pred | block_number | eligible | tx_topology | gas_used
    """
    # Preconf txs
    preconf_txs = y_results.loc[y_results['priority_fee_per_gas']
                                > y_results['y_pred']]

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_results['y_true'] - y_results['y_pred']))

    # MSE (Mean Squared Error)
    # mse = np.mean((y_results['y_true'] - y_results['y_pred']) ** 2)

    # Avg of differences (y_true - y_pred)
    avg_preconfirmed_errors = (
        y_results['priority_fee_per_gas'] - y_results['y_pred']).clip(lower=0).mean()

    # Avg nb of preconfirmations (when y_true > y_pred)
    preconfirmations_eligible = (
        y_results['priority_fee_per_gas'] > y_results['y_pred']).mean()

    # Preconfirmable value
    preconf_value = (preconf_txs['priority_fee_per_gas']
                     * preconf_txs['gas_used']).sum()

    # Compute the metrics for each tx_topology group
    grouped_metrics = y_results.groupby('tx_topology').apply(lambda group: {
        'avg_preconfirmed_errors': (group['priority_fee_per_gas'] - group['y_pred']).clip(lower=0).mean() * 1e9,
        'preconfirmations_eligible': (group['priority_fee_per_gas'] > group['y_pred']).mean(),
        'preconf_value': group.loc[group.priority_fee_per_gas > group.y_pred][['priority_fee_per_gas', 'gas_used']].prod(axis=1).sum()
    }).to_dict()

    metrics = {
        'mae': mae * 1e9,
        'avg_preconfirmed_errors': avg_preconfirmed_errors * 1e9,
        'preconfirmations_eligible': preconfirmations_eligible*100,
        'preconf_value': preconf_value
    }

    return metrics, grouped_metrics


def compute_rolling_metrics(y_results):
    y_results = y_results.sort_values(by='block_number').reset_index(drop=True)
    y_results['abs_error'] = np.abs(
        y_results['priority_fee_per_gas'] * 1e9 - y_results['y_pred'] * 1e9)
    y_results['preconf_value'] = np.clip(
        y_results['priority_fee_per_gas'] - y_results['y_pred'], 0, None)
    y_results['preconf_eligible'] = (
        y_results['priority_fee_per_gas'] > y_results['y_pred']).astype(int)
    y_results['cumulative_abs_error'] = y_results['abs_error'].cumsum()
    y_results['cumulative_preconf_value'] = y_results['preconf_value'].cumsum()
    y_results['cumulative_preconf_eligible'] = y_results['preconf_eligible'].cumsum()
    y_results['cumulative_transaction_count'] = np.arange(
        1, len(y_results) + 1)
    block_agg = y_results.groupby('block_number').last()

    block_agg['expanding_mae'] = block_agg['cumulative_abs_error'] / \
        block_agg['cumulative_transaction_count']
    block_agg['expanding_preconf_value'] = block_agg['cumulative_preconf_value']
    block_agg['expanding_preconfirmations_eligible'] = block_agg['cumulative_preconf_eligible'] / \
        block_agg['cumulative_transaction_count']

    rolling_metrics = block_agg[['expanding_mae', 'expanding_preconf_value',
                                 'expanding_preconfirmations_eligible']].reset_index()

    return rolling_metrics


def compute_lags_stat(df, feature='priority_fee_per_gas', statistic='mean', lags=[1, 2, 3, 4, 5]):
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
        agg[f'{statistic}_{feature}_lag_{lag}'] = agg.groupby(
            level='tx_topology')[f'{statistic}_{feature}'].shift(lag)

    return agg


def last_block_pf_estimator(df, stat):
    df = df.rename(columns={
                   f'{stat}_priority_fee_per_gas_lag_1': 'y_pred', 'priority_fee_per_gas': 'y_true'})
    return df[['block_number', 'y_true', 'y_pred', 'tx_topology', 'gas_used']].dropna().reset_index(drop=True)


def rolling_mean_block_pf_estimator(df, stat, lags):
    df[f'{stat}_pfpg_rolling_{len(lags)}'] = df[[
        f'{stat}_priority_fee_per_gas_lag_{l}' for l in lags]].mean(1)
    df['y_true'] = df[f'{stat}_priority_fee_per_gas']
    df = df.rename(columns={f'{stat}_pfpg_rolling_{len(lags)}': 'y_pred'})
    return df[['block_number', 'priority_fee_per_gas', 'y_true', 'y_pred', 'tx_topology', 'gas_used']].dropna().reset_index(drop=True)


def build_block_features(df):
    block_metrics = df.groupby('block_number').agg(
        block_gas_used=('gas_used', 'sum'),
        tx_count=('gas_used', 'size')
    ).reset_index()
    block_metrics['block_gas_used_lag_1'] = block_metrics['block_gas_used'].shift(
        1)
    block_metrics['tx_count_lag_1'] = block_metrics['tx_count'].shift(1)
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
    quantile_agg.columns = [
        f'q{int(quantile * 100)}_priority_fee_per_gas' for quantile in quantiles]

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


def build_agg_features2(df, lags=[1, 2, 3, 4, 5]):
    quantiles = np.arange(0.1, 1, 0.1)
    quantile_agg = (
        df.groupby('block_number')['priority_fee_per_gas']
        .quantile(quantiles)
        .unstack(level=-1)
    )
    quantile_agg.columns = [
        f'q{int(quantile * 100)}_priority_fee_per_gas' for quantile in quantiles]

    additional_agg = df.groupby('block_number')['priority_fee_per_gas'].agg(
        mean_priority_fee_per_gas='mean',
        min_priority_fee_per_gas='min',
        max_priority_fee_per_gas='max',
        skew_priority_fee_per_gas='skew'
    )

    agg = pd.concat([quantile_agg, additional_agg], axis=1)

    lagged_features = []
    for lag in lags:
        shifted_agg = agg.shift(lag)
        shifted_agg.columns = [f'{col}_lag_{lag}' for col in shifted_agg.columns]
        lagged_features.append(shifted_agg)

    final_agg = pd.concat([agg] + lagged_features, axis=1)

    return final_agg.reset_index()


def train_linear_regression_one_hot(df_agg, features, training_threshold, target='q50_priority_fee_per_gas'):
    df = pd.get_dummies(df_agg, columns=['tx_topology'], drop_first=True)
    df['tx_topology'] = df_agg['tx_topology']

    feature_columns = features + \
        [col for col in df.columns if 'tx_topology_' in col]

    missing_features = [
        col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the DataFrame: {missing_features}")

    train_data = df[df['block_number'] <= training_threshold].dropna(
        subset=feature_columns + [target])
    test_data = df[df['block_number'] > training_threshold].dropna(
        subset=feature_columns + [target])

    X_train = train_data[feature_columns]
    y_train = train_data[target]
    X_test = test_data[feature_columns]
    y_test = test_data[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results_train = pd.DataFrame()
    results_train['y_pred'] = y_train_pred
    results_train['y_true'] = y_train.reset_index(drop=True)
    results_train['block_number'] = train_data['block_number'].reset_index(
        drop=True)
    results_train['tx_topology'] = train_data['tx_topology'].reset_index(
        drop=True)

    results_test = pd.DataFrame()
    results_test['y_pred'] = y_test_pred
    results_test['y_true'] = y_test.reset_index(drop=True)
    results_test['block_number'] = test_data['block_number'].reset_index(
        drop=True)
    results_test['tx_topology'] = test_data['tx_topology'].reset_index(
        drop=True)

    return results_test, results_train, model


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


def plot_grouped_bar(value_to_plot, q_pl, lr_pl, ml_pl, ub_pl):
    tx_topologies = list(set(q_pl.keys()) & set(
        ml_pl.keys()) & set(lr_pl.keys()) & set(ub_pl.keys()))
    q_values = [q_pl[tx][value_to_plot] for tx in tx_topologies]
    lr_values = [lr_pl[tx][value_to_plot] for tx in tx_topologies]
    ml_values = [ml_pl[tx][value_to_plot] for tx in tx_topologies]
    ub_values = [ub_pl[tx][value_to_plot] for tx in tx_topologies]

    trace_q = go.Bar(x=tx_topologies, y=q_values, name='GETH')
    trace_lr = go.Bar(x=tx_topologies, y=lr_values, name='LR')
    trace_ml = go.Bar(x=tx_topologies, y=ml_values, name='RF')
    trace_ub = go.Bar(x=tx_topologies, y=ub_values, name='UB')

    fig = go.Figure(data=[trace_q, trace_lr, trace_ml, trace_ub])
    fig.update_layout(barmode='group')

    return fig


def agg_pf_per_gas_per_position(tx_data):
    aggregated = tx_data.groupby('position').agg(
        mean_priority_fee=('priority_fee_per_gas', 'mean'),
        min_priority_fee=('priority_fee_per_gas', 'min'),
        max_priority_fee=('priority_fee_per_gas', 'max'),
        median_priority_fee=('priority_fee_per_gas', 'median'),
        quantile_25_priority_fee=(
            'priority_fee_per_gas', lambda x: x.quantile(0.25)),
        quantile_75_priority_fee=(
            'priority_fee_per_gas', lambda x: x.quantile(0.75)),
        quantile_95_priority_fee=(
            'priority_fee_per_gas', lambda x: x.quantile(0.95))
    )
    aggregated *= 1e9
    return aggregated.reset_index()


def train_random_forest_one_hot(df_agg, features, training_threshold, target='q50_priority_fee_per_gas', params={'n_estimators': 100, "max_depth": 10}, use_topo=True):
    if use_topo:
        df = pd.get_dummies(df_agg, columns=['tx_topology'], drop_first=True)
        df['tx_topology'] = df_agg['tx_topology']

        feature_columns = features + \
            [col for col in df.columns if 'tx_topology_' in col]
    else:
        feature_columns = features
        df = df_agg.copy()
    df[[c for c in df.columns if 'priority_fee_per_gas' in c]] *= 1e9

    missing_features = [
        col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the DataFrame: {missing_features}")

    train_data = df[df['block_number'] <= training_threshold].dropna(
        subset=feature_columns + [target])
    test_data = df[df['block_number'] > training_threshold].dropna(
        subset=feature_columns + [target])

    X_train = train_data[feature_columns]
    y_train = train_data[target]
    X_test = test_data[feature_columns]
    y_test = test_data[target]

    model = RandomForestRegressor(
        n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results_train = pd.DataFrame()
    results_train['y_pred'] = y_train_pred * 1e-9
    results_train['y_true'] = y_train.reset_index(drop=True) * 1e-9
    results_train['block_number'] = train_data['block_number'].reset_index(
        drop=True)
    if use_topo:
        results_train['tx_topology'] = train_data['tx_topology'].reset_index(
            drop=True)

    results_test = pd.DataFrame()
    results_test['y_pred'] = y_test_pred * 1e-9
    results_test['y_true'] = y_test.reset_index(drop=True) * 1e-9
    results_test['block_number'] = test_data['block_number'].reset_index(
        drop=True)
    if use_topo:
        results_test['tx_topology'] = test_data['tx_topology'].reset_index(
            drop=True)

    return results_test, results_train, model


def build_lag_features(target, lags):
    return [f'{target}_lag_{lag}' for lag in lags]


def create_rolling_features(data, rolling_windows, features):
    df = data.set_index('block_number')[features].copy()
    output = [data.set_index('block_number')]

    for rolling_window in rolling_windows:
        mean_df = df.shift(1).rolling(rolling_window).mean().rename(
            columns={f: f + f'_mean_{rolling_window}' for f in features})
        std_df = df.shift(1).rolling(rolling_window).std().rename(
            columns={f: f + f'_std_{rolling_window}' for f in features})
        output.append(mean_df)
        output.append(std_df)

    output = pd.concat(output, axis=1)
    return output.dropna().reset_index()


