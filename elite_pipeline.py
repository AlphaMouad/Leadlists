import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, auc
from tensorflow.keras import layers, models, Input, callbacks, optimizers, regularizers
from sklearn.utils import class_weight
from numpy.lib.stride_tricks import sliding_window_view
import warnings
import io
import requests

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. ELITE DATA ENGINEERING MODULES
# ------------------------------------------------------------------------------

def elite_multiframe_merge(hf_data: pd.DataFrame, lf_data: pd.DataFrame,
                          tolerance: pd.Timedelta = None) -> pd.DataFrame:
    hf_data = hf_data.sort_index()
    lf_data = lf_data.sort_index()
    hf_temp = hf_data.reset_index()
    lf_temp = lf_data.reset_index()
    time_key = hf_temp.columns[0]
    lf_temp = lf_temp.add_prefix('LF_')
    lf_col_map = {f'LF_{time_key}': time_key}
    lf_temp = lf_temp.rename(columns=lf_col_map)
    merged = pd.merge_asof(
        hf_temp,
        lf_temp,
        on=time_key,
        direction='backward',
        tolerance=tolerance,
        allow_exact_matches=True
    )
    merged = merged.set_index(time_key)
    return merged

def get_weights_ffd(d: float, thres: float) -> np.ndarray:
    w, k = [1.], 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series: pd.DataFrame, d: float, thres=1e-5) -> pd.DataFrame:
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    df_out = {}
    for name in series.columns:
        series_f = series[[name]].ffill().dropna()
        temp_df = pd.Series(index=series_f.index, dtype=float)
        for i in range(width, len(series_f)):
            window0 = series_f.iloc[i-width:i+1].values
            temp_df.iloc[i] = np.dot(w.T, window0).item()
        df_out[name] = temp_df
    return pd.DataFrame(df_out)

def vectorized_rolling_zscore(data: np.ndarray, window: int) -> np.ndarray:
    shape_view = sliding_window_view(data, window_shape=window)
    rolling_mean = np.mean(shape_view, axis=1)
    rolling_std = np.std(shape_view, axis=1)
    valid_data_slice = data[window-1:]
    epsilon = 1e-8
    valid_zscores = (valid_data_slice - rolling_mean) / (rolling_std + epsilon)
    pad = np.full(window - 1, np.nan)
    result = np.concatenate((pad, valid_zscores))
    return result

# ------------------------------------------------------------------------------
# 2. TECHNICAL INDICATORS
# ------------------------------------------------------------------------------

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return atr

def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0):
    high = df['high']
    low = df['low']
    close = df['close']
    atr = calculate_atr(high, low, close, period=atr_period)
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    close_arr = close.values
    bu_arr = basic_upper.values
    bl_arr = basic_lower.values

    fu = np.zeros(len(df))
    fl = np.zeros(len(df))
    st = np.zeros(len(df))
    td = np.zeros(len(df))

    fu[:] = np.nan
    fl[:] = np.nan
    st[:] = np.nan

    valid_start = atr.first_valid_index()
    if valid_start is None:
        return pd.Series(st, index=df.index), pd.Series(td, index=df.index)

    start_idx = df.index.get_loc(valid_start)

    for i in range(start_idx, len(df)):
        if i == start_idx:
            fu[i] = bu_arr[i]
            fl[i] = bl_arr[i]
            st[i] = fu[i]
            td[i] = -1
        else:
            if (bu_arr[i] < fu[i-1]) or (close_arr[i-1] > fu[i-1]):
                fu[i] = bu_arr[i]
            else:
                fu[i] = fu[i-1]
            if (bl_arr[i] > fl[i-1]) or (close_arr[i-1] < fl[i-1]):
                fl[i] = bl_arr[i]
            else:
                fl[i] = fl[i-1]
            if st[i-1] == fu[i-1]:
                if close_arr[i] > fu[i]:
                    td[i] = 1
                else:
                    td[i] = -1
            else:
                if close_arr[i] < fl[i]:
                    td[i] = -1
                else:
                    td[i] = 1
            if td[i] == 1:
                st[i] = fl[i]
            else:
                st[i] = fu[i]

    return pd.Series(st, index=df.index), pd.Series(td, index=df.index)

def calculate_trend_state(df, up_col, down_col):
    ts = []
    up_vals = df[up_col].values
    down_vals = df[down_col].values
    for i in range(len(df)):
        if i == 0:
            ts.append(0.5)
            continue
        u = up_vals[i]
        d = down_vals[i]
        u_prev = up_vals[i-1]
        d_prev = down_vals[i-1]

        is_up_valid = not np.isnan(u) and u != 0
        is_down_valid = not np.isnan(d) and d != 0
        is_up_prev_valid = not np.isnan(u_prev) and u_prev != 0
        is_down_prev_valid = not np.isnan(d_prev) and d_prev != 0

        if is_up_valid and not is_down_valid:
            if i > 0 and is_up_prev_valid and u > u_prev:
                ts.append(1.0)
            else:
                ts.append(0.5)
        elif not is_up_valid and is_down_valid:
            if i > 0 and is_down_prev_valid and d < d_prev:
                ts.append(0.0)
            else:
                ts.append(0.5)
        else:
            ts.append(0.5)
    return pd.Series(ts, index=df.index)

def add_trend_features(df, prefix=""):
    st, direction = calculate_supertrend(df)
    up = np.where(direction == 1, st, np.nan)
    down = np.where(direction == -1, st, np.nan)
    df[f"{prefix}Up Trend"] = pd.Series(up, index=df.index)
    df[f"{prefix}Down Trend"] = pd.Series(down, index=df.index)
    df[f"{prefix}Trend State"] = calculate_trend_state(df, f"{prefix}Up Trend", f"{prefix}Down Trend")
    df[f"{prefix}Up Trend"] = df[f"{prefix}Up Trend"].fillna(0)
    df[f"{prefix}Down Trend"] = df[f"{prefix}Down Trend"].fillna(0)
    return df

# ------------------------------------------------------------------------------
# 3. DATA LOADING AND PREPROCESSING
# ------------------------------------------------------------------------------

def load_data(url_m5: str, url_h4: str) -> pd.DataFrame:
    print(f"Loading data from {url_m5} and {url_h4}...")
    try:
        data_h4 = pd.read_csv(url_h4)
        data_m5 = pd.read_csv(url_m5)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

    if 'Date' in data_h4.columns:
        data_h4['Date'] = pd.to_datetime(data_h4['Date'], utc=True)
        data_h4 = data_h4.set_index('Date')

    if 'Date' in data_m5.columns:
        data_m5['Date'] = pd.to_datetime(data_m5['Date'], utc=True)
        data_m5 = data_m5.set_index('Date')

    data_h4 = add_trend_features(data_h4, prefix="H4_")
    macd_h4, sig_h4, _ = calculate_macd(data_h4['close'])
    data_h4['H4_MACD'] = macd_h4
    data_h4['H4_MACD_signal'] = sig_h4

    data_h4.index = data_h4.index + pd.Timedelta(hours=4)

    data_m5 = add_trend_features(data_m5)
    macd_m5, sig_m5, _ = calculate_macd(data_m5['close'])
    data_m5['MACD'] = macd_m5
    data_m5['MACD_signal'] = sig_m5

    h4_cols = ['H4_Up Trend', 'H4_Down Trend', 'H4_Trend State', 'H4_MACD', 'H4_MACD_signal']
    merged_data = elite_multiframe_merge(
        data_m5,
        data_h4[h4_cols],
        tolerance=pd.Timedelta(hours=4)
    )
    # Rename LF_ prefixed columns back to original H4 names
    rename_map = {f"LF_{c}": c for c in h4_cols}
    merged_data = merged_data.rename(columns=rename_map)

    merged_data = merged_data.ffill().bfill()
    return merged_data

def prepare_features(data: pd.DataFrame, norm_window: int = 250):
    print("Feature Engineering...")
    data['Close_FFD'] = frac_diff_ffd(data[['close']], d=0.4)['close']
    data['RSI_14'] = calculate_rsi(data['close'])
    data['Vol_20'] = data['close'].rolling(20).std()

    feature_cols = ['open', 'high', 'low', 'close', 'Volume', 'RSI_14', 'Vol_20', 'Close_FFD']
    data = data.dropna()
    feature_values = data[feature_cols].values

    normalized_features = []
    for i, col in enumerate(feature_cols):
        col_values = feature_values[:, i]
        norm_col = vectorized_rolling_zscore(col_values, norm_window)
        normalized_features.append(norm_col)

    normalized_features = np.column_stack(normalized_features)
    norm_df = pd.DataFrame(normalized_features, columns=[f"{c}_norm" for c in feature_cols], index=data.index)
    final_df = pd.concat([data, norm_df], axis=1)
    final_df = final_df.dropna()
    return final_df

# ------------------------------------------------------------------------------
# 4. LABELING AND MODELING
# ------------------------------------------------------------------------------

def apply_triple_barrier_final(close: pd.Series, t_events: pd.DatetimeIndex,
                        pt_sl: list, target_vol: pd.Series,
                        vertical_barrier_bars: int, side: int = 1) -> pd.DataFrame:
    events = pd.DataFrame(index=t_events)
    events['t1'] = pd.NaT
    events['trgt'] = target_vol.loc[events.index]
    out = pd.DataFrame(index=events.index)
    # Initialize with the same datetime dtype as the input close series to avoid TZ mismatch
    out['t1'] = pd.Series(pd.NaT, index=events.index, dtype=close.index.dtype)
    out['label'] = 0

    for loc, row in events.iterrows():
        try:
            start_idx = close.index.get_loc(loc)
        except KeyError:
            continue
        end_idx = min(start_idx + vertical_barrier_bars, len(close) - 1)
        t1 = close.index[end_idx]
        trgt = row['trgt']
        if pd.isna(trgt) or trgt == 0:
            continue
        df0 = close.iloc[start_idx : end_idx+1]
        entry_price = df0.iloc[0]
        returns = (df0 / entry_price) - 1
        returns = returns * side
        upper = trgt * pt_sl[0]
        lower = -trgt * pt_sl[1]
        idx_upper = returns[returns > upper].index.min()
        idx_lower = returns[returns < lower].index.min()
        if pd.isna(idx_upper) and pd.isna(idx_lower):
            out.loc[loc, 't1'] = t1
            out.loc[loc, 'label'] = 0
        elif pd.isna(idx_lower):
            out.loc[loc, 't1'] = idx_upper
            out.loc[loc, 'label'] = 1
        elif pd.isna(idx_upper):
            out.loc[loc, 't1'] = idx_lower
            out.loc[loc, 'label'] = 0
        else:
            if idx_upper < idx_lower:
                out.loc[loc, 't1'] = idx_upper
                out.loc[loc, 'label'] = 1
            else:
                out.loc[loc, 't1'] = idx_lower
                out.loc[loc, 'label'] = 0
    return out

def get_meta_labels_final(data: pd.DataFrame,
                    daily_vol_window: int = 50,
                    pt_sl: list = [1, 1],
                    min_ret: float = 0.001,
                    vertical_barrier_bars: int = 24) -> pd.DataFrame:
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=daily_vol_window).std()
    cond_h4_trend = (data['H4_Trend State'] == 0)
    cond_m5_trend = (data['Up Trend'] == 0)
    cond_macd = (data['MACD'] < data['MACD_signal'])
    cond_h4_macd = (data['H4_MACD'] < data['H4_MACD_signal'])
    entry_mask = cond_h4_trend & cond_m5_trend & cond_macd & cond_h4_macd
    t_events = data.index[entry_mask]
    t_events = t_events[t_events.isin(volatility.index)]
    if len(t_events) == 0:
        return pd.DataFrame()
    labels = apply_triple_barrier_final(
        data['close'], t_events, pt_sl, volatility, vertical_barrier_bars, side=-1
    )
    return labels.dropna()

def create_sequences(features_df, t_events, seq_length=60):
    X = []
    valid_events = []
    for timestamp in t_events:
        try:
            idx = features_df.index.get_loc(timestamp)
        except KeyError:
            continue
        if idx < seq_length:
            continue
        seq = features_df.iloc[idx - seq_length + 1 : idx + 1].values
        X.append(seq)
        valid_events.append(timestamp)
    return np.array(X), valid_events

def build_attention_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def train_model(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(weights))
    print(f"Class Weights: {class_weights}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nTraining Fold {fold}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = build_attention_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        y_pred_prob = model.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
        print(classification_report(y_val, y_pred))
        try:
            print(f"AUC: {roc_auc_score(y_val, y_pred_prob)}")
        except:
            print("AUC calculation failed (possibly only one class in validation set).")

def main():
    print("Starting Elite ML Pipeline...")
    url_m5 = "https://docs.google.com/spreadsheets/d/13BV01jpvVQ5XaN1XTAS_Rchf8GfbuUy-YFSzjNX903w/export?format=csv"
    url_h4 = "https://docs.google.com/spreadsheets/d/1kUa2NSzEz25kFI7t4UvTIAGcwCclnAa14AamVCrHB68/export?format=csv"
    data = load_data(url_m5, url_h4)
    if data.empty:
        print("Failed to load data.")
        return
    data_processed = prepare_features(data, norm_window=250)
    labels = get_meta_labels_final(data_processed)
    if labels.empty:
        print("No events found for labeling.")
        return
    print(f"Generated {len(labels)} labels.")
    print(labels['label'].value_counts())
    norm_cols = [c for c in data_processed.columns if c.endswith('_norm')]
    X, valid_timestamps = create_sequences(data_processed[norm_cols], labels.index, seq_length=60)
    y = labels.loc[valid_timestamps, 'label'].values
    if len(X) == 0:
        print("No valid sequences created.")
        return
    print(f"Dataset Shape: X={X.shape}, y={y.shape}")
    train_model(X, y, n_splits=5)
    print("Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
