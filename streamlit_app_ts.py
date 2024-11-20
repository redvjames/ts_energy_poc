__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
from buildings_bench.transforms import TimestampTransform, BoxCoxTransform as transform
from buildings_bench import load_torch_dataset, load_pandas_dataset, benchmark_registry, utils, BuildingTypes
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.data.datasets import PandasTransformerDataset, keep_buildings
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench.evaluation import aggregate, scoring_rule_factory
from buildings_bench.models import model_factory
from torch.utils.data import ConcatDataset, Subset
from pathlib import Path
import os

try:
    df_input = pd.read_csv('./data/data.csv', index_col=0)
except FileNotFoundError:
    df_input = pd.DataFrame(columns=['datetime', 'energy'])


# Create columns for the title and logo
col1, col2 = st.columns([3.5, 1])  # Adjust the ratio as needed

# Title in the first column
with col1:
    st.title("⚡ Energy Consumption Forecasting POC")
    st.write(
        "This app forecasts hourly energy consumption of buildings"
        " with an input of at least 7 days of hourly data."
    )
# Logo and "Developed by CAIR" text in the second column
with col2:
    st.image("images/CAIR_cropped.png", use_column_width=True)
    st.markdown(
        """
        <div style="text-align: center; margin-top: -10px;">
            Developed by CAIR
        </div>
        """, 
        unsafe_allow_html=True)

# Adding the sidebar for selecting the repo_id
st.sidebar.title("Input Data")

uploaded_file = st.sidebar.file_uploader("Choose a file")

edited_df = st.sidebar.data_editor(df_input)

horizon = st.radio(
    "Forecast Length",
    ["1 Day", "1 Week"],
    captions=[
        "24 Hours",
        "168 Hours"
    ], horizontal=True
)

if horizon == "1 Day":
    st.write("The model will predict 1 Day Ahead.")
else:
    st.write("The model will predict 1 Week Ahead.")


# Model Setup

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Instantiate with dictionary values
args = Args(
    # repo_path=r'/mount/src/ts_energy_poc/',
    model='TransformerWithGaussian-M',
    checkpoint=r'/mount/src/ts_energy_poc/model/TransformerWithGaussian-M_Thai.pt',
    device=None,
    # benchmark=[data_source],
    ignore_scoring_rules=False,
    dont_subsample_buildings=True, 
    apply_scaler_transform='boxcox',
    include_outliers=False,
    num_training_days = 60,
    variant_name = 'ph_data',
    eval_zero_shot = False,
    lr = 1e-3,
    batch_size=360,
    num_workers = 4,
    max_epochs=25,
    patience=10
    # results_path = r'/mount/src/ts_energy_poc/'
)

model_args = {'context_len': 168,
 'pred_len': 24,
 'num_encoder_layers': 3,
 'num_decoder_layers': 3,
 'd_model': 512,
 'nhead': 8,
 'dim_feedforward': 1024,
 'dropout': 0.0,
 'activation': 'gelu',
 'continuous_loads': True,
 'ignore_spatial': False,
 'continuous_head': 'gaussian_nll'}

# load and configure the model for transfer learning
model, loss, _ = model_factory(args.model, model_args)
model = model.to(args.device)
# transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) / 'metadata' / 'transforms'

checkpoint_path = args.checkpoint
st.write(checkpoint_path)
st.write(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")

# if args.checkpoint != '':
#     # By default, fine tune all layers
#     model.load_from_checkpoint(args.checkpoint)
# model.train()

# st.write(os.path.dirname(np.__file__))
# st.write(os.getcwd())
# st.write(os.listdir('/mount/src/ts_energy_poc'))
# st.write(os.listdir('/home/adminuser/venv'))
# st.write(os.listdir('/home/adminuser/venv/lib/python3.11/site-packages'))
# st.write(os.listdir('/home/adminuser/venv/bin'))

# if st.button('Predict Energy Consumption'):
#     if uploaded_file is not None:
#         # Can be used wherever a "file-like" object is accepted:
#         df_input = pd.read_csv(uploaded_file, index_col=0)
#         # st.write(df_input)
#     else:
#         df_input = edited_df.copy()

#     from model_load import load_model
#     model = load_model()
    
#     if horizon == "1 Day":
#         extended_index = pd.date_range(start=df_input.index[-1], periods=25, freq='H')[1:]
#         df_input = pd.concat([df_input, pd.DataFrame(index=extended_index)])

#         df_ts = pd.DataFrame()
#         df_ts['y'] = df_input[['energy']]

#         ## Plotting ##

#         # Lookback window size 
#         window_size = 24
#         h = 24

#         for w in range(window_size):
#             df_ts['y-' + str(w + 1)] = df_input[['energy']].shift(w+1)

#         df_ts = df_ts[window_size:]
#         # display(df_ts)

#         X = df_ts[df_ts.columns[1:]]
#         X = df_ts[df_ts.columns[1:]]  # Feature
#         Y = df_ts['y']  # Target

#         X_train = X[:-h]
#         X_test = X[-h:]
#         Y_train = Y[:-h]
#         Y_test = Y[-h:]

#         scaler_x = StandardScaler().fit(X_test)
#         scaler_y = StandardScaler().fit(df_input.values.reshape(-1, 1))
#         x_train = scaler_x.transform(X_test)


#         y_pred = model.predict(x_train)
#         y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        
#         ## Plotting ##
        
#         df_plot = pd.DataFrame(index=df_input.index[-h:])
#         df_plot['Baseline'] = X_test['y-24'].values
#         df_plot['Prediction'] = y_pred
        
#         fig, ax = plt.subplots(figsize=(10, 6))
#         df_plot['Prediction'].plot(ax=ax, label='Actual')
#         df_plot['Baseline'].plot(ax=ax, linestyle='--')
#         plt.autoscale()
#         plt.legend(['Prediction', 'Baseline'])
#         plt.xlabel('Datetime')
#         plt.ylabel('Energy')
#         plt.title('Prediction to Baseline Comparison')
#         st.pyplot(fig)

#         df_test = pd.concat([df_input, df_plot]).dropna(how='all')
#         fig, ax = plt.subplots(figsize=(10, 6))
#         df_test[['energy', 'Prediction']].plot(ax=ax, label='energy')
#         plt.axvline(x=len(df_input)-25, color='red', linestyle='--', linewidth=1)
#         plt.autoscale()
#         plt.legend(['Historical', 'Prediction'])
#         plt.xlabel('Datetime')
#         plt.ylabel('Energy')
#         plt.title('Historical with Forecast Plot')
#         ax.tick_params(axis='x', labelrotation=30)
#         plt.show()
#         st.pyplot(fig)

#         st.dataframe(pd.DataFrame(df_plot['Prediction']), height=300, width=400)
    
#     else:
        
#         extended_index = pd.date_range(start=df_input.index[-1], periods=169, freq='H')[1:]
#         df_input = pd.concat([df_input, pd.DataFrame(index=extended_index)])
        
#         df_ts = pd.DataFrame()
#         df_ts['y'] = df_input[['energy']]

#         ## Modelling ##
           
#         # Lookback window size 
#         window_size = 24
#         h =168

#         for w in range(window_size):
#             df_ts['y-' + str(w + 1)] = df_input[['energy']].shift(w+1)

#         df_ts = df_ts[window_size:]
#         # display(df_ts)

#         X = df_ts[df_ts.columns[1:]]
#         X = df_ts[df_ts.columns[1:]]  # Feature
#         Y = df_ts['y']  # Target

#         X_train = X[:-h]
#         X_test = X[-h:]
#         Y_train = Y[:-h]
#         Y_test = Y[-h:]

#         scaler_x = StandardScaler().fit(X_test)
#         scaler_y = StandardScaler().fit(df_input.values.reshape(-1, 1))
#         x_train = scaler_x.transform(X_test)
        
#         y_pred = model.predict(x_train)
#         y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        
#         ## Plotting ##
        
#         df_plot = pd.DataFrame(index=df_input.index[-h:])
#         df_plot['Baseline'] = X_test['y-24'].values
#         df_plot['Prediction'] = y_pred

#         fig, ax = plt.subplots(figsize=(10, 6))
#         df_plot['Prediction'].plot(ax=ax, label='Actual')
#         df_plot['Baseline'].plot(ax=ax, linestyle='--')
#         plt.autoscale()
#         plt.legend(['Prediction', 'Baseline'])
#         plt.xlabel('Datetime')
#         plt.ylabel('Energy')
#         plt.title('Prediction to Baseline Comparison')
#         st.pyplot(fig)
        
#         df_test = pd.concat([df_input, df_plot[['Prediction']]]).dropna(how='all')
#         fig, ax = plt.subplots(figsize=(10, 6))
#         df_test[['energy', 'Prediction']].plot(ax=ax, label='energy')
#         plt.axvline(x=len(df_input)-169, color='red', linestyle='--', linewidth=1)
#         plt.autoscale()
#         plt.legend(['Historical', 'Prediction'])
#         plt.xlabel('Datetime')
#         plt.ylabel('Energy')
#         plt.title('Historical with Forecast Plot')
#         ax.tick_params(axis='x', labelrotation=30)
#         st.pyplot(fig)
        
#         # df_day = df_input.reset_index().dropna()
#         # df_day['index'] = pd.to_datetime(df_day['index'])
#         # df_input_day = df_day.groupby(df_day['index'].dt.floor('D')).sum()
#         # df_day2 = df_test['Prediction'].reset_index().dropna()
#         # df_day2['index'] = pd.to_datetime(df_day2['index'])
#         # df_pred_day = df_day2.groupby(df_day2['index'].dt.floor('D')).sum()

#         # fig, ax = plt.subplots(figsize=(10, 6))
#         # df_input_day.plot(ax=ax, label='energy')
#         # df_pred_day.plot(ax=ax, label='historical')
#         # plt.autoscale()
#         # plt.legend(['Historical', 'Prediction'])
#         # plt.xlabel('Datetime')
#         # plt.ylabel('Energy')
#         # plt.title('Model Comparison')
#         # st.pyplot(fig)
        
#         st.dataframe(pd.DataFrame(df_plot['Prediction']), height=300, width=400)
