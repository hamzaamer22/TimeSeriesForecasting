import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from statsmodels.tsa.seasonal import seasonal_decompose


# Placeholder function for generating the forecast
def generate_forecast(data, column_name):
    st.write(f"Forecasting for column: {column_name}")

    # Function to identify date columns and rename them to 'Period'
    def identify_date_column(df):
        date_column = None
        # Iterate through columns to find date-like column
        for col in df.columns:
            if pd.to_datetime(df[col], errors='coerce').notnull().all():
                date_column = col
                break
        return date_column

    # Function to detect date format
    def detect_date_format(date_str):
        import re
        # Define regular expressions for different date formats
        formats = {
            '%d.%m.%Y': re.compile(r'\d{2}\.\d{2}\.\d{4}'),
            '%d-%m-%Y': re.compile(r'\d{2}-\d{2}-\d{4}'),
            '%Y-%m-%d': re.compile(r'\d{4}-\d{2}-\d{2}'),
            '%Y-%m': re.compile(r'\d{4}-\d{2}'),
            '%m-%Y': re.compile(r'\d{2}-\d{4}')
        }
        # Check each format
        for fmt, regex in formats.items():
            if regex.fullmatch(date_str):
                return fmt
        # If no match, return None
        return None

    def adfuller_test(data, column_name):
        counter = 0
        # Perform ADF test
        result = adfuller(data[column_name], autolag='AIC')
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])

        data[f'{column_name}_diff'] = data[column_name]

        while result[1] > 0.05:
            # Log transform the 'Sales_quantity' column
            data[f'{column_name}_diff'] = data[f'{column_name}_diff'].diff().dropna()
            result = adfuller(data[f'{column_name}_diff'].dropna(), autolag='AIC')
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            counter += 1

        print(f"\nOrder of differencing: {counter}")
        return counter



    def get_arima_parameters(data, target_variable, counter):
        st.write("Getting best parameter orders")

        # Calculate the indices for the middle 50% portion
        n = len(data)
        start_idx = int(n * 0.25)
        end_idx = int(n * 0.75)

        # Extract the middle 50% portion of the data
        middle_50_data = data.iloc[start_idx:end_idx]

        # Define parameter grid for p, d, q
        p = range(0, 15)  # Example range for p
        d = counter
        q = range(0, 15)  # Example range for q

        # Total number of combinations
        total_combinations = len(p) * len(q)

        # Initialize progress bar
        progress_bar = st.progress(0)
        current_progress = 0

        # Grid search for ARIMA hyperparameters
        best_aic = np.inf
        best_order = None
        best_model = None

        for i, param in enumerate(product(p, q)):
            try:
                model = ARIMA(middle_50_data[target_variable], order=(param[0], d, param[1]))
                results = model.fit()
                aic = results.aic
                # st.write(f"Testing ARIMA({param[0]}, {d}, {param[1]}): AIC = {aic}")

                if aic < best_aic:
                    best_aic = aic
                    best_order = param
                    best_model = results

            except Exception as e:
                print(f"Skipping ARIMA({param[0]}, {d}, {param[1]}) due to error: {e}")
                # st.write(f"Skipping ARIMA({param[0]}, {d}, {param[1]}) due to error: {e}")

            # Update progress bar
            current_progress += 1
            progress_percentage = current_progress / total_combinations
            progress_bar.progress(progress_percentage)

        # st.write(f"Best ARIMA({best_order}) with AIC = {best_aic}")
        return best_order

    def calculate_seasonal_strength(data, target_variable):
        # Decompose the time series
        decomposition = seasonal_decompose(data[target_variable], model='additive', period=12)

        # Extract the components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Calculate seasonal strength
        sst = np.nanvar(seasonal)
        sst_plus_sert = np.nanvar(seasonal + residual)
        seasonal_strength = max(0, 1 - (sst / sst_plus_sert))

        # Plot the decomposition
        decomposition.plot()

        print(f"Seasonal Strength: {seasonal_strength}")
        return seasonal_strength

    def select_model(seasonal_strength):
        # Define the threshold for model selection
        threshold = 0.5
        # Model selection based on seasonal strength
        if seasonal_strength > threshold:
            print(f"Seasonal Strength: {seasonal_strength} (above {threshold}), using SARIMAX model.")
            model_type = "SARIMAX"
            return model_type
        else:
            print(f"Seasonal Strength: {seasonal_strength} (below {threshold}), using ARIMA model.")
            model_type = "ARIMA"
            return model_type

    def get_sarimax_parameters(data, target_variable, new_p, new_d, new_q):
        # Define parameter grid for P, D, Q, s
        P = range(1, 3)  # Adjust range as needed
        D = range(1, 3)  # Typical values for D (0 or 1)
        Q = range(1, 3)  # Adjust range as needed
        s = 12  # Yearly seasonality for daily data

        # Grid search for seasonal hyperparameters
        best_aic = np.inf  # AIC should be minimized, so start with infinity
        best_seasonal_order = None

        # Calculate total combinations for progress bar
        total_combinations = len(P) * len(D) * len(Q)

        # Initialize custom progress bar using HTML
        progress_html = st.empty()
        current_progress = 0

        for param_seasonal in product(P, D, Q):
            try:
                model = SARIMAX(data[target_variable],
                                order=(new_p, new_d, new_q),
                                seasonal_order=(param_seasonal[0], param_seasonal[1], param_seasonal[2], s),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                aic = results.aic
                # st.write(f"Testing SARIMAX({new_p}, {new_d}, {new_q})x{param_seasonal}: AIC = {aic}")

                if aic < best_aic:  # Update to find the minimum AIC
                    best_aic = aic
                    best_seasonal_order = param_seasonal
                    best_model = results

            except Exception as e:
                print(f"Skipping SARIMAX({new_p}, {new_d}, {new_q})x{param_seasonal} due to error: {e}")
                # st.write(f"Skipping SARIMAX({new_p}, {new_d}, {new_q})x{param_seasonal} due to error: {e}")

            # Update progress bar
            current_progress += 1
            progress_percentage = (current_progress / total_combinations) * 100

            # Custom HTML for red progress bar
            progress_html.markdown(f"""
                <div style="width: 100%; background-color: #ddd;">
                    <div style="width: {progress_percentage}%; background-color: red; height: 24px;"></div>
                </div>
            """, unsafe_allow_html=True)

        # st.write(f"Best SARIMAX({new_p}, {new_d}, {new_q})x{best_seasonal_order} with AIC = {best_aic}")
        return best_seasonal_order

    # Placeholder function for generating the forecast
    def generate_forecast_plot(data, column_name):

        d = adfuller_test(data, column_name)

        best_order = get_arima_parameters(data, column_name, d)
        new_p = best_order[0]
        new_q = best_order[1]
        new_d = d

        seasonal_strength_index = calculate_seasonal_strength(data, column_name)

        model_type = select_model(seasonal_strength_index)

        st.write(f"Applying model: {model_type}")

        # Log transform the target variable
        data[column_name] = np.log(data[column_name])

        # Split the data into training and test sets
        st.write("Training Model")
        train_size = int(len(data) * 0.8)  # 80% training, 20% test
        train, test = data[0:train_size], data[train_size:]


        print(f"New p: {new_p}, New q: {new_q}, New d: {new_d}")

        if model_type == "ARIMA":
            # Train ARIMA model
            model = ARIMA(train[column_name], order=(new_p, new_d, new_q))  # Replace with the appropriate order
            model_fit = model.fit()

        else:
            best_seasonal_order = get_sarimax_parameters(data, column_name, new_p, new_d, new_q)
            new_P = best_seasonal_order[0]
            new_D = best_seasonal_order[1]
            new_Q = best_seasonal_order[2]
            print(f"New P: {new_P}, New Q: {new_Q}, New D: {new_D}")
            model = SARIMAX(train[column_name], order=(new_p, new_d, new_q),
                            seasonal_order=(new_P, new_D, new_Q, 12))
            model_fit = model.fit(disp=False)





        # Forecast the test data
        forecast = model_fit.forecast(steps=len(test))

        # Plot the forecasted values vs actual values
        plt.figure(figsize=(12, 6))
        plt.plot(train[column_name], label='Training Data')
        plt.plot(test[column_name], label='Test Data')
        plt.plot(forecast, label='Forecasted Data', color='red')
        plt.legend()
        plt.title(f'{column_name} Forecast vs Actual')
        plt.xlabel('Time')
        plt.ylabel(column_name)
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(plt)
######
    date_column = identify_date_column(data)

    # Rename the date column to 'period' if found
    if date_column:
        data.rename(columns={date_column: 'Period'}, inplace=True)
        print(f"Column '{date_column}' has been renamed to 'Period'.")
    else:
        print("No date column found.")

#########
    date_str = data.at[0, 'Period']
    date_format = detect_date_format(date_str)

    if date_format:
        print(f"Date: {date_str} | Detected format: {date_format}")
    else:
        print(f"Date: {date_str} | Detected format: Not detected")

##########
    data['Period'] = pd.to_datetime(data['Period'], format=date_format)
##########
    data.set_index('Period', inplace=True)
    data.head(10)

    generate_forecast_plot(data, column_name)
    # Streamlit UI already defined to handle file upload and column selection
    # Functionality will be triggered by the "Generate Forecast" button


# Set page configuration
st.set_page_config(page_title="Time Series Forecasting", layout="centered")

# Custom CSS to adjust the style
st.markdown("""
    <style>
        .main {
            background-color: #e5e5e5;
            color: #000;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-top: 50px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            margin-top: -20px;
            margin-bottom: 30px;
        }
        .file-upload-container, .column-selection-container, .button-container {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .button-container button {
            background-color: #007BFF; /* Button color */
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .button-container button:hover {
            background-color: #0056b3; /* Darker shade on hover */
        }
        .footer {
            font-size: 14px;
            text-align: center;
            color: #888888;
            margin-top: 50px;
        }
        .plot-container {
            display: flex;
            justify-content: center;
            margin-bottom: 50px;
        }
        .plot-container img {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Time Series Forecasting</div>', unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="subtitle">Predict future values from historical data</div>', unsafe_allow_html=True)

# File Upload
st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    st.write("## Uploaded Data")
    st.dataframe(data)

    # Column Selection
    st.markdown('<div class="column-selection-container">', unsafe_allow_html=True)

    column_to_forecast = st.selectbox(
        "Select the column you want to forecast:",
        options=data.columns,
        help="Choose the column that contains the time series data for forecasting."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Generate Forecast Button
    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    if st.button("Generate Forecast"):
        generate_forecast(data, column_to_forecast)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Sample Time Series Plot
st.markdown('<div class="plot-container">', unsafe_allow_html=True)

# Generating sample data if no file is uploaded
if uploaded_file is None:
    np.random.seed(0)
    dates = pd.date_range(start='2023-01-01', periods=12, freq='ME')
    data = np.random.randn(12).cumsum()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dates, data, marker='o', color='#0072B2')
    ax.set_title('Sample Time Series', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.grid(True)

    # Display the plot
    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# Footer note
st.markdown('<div class="footer">Time Series Forecasting &copy; 2024</div>', unsafe_allow_html=True)



