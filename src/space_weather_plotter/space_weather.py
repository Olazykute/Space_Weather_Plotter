"""
This module fetches and visualizes space weather data from the NASA API.

It includes functions to fetch data from the NASA API, prepare the data into Polars DataFrames,
and plot the data using Matplotlib.
"""
import requests #type: ignore
import matplotlib.pyplot as plt
import polars as pl

# Define constants
BASE_URL = "https://api.nasa.gov/DONKI/"
API_KEY = "DEMO_KEY"  # Replace with your NASA API key

# Function to fetch data from NASA API
def fetch_data(endpoint, params):
    """
    Fetches data from the NASA API.

    Args:
        endpoint (str): The API endpoint to fetch data from.
        params (dict): The parameters to include in the API request.

    Returns:
        dict or None: The JSON response from the API if the request is successful, otherwise None.
    """
    params['api_key'] = API_KEY
    response = requests.get(BASE_URL + endpoint, params = params, timeout = 10000)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to prepare a Polars DataFrame
def prepare_dataframe(data, columns_mapping, nested_keys=None, timestamp_col=None):
    """
    Prepares a Polars DataFrame from the input JSON data.

    Args:
        data (list): The raw JSON data (list of dictionaries).
        columns_mapping (dict): A dictionary mapping original JSON keys to desired DataFrame column names.
        nested_keys (list, optional): A list of keys to extract from nested JSON objects. Defaults to None.
        timestamp_col (str, optional): The column name to parse as a timestamp. Defaults to None.

    Returns:
        pl.DataFrame: The prepared Polars DataFrame.
    """
    if not data:
        return pl.DataFrame()

    def extract_nested(entry, nested_keys):
        if not isinstance(nested_keys, dict):
            return entry
        for nested_key, extraction in nested_keys.items():
            nested_values = entry.get(nested_key, [])
            if isinstance(nested_values, list):
                for nested_entry in nested_values:
                    if isinstance(nested_entry, dict):
                        for sub_key, sub_col in extraction.items():
                            entry[sub_col] = nested_entry.get(sub_key)
        return entry

    processed_data = [
        extract_nested({columns_mapping.get(k, k): v for k, v in entry.items()}, nested_keys)
        for entry in data if isinstance(entry, dict)
    ]

    # Create Polars DataFrame
    df = pl.DataFrame(processed_data)

    # Convert timestamp column to datetime
    if timestamp_col in df.columns:
        df = df.with_columns(pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%MZ"))

    return df


# Function to plot data using Polars
def plot_data(df, x_col, y_col, title, x_label, y_label, kind="line", color="blue"):
    """
    Plots data using Matplotlib.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the data to plot.
        x_col (str): The column name to use for the x-axis.
        y_col (str): The column name to use for the y-axis.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        kind (str, optional): The type of plot ('line' or 'bar'). Defaults to "line".
        color (str, optional): The color of the plot. Defaults to "blue".

    Returns:
        None
    """
    if df.is_empty():
        print(f"No data available for {title}.")
        return

    # Plot using Matplotlib
    plt.figure(figsize=(10, 5))
    if kind == "line":
        plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', color=color)
    elif kind == "bar":
        plt.bar(df[x_col], df[y_col], color=color)


    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

# Function to fetch and plot data based on user selection
def main():
    """
    Main function to fetch and plot space weather data.

    This function fetches current space weather data, including solar flares and geomagnetic storms,
    prepares the data into Polars DataFrames, and plots the data using Matplotlib.
    
    Args:
        None

    Returns:
        None
    """
    print("Welcome to the Space Weather Observer")
    print("Fetching current space weather data please wait a few seconds...")

    # Fetch and prepare solar flare data
    flare_params = {"startDate": "2024-01-01", "endDate": "2024-11-23"}
    solar_flares = fetch_data("FLR", flare_params)
    flare_columns = {"beginTime": "time", "classType": "intensity"}
    solar_flares_df = prepare_dataframe(solar_flares, flare_columns, "time")

    # Fetch and prepare geomagnetic storm data
    storm_params = {"startDate": "2024-01-01", "endDate": "2024-11-23"}
    geomagnetic_storms = fetch_data("GST", storm_params)
    storm_columns = {"startTime": "time"}
    storm_nested_col = {"allKpIndex": {"observedTime": "time", "kpIndex": "kp_index"}}
    geomagnetic_storms_df = prepare_dataframe(geomagnetic_storms, storm_columns, storm_nested_col, "time")

    # Fetch and prepare solar wind data
    wind_params = {"startDate": "2024-11-22", "endDate": "2024-11-23"}
    solar_wind = fetch_data("WSAEnlilSimulations", wind_params)
    wind_columns = {"time21_5": "time"}
    wind_nested_col = {"cmeInputs": {"cmeStartTime": "time", "speed": "speed"}}
    solar_wind_df = prepare_dataframe(solar_wind, wind_columns, wind_nested_col, "time")

    while True:
        print("\nChoose an option to visualize:")
        print("1. Solar Flares")
        print("2. Geomagnetic Storms")
        print("3. Solar Wind")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")
        if choice == "1":
            plot_data(solar_flares_df, "time", "intensity", "Solar Flares", "Time", "Intensity", kind="bar", color="orange")
        elif choice == "2":
            plot_data(geomagnetic_storms_df, "time", "kp_index", "Geomagnetic Storms", "Start Time", "KP Index", kind="line", color="blue")
        elif choice == "3":
            plot_data(solar_wind_df, "time", "speed", "Solar Wind Speeds", "Time", "Speed (km/s)", kind="line", color="green")
        elif choice == "4":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
