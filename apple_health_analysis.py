"""
apple_health_analysis.py

This script processes and analyzes Apple Health data exported from an iOS device.

Features:
1. Parses the Apple Health export XML (export.xml) to extract health records.
2. Reads ECG CSV files from the 'electrocardiograms' folder.
3. Reads GPX workout routes from the 'workout-routes' folder.
4. Analyzes and visualizes daily energy burned since January 1, 2025, with a 7-day rolling average.
5. Analyzes and visualizes daily sleep duration since January 1, 2025, with a 7-day rolling average.

Usage:
- Ensure the paths to the export.xml, ECG folder, and GPX folder are correctly set.
- Run the script to generate visualizations and summaries of the data.
- The plots are saved to the desktop as PNG files.

Dependencies:
- xmltodict
- pandas
- gpxpy
- matplotlib

"""

import xmltodict
import pandas as pd
import gpxpy
import glob
import matplotlib.pyplot as plt
from pytz import timezone
import xml.etree.ElementTree as ET
import argparse

# Define the PST timezone
pst = timezone('US/Pacific')

def parse_apple_health_export_xml(xml_path):
    """
    Reads Apple Health export XML and returns a DataFrame of records.
    """
    with open(xml_path, 'r', encoding='utf-8') as file:
        xml_data = file.read()

    # Convert XML to a nested Python dict
    data_dict = xmltodict.parse(xml_data)
    health_data = data_dict.get('HealthData', {})
    records = health_data.get('Record', [])

    parsed_records = []
    for record in records:
        parsed_records.append({
            'type': record.get('@type'),
            'startDate': record.get('@startDate'),
            'endDate': record.get('@endDate'),
            'value': record.get('@value'),
            'unit': record.get('@unit'),
            'sourceName': record.get('@sourceName')
        })

    df = pd.DataFrame(parsed_records)
    return df

def parse_ecg_csvs(ecg_folder_path):
    """
    Reads all CSV files in the ECG folder and returns a combined DataFrame.
    """
    csv_files = glob.glob(ecg_folder_path + "/*.csv")
    df_list = []

    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file)
        # If needed, rename columns or parse date columns here
        df_list.append(temp_df)

    if df_list:
        combined_ecg_df = pd.concat(df_list, ignore_index=True)
    else:
        combined_ecg_df = pd.DataFrame()

    return combined_ecg_df

def parse_gpx_files(gpx_folder_path):
    """
    Reads all .gpx route files in the folder and returns a list of DataFrames
    (one DataFrame per route). Each DataFrame includes lat, lon, elevation, time.
    """
    gpx_files = glob.glob(gpx_folder_path + "/*.gpx")
    route_dfs = []

    for gpx_file in gpx_files:
        with open(gpx_file, 'r', encoding='utf-8') as file:
            gpx_data = gpxpy.parse(file)

        route_points = []
        for track in gpx_data.tracks:
            for segment in track.segments:
                for point in segment.points:
                    route_points.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'time': point.time
                    })

        route_df = pd.DataFrame(route_points)
        route_df['filename'] = gpx_file
        route_dfs.append(route_df)

    return route_dfs

def generate_sleep_csv(health_df):
    sleep_data = health_df[(health_df['type'] == 'HKCategoryTypeIdentifierSleepAnalysis') & (health_df['startDate'] >= '2025-01-01')]
    
    # Ensure startDate and endDate are converted to datetime
    sleep_data['startDate'] = pd.to_datetime(sleep_data['startDate'], errors='coerce')
    sleep_data['endDate'] = pd.to_datetime(sleep_data['endDate'], errors='coerce')
    
    # Drop rows where date conversion failed
    sleep_data = sleep_data.dropna(subset=['startDate', 'endDate'])
    
    # Calculate duration safely
    try:
        time_diff = sleep_data['endDate'] - sleep_data['startDate']
        sleep_data['duration'] = time_diff.dt.total_seconds() / 3600.0
    except AttributeError:
        print("Using fallback method for duration calculation in generate_sleep_csv")
        sleep_data['duration'] = sleep_data.apply(
            lambda row: (row['endDate'] - row['startDate']).total_seconds() / 3600.0 
            if pd.notnull(row['startDate']) and pd.notnull(row['endDate']) else None, 
            axis=1
        )
    
    # Extract date safely
    try:
        sleep_data['date'] = sleep_data['startDate'].dt.date
    except AttributeError:
        print("Using fallback method for date extraction in generate_sleep_csv")
        sleep_data['date'] = sleep_data['startDate'].astype(str).str[:10]
    
    # Combine sleep sessions by date
    combined_sleep_data = sleep_data.groupby('date').agg(
        start_time=('startDate', 'min'),
        end_time=('endDate', 'max'),
        total_duration=('duration', 'sum')
    ).reset_index()
    
    # Convert start_time and end_time to PST safely
    def convert_to_pst(dt):
        try:
            if pd.notnull(dt):
                return dt.tz_convert(pst).strftime('%Y-%m-%d %H:%M:%S')
            else:
                return 'Unknown'
        except Exception as e:
            print(f"Error converting to PST: {e}")
            return 'Unknown'
    
    combined_sleep_data['start_time'] = combined_sleep_data['start_time'].apply(convert_to_pst)
    combined_sleep_data['end_time'] = combined_sleep_data['end_time'].apply(convert_to_pst)
    
    # Save combined sleep sessions to CSV
    combined_sleep_data.to_csv('sleep.csv', index=False)
    print("sleep.csv generated.")

def generate_workout_csv(xml_path, output_csv_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # List to store workout data
    workout_data = []

    # Iterate over each Workout element
    for workout in root.findall('Workout'):
        workout_type = workout.get('workoutActivityType')
        duration_minutes = float(workout.get('duration'))
        source_name = workout.get('sourceName')
        source_version = workout.get('sourceVersion')
        device_info = workout.get('device')
        creation_date = workout.get('creationDate')
        start_date = workout.get('startDate', 'Unknown')
        end_date = workout.get('endDate', 'Unknown')

        # Extract device details from the device_info string
        if device_info:
            device_details = device_info.split(', ')
            try:
                device_name = device_details[0].split(': ')[1]
                device_manufacturer = device_details[1].split(': ')[1]
                device_model = device_details[2].split(': ')[1]
                device_hardware = device_details[3].split(': ')[1]
                device_software = device_details[4].split(': ')[1]
                device_creation_date = device_details[5].split(': ')[1]
            except IndexError:
                device_name = device_manufacturer = device_model = device_hardware = device_software = device_creation_date = 'Unknown'
        else:
            device_name = device_manufacturer = device_model = device_hardware = device_software = device_creation_date = 'Unknown'

        # Extract metadata
        metadata = {entry.get('key'): entry.get('value') for entry in workout.findall('MetadataEntry')}
        indoor_workout = metadata.get('HKIndoorWorkout', '0')
        fitness_plus_session = metadata.get('HKMetadataKeyAppleFitnessPlusSession', '0')
        timezone = metadata.get('HKTimeZone', '')
        weather_humidity = metadata.get('HKWeatherHumidity', '')
        weather_temperature = metadata.get('HKWeatherTemperature', '')
        average_mets = metadata.get('HKAverageMETs', '')

        # Extract workout statistics
        stats = {stat.get('type'): stat.get('sum') for stat in workout.findall('WorkoutStatistics')}
        active_energy_burned = stats.get('HKQuantityTypeIdentifierActiveEnergyBurned', '0')
        average_heart_rate = stats.get('HKQuantityTypeIdentifierHeartRate', '0')
        min_heart_rate = stats.get('HKQuantityTypeIdentifierHeartRate', '0')
        max_heart_rate = stats.get('HKQuantityTypeIdentifierHeartRate', '0')
        basal_energy_burned = stats.get('HKQuantityTypeIdentifierBasalEnergyBurned', '0')

        # Ensure startDate and endDate are correctly parsed and added to the DataFrame
        workout_data.append({
            'workout_type': workout_type,
            'duration_minutes': duration_minutes,
            'source_name': source_name,
            'source_version': source_version,
            'device_name': device_name,
            'device_manufacturer': device_manufacturer,
            'device_model': device_model,
            'device_hardware': device_hardware,
            'device_software': device_software,
            'device_creation_date': device_creation_date,
            'creation_date': creation_date,
            'start_date': start_date,
            'end_date': end_date,
            'indoor_workout': indoor_workout,
            'fitness_plus_session': fitness_plus_session,
            'timezone': timezone,
            'weather_humidity': weather_humidity,
            'weather_temperature': weather_temperature,
            'average_mets': average_mets,
            'active_energy_burned': active_energy_burned,
            'average_heart_rate': average_heart_rate,
            'min_heart_rate': min_heart_rate,
            'max_heart_rate': max_heart_rate,
            'basal_energy_burned': basal_energy_burned
        })

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(workout_data)
    
    # Filter workouts to include only those starting from 2025-01-01
    df = df[df['start_date'] >= '2025-01-01']
    
    # Ensure startDate and endDate are correctly parsed and added to the DataFrame
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    
    # Drop rows where date conversion failed
    df = df.dropna(subset=['start_date', 'end_date'])
    
    # Convert to PST timezone safely 
    def convert_to_pst_str(dt):
        try:
            if pd.notnull(dt):
                return dt.tz_convert(pst).strftime('%Y-%m-%d %H:%M:%S')
            else:
                return 'Unknown'
        except Exception as e:
            print(f"Error converting to PST: {e}")
            return 'Unknown'
    
    df['start_time'] = df['start_date'].apply(convert_to_pst_str)
    df['end_time'] = df['end_date'].apply(convert_to_pst_str)
    
    df.to_csv(output_csv_path, index=False)
    print(f"Workout data saved to {output_csv_path}")

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Apple Health data.')
    parser.add_argument('xml_file_path', type=str, help='Path to the Apple Health export XML file')
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = parse_arguments()
    xml_file_path = args.xml_file_path

    # Use xml_file_path in the parsing function
    health_df = parse_apple_health_export_xml(xml_file_path)
    print("Health records loaded:", len(health_df))

    # 2) Parse ECG data CSVs
    ecg_df = parse_ecg_csvs("electrocardiograms")
    print("ECG records loaded:", len(ecg_df))

    # 3) Parse GPX workout routes
    route_dataframes = parse_gpx_files("workout-routes")
    print("Total GPX routes loaded:", len(route_dataframes))

    #--- EXAMPLE: Summaries and Quick Analyses ---

    # Example A: Count how many records of each type in health_df
    if not health_df.empty:
        print("\nTop 5 record types:")
        print(health_df['type'].value_counts().head())

    # Example C: If we have at least one GPX route, preview the first few lines
    if route_dataframes:
        first_route = route_dataframes[0]
        print("\nSample GPX route data (first 5 points):")
        print(first_route.head())

    # Example D: Summarize daily energy burned since 2025/1/1
    energy_data = health_df[(health_df['type'] == 'HKQuantityTypeIdentifierActiveEnergyBurned') & (health_df['startDate'] >= '2025-01-01')]
    if not energy_data.empty:
        # Convert 'value' to numeric
        energy_data['value'] = pd.to_numeric(energy_data['value'], errors='coerce')
        
        # Print data types for debugging
        print("\nData types before conversion:")
        print(energy_data.dtypes)
        
        # Ensure startDate is converted to datetime and handle errors robustly
        energy_data['startDate'] = pd.to_datetime(energy_data['startDate'], errors='coerce')
        
        # Check if any rows failed conversion
        null_dates = energy_data['startDate'].isnull().sum()
        if null_dates > 0:
            print(f"\nWarning: {null_dates} rows had startDate that could not be converted to datetime")
            # Drop rows where startDate could not be converted to datetime
            energy_data = energy_data.dropna(subset=['startDate'])
            
        # Print data types after conversion
        print("\nData types after conversion:")
        print(energy_data.dtypes)
        
        # Extract date using string manipulation as a fallback method
        try:
            # Try using the .dt accessor first
            energy_data['date'] = energy_data['startDate'].dt.date
        except AttributeError:
            # If .dt accessor fails, try string manipulation
            print("\nUsing fallback method for date extraction")
            energy_data['date'] = energy_data['startDate'].astype(str).str[:10]
            
        # Verify the date column was created
        print("\nDate column created successfully:", 'date' in energy_data.columns)
        
        if 'date' in energy_data.columns:
            print("Sample dates:", energy_data['date'].head().tolist())
            
            # Continue with rest of processing
            daily_energy = energy_data.groupby('date')['value'].sum().reset_index()
            print(f"\nTotal energy burned since 2025/1/1: {daily_energy['value'].sum()}")
            
            # Quick bar plot of daily energy burned
            plt.figure(figsize=(12, 6))
            plt.bar(daily_energy['date'].astype(str), daily_energy['value'], color='#f5cba7', label='Daily Energy Burned')

            # Add a 7-day rolling average line
            daily_energy['7_day_avg'] = daily_energy['value'].rolling(window=7).mean()
            plt.plot(daily_energy['date'].astype(str), daily_energy['7_day_avg'], color='blue', label='7-Day Average')

            plt.title("Daily Energy Burned Since 2025/1/1")
            plt.xlabel("Date")
            plt.ylabel("Energy (kcal)")

            # Rotate x-axis labels and show one label per week
            plt.xticks(ticks=daily_energy.index[::7], labels=daily_energy['date'].astype(str)[::7], rotation=45)

            plt.legend()
            plt.tight_layout()
            plt.savefig('daily_energy_burned.png')
            # plt.show(block=True)

    # Example E: Summarize sleep data
    sleep_data = health_df[(health_df['type'] == 'HKCategoryTypeIdentifierSleepAnalysis') & (health_df['startDate'] >= '2025-01-01')]
    if not sleep_data.empty:
        # Print data types for debugging
        print("\nSleep data types before conversion:")
        print(sleep_data.dtypes)
        
        # Ensure startDate is converted to datetime
        sleep_data['startDate'] = pd.to_datetime(sleep_data['startDate'], errors='coerce')
        # Drop rows where startDate could not be converted to datetime
        sleep_data = sleep_data.dropna(subset=['startDate'])
        sleep_data['endDate'] = pd.to_datetime(sleep_data['endDate'], errors='coerce')
        # Drop rows where endDate could not be converted to datetime
        sleep_data = sleep_data.dropna(subset=['endDate'])
        
        # Print data types after conversion
        print("\nSleep data types after conversion:")
        print(sleep_data.dtypes)
        
        # Print sample of datetime values
        print("\nSample startDate values:")
        print(sleep_data['startDate'].head())
        print("\nSample endDate values:")
        print(sleep_data['endDate'].head())
        
        # Calculate duration using a safer approach
        try:
            # Try using timedelta approach
            time_diff = sleep_data['endDate'] - sleep_data['startDate']
            print("\nTime difference data type:", type(time_diff))
            sleep_data['duration'] = time_diff.dt.total_seconds() / 3600.0
        except AttributeError:
            # If .dt accessor fails, use a fallback method
            print("\nUsing fallback method for duration calculation")
            # Calculate duration in a safer way by explicitly comparing timestamps
            def calculate_duration(row):
                try:
                    if pd.notnull(row['startDate']) and pd.notnull(row['endDate']):
                        return (row['endDate'] - row['startDate']).total_seconds() / 3600.0
                    else:
                        return None
                except Exception as e:
                    print(f"Error calculating duration: {e}")
                    return None
            
            sleep_data['duration'] = sleep_data.apply(calculate_duration, axis=1)
        
        # Extract date using a safer approach
        try:
            # Try using the .dt accessor first
            sleep_data['date'] = sleep_data['startDate'].dt.date
        except AttributeError:
            # If .dt accessor fails, try string manipulation
            print("\nUsing fallback method for date extraction in sleep data")
            sleep_data['date'] = sleep_data['startDate'].astype(str).str[:10]
        
        # Verify the date and duration columns were created
        print("\nDate column created:", 'date' in sleep_data.columns)
        print("Duration column created:", 'duration' in sleep_data.columns)
        
        if 'date' in sleep_data.columns and 'duration' in sleep_data.columns:
            # Group by date and sum durations
            daily_sleep = sleep_data.groupby('date')['duration'].sum().reset_index()
            print(f"\nTotal sleep records: {len(sleep_data)}")
            print(f"Average sleep duration: {daily_sleep['duration'].mean():.2f} hours")

            # Quick bar plot of daily sleep duration
            plt.figure(figsize=(12, 6))
            plt.bar(daily_sleep['date'].astype(str), daily_sleep['duration'], color='#a3c1ad', label='Daily Sleep Duration')

            # Add a 7-day rolling average line
            daily_sleep['7_day_avg'] = daily_sleep['duration'].rolling(window=7).mean()
            plt.plot(daily_sleep['date'].astype(str), daily_sleep['7_day_avg'], color='blue', label='7-Day Average')

            plt.title("Daily Sleep Duration since 2025/1/1")
            plt.xlabel("Date")
            plt.ylabel("Hours")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('daily_sleep_duration.png')
            # plt.show(block=True)

    generate_sleep_csv(health_df)
    generate_workout_csv(xml_file_path, 'workout_data.csv')