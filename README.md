# Apple Health Data Analysis

This project provides a comprehensive analysis of Apple Health data exported from iOS devices. It processes the Apple Health export XML file to extract, analyze, and visualize various health metrics including energy burned, sleep patterns, and workout sessions.

## Features

- Parses the Apple Health export XML to extract health records
- Analyzes and visualizes daily energy burned since January 1, 2025, with a 7-day rolling average
- Analyzes and visualizes daily sleep duration since January 1, 2025, with a 7-day rolling average
- Combines sleep sessions to provide start and end times for each day
- Processes workout data with start and end times
- Generates CSV files for sleep and workout data

## Prerequisites

To run this project, you'll need:

1. **Python 3.6+**: The script is written in Python and requires version 3.6 or higher.
2. **Required Python packages**:
   - xmltodict: For parsing XML data
   - pandas: For data manipulation
   - matplotlib: For data visualization
   - pytz: For timezone handling
3. **Apple Health Export**: You need to export your Apple Health data from your iOS device.

## Setup Instructions

1. **Export Apple Health Data from your iOS device**:
   - Open the Health app on your iOS device
   - Tap on your profile picture in the top-right corner
   - Tap on "Export All Health Data" at the bottom
   - The export will be created as a zip file
   - Transfer this zip file to your computer
   - Extract the zip file to access the "export.xml" file

2. **Clone this repository**:
   ```
   git clone https://github.com/yourusername/apple_health_analysis.git
   cd apple_health_analysis
   ```

3. **Create and activate a virtual environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install required packages**:
   ```
   pip install xmltodict pandas matplotlib pytz
   ```

5. **Prepare your data**:
   - Place your "export.xml" file in a known location

## How to Run

Run the script by providing the path to your Apple Health export XML file:

```
python apple_health_analysis.py /path/to/your/export.xml
```

Replace `/path/to/your/export.xml` with the actual path to your export.xml file.

## Output

The script generates the following output:

1. **CSV Files**:
   - `sleep.csv`: Contains combined sleep sessions with start time, end time, and total duration
   - `workout_data.csv`: Contains workout data with details like type, duration, and energy burned

2. **Visualizations**:
   - `daily_energy_burned.png`: A bar chart showing daily energy burned with a 7-day rolling average
   - `daily_sleep_duration.png`: A bar chart showing daily sleep duration with a 7-day rolling average


## Notes

- Only data from January 1, 2025 onwards is processed
- Sleep sessions are combined to provide a single start time and end time for each day
- Workout sessions are also combined with start and end times
