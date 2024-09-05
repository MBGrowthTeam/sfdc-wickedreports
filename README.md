# Salesforce Data Export Dashboard

## Description

This project provides a Streamlit application to export and visualize Salesforce order data. It connects to Salesforce using the simple_salesforce library, retrieves order data within a specified date range, processes the data, and displays it in an interactive dashboard with various charts and tables.

## Features

- **Data Export:** Fetches order data from Salesforce, including order details, customer information, and product details.
- **Data Processing:** Cleans and transforms the raw Salesforce data into a standardized format for analysis.
- **Interactive Dashboard:** Presents the data in a user-friendly Streamlit dashboard.
- **Visualizations:** Includes charts such as:
    - Orders by State (bar chart)
    - Orders by Product Family (pie chart)
    - Daily Order Totals with Moving Average (line chart)
    - Order Frequency Heatmap (heatmap)
- **Data Table:** Displays the processed data in a tabular format.
- **CSV Export:** Allows users to download the processed data as a CSV file.

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:MBGrowthTeam/sfdc-wickedreports.git
   ```

2. **Create and activate a Conda environment:**
   ```bash
   conda create -n salesforce_export_env python=3.9 -y
   conda activate salesforce_export_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Salesforce credentials:**
   - Create a `.env` file in the project's root directory.
   - Add your Salesforce credentials to the `.env` file:
     ```
     SF_USERNAME=your_username
     SF_PASSWORD=your_password
     SF_TOKEN=your_security_token
     SF_CONSUMER_KEY=your_consumer_key
     SF_CONSUMER_SECRET=your_consumer_secret
     ```

## Usage

1. **Activate the Conda environment:**
   ```bash
   conda activate salesforce_export_env
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Use the dashboard:**
   - Select the desired date range.
   - Click "Fetch Data" to retrieve and display the Salesforce data.
   - Interact with the charts and tables.
   - Download the data as a CSV file using the "Download CSV" button.

## Version

0.1.0

## Author

Jonathan Nelson

## License

MIT License
