import streamlit as st
import pandas as pd
from simple_salesforce import Salesforce
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Salesforce credentials
SF_USERNAME = os.getenv("SF_USERNAME")
SF_PASSWORD = os.getenv("SF_PASSWORD")
SF_TOKEN = os.getenv("SF_TOKEN")
SF_CONSUMER_KEY = os.getenv("SF_CONSUMER_KEY")
SF_CONSUMER_SECRET = os.getenv("SF_CONSUMER_SECRET")

# List of primary objects
PRIMARY_OBJECTS = [
    "Account",
    "Asset",
    "Case",
    "Contact",
    "Contract",
    "Lead",
    "Opportunity",
    "Order",
    "Product2",
    "Quote",
    "SBQQ__Subscription__c", 
    "WeGather_Project__c", 
    "WeShare_Site__c"
]

# Initialize session state for memory
if "sf_connection" not in st.session_state:
    st.session_state.sf_connection = None
if "sf_objects" not in st.session_state:
    st.session_state.sf_objects = None

def connect_to_salesforce():
    if st.session_state.sf_connection is None:
        try:
            sf = Salesforce(
                username=SF_USERNAME,
                password=SF_PASSWORD,
                security_token=SF_TOKEN,
                consumer_key=SF_CONSUMER_KEY,
                consumer_secret=SF_CONSUMER_SECRET,
            )
            st.session_state.sf_connection = sf
            return sf
        except Exception as e:
            st.error(f"Failed to connect to Salesforce: {str(e)}")
            return None
    return st.session_state.sf_connection

def get_salesforce_objects(sf):
    if st.session_state.sf_objects is None:
        try:
            objects = sf.describe()["sobjects"]
            object_names = [obj["name"] for obj in objects]
            st.session_state.sf_objects = object_names
            return object_names
        except Exception as e:
            st.error(f"Error fetching Salesforce objects: {str(e)}")
            return []
    return st.session_state.sf_objects

def get_object_fields(sf, object_name):
    try:
        describe = getattr(sf, object_name).describe()
        fields = [field["name"] for field in describe["fields"]]
        return fields
    except Exception as e:
        st.error(f"Error fetching fields for {object_name}: {str(e)}")
        return []

def fetch_object_data(sf, object_name):
    fields = get_object_fields(sf, object_name)
    if not fields:
        return pd.DataFrame()
    
    query = f"SELECT {', '.join(fields)} FROM {object_name} LIMIT 100"
    try:
        result = sf.query(query)
        records = result['records']
        df = pd.DataFrame(records).drop(columns=['attributes'])
        return df
    except Exception as e:
        st.error(f"Error fetching data for {object_name}: {str(e)}")
        return pd.DataFrame()

# Function to generate SQL for all primary objects
def generate_combined_sql():
    combined_sql = ""
    sf = st.session_state.sf_connection

    if sf:
        for obj in PRIMARY_OBJECTS:
            fields = get_object_fields(sf, obj)
            if fields:
                # Create SQL for each object with start and end comments
                sql_query = f"/* Start of {obj} SQL */\nSELECT {', '.join(fields)} FROM {obj};\n/* End of {obj} SQL */\n\n"
                combined_sql += sql_query
        
        # Display SQL
        st.code(combined_sql, language='sql')

        # Button to export SQL to a file
        st.download_button(
            label="Download Combined SQL",
            data=combined_sql,
            file_name="combined_primary_objects.sql",
            mime="text/sql"
        )
    else:
        st.error("Not connected to Salesforce!")

def main():
    st.set_page_config(layout="wide", page_title="SFDC Object Explorer")
    st.title("Salesforce Object Explorer")

    # Connect to Salesforce and fetch objects
    if st.button("Connect to Salesforce and Fetch Objects"):
        with st.spinner("Connecting to Salesforce and fetching objects..."):
            sf = connect_to_salesforce()
            if sf:
                objects = get_salesforce_objects(sf)
                if objects:
                    st.success("Connected to Salesforce and fetched objects successfully!")
                else:
                    st.error("Failed to fetch Salesforce objects.")

    # Display objects and allow selection
    if st.session_state.sf_objects:
        object_type = st.radio("Select Object Type:", ("All Objects", "Primary Objects"))

        if object_type == "All Objects":
            display_objects = st.session_state.sf_objects
        else:
            display_objects = [obj for obj in st.session_state.sf_objects if obj in PRIMARY_OBJECTS]

        selected_object = st.selectbox("Select a Salesforce Object", display_objects)

        if st.button(f"Generate Rows for {selected_object}"):
            sf = st.session_state.sf_connection
            if sf:
                with st.spinner(f"Fetching data for {selected_object}..."):
                    df = fetch_object_data(sf, selected_object)
                    if not df.empty:
                        st.write(f"First 100 rows of data for {selected_object}:")
                        st.dataframe(df)
                    else:
                        st.write(f"No data available for {selected_object} or unable to retrieve data.")
    
    # Button to generate and download combined SQL
    st.markdown("## Export Combined SQL for Primary Objects")
    if st.button("Generate Combined SQL"):
        generate_combined_sql()

if __name__ == "__main__":
    main()