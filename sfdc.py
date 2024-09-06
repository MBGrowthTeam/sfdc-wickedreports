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
    
    query = f"SELECT {', '.join(fields)} FROM {object_name} LIMIT 10"
    try:
        result = sf.query(query)
        records = result['records']
        df = pd.DataFrame(records).drop(columns=['attributes'])
        return df
    except Exception as e:
        st.error(f"Error fetching data for {object_name}: {str(e)}")
        return pd.DataFrame()

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
        selected_object = st.selectbox("Select a Salesforce Object", st.session_state.sf_objects)

        if st.button(f"Generate Rows for {selected_object}"):
            sf = st.session_state.sf_connection
            if sf:
                with st.spinner(f"Fetching data for {selected_object}..."):
                    df = fetch_object_data(sf, selected_object)
                    if not df.empty:
                        st.write(f"First 10 rows of data for {selected_object}:")
                        st.dataframe(df)
                    else:
                        st.write(f"No data available for {selected_object} or unable to retrieve data.")

if __name__ == "__main__":
    main()