# Frontend/UI Libraries
import streamlit as st  # Streamlit for building interactive web apps

# Security
from passlib.hash import bcrypt # Hashlib for hashing sensitive information

# Set the page configuration as the first Streamlit command
st.set_page_config(layout="wide", page_title="Salesforce Data Dashboard")

# Data Manipulation/Analysis
import pandas as pd  # Pandas for data manipulation and analysis

# Fuzzy Logic
from fuzzywuzzy import process, fuzz

# Data Visualization
import plotly.express as px  # Plotly Express for quick, interactive plots
import plotly.graph_objects as go  # Plotly Graph Objects for detailed, customizable visualizations

# Salesforce Integration
from simple_salesforce import Salesforce  # Simple Salesforce for interacting with Salesforce API

# Date and Time Handling
from datetime import datetime, timedelta, timezone  # DateTime and Timedelta for manipulating dates and times
import pytz  # Pytz for timezone handling
import json # Structured data

# Environment Management
from dotenv import load_dotenv  # Load environment variables from a .env file
import os  # OS for interacting with the operating system, managing environment variables, file paths, etc.
import time

# Type Hinting and Annotations
from typing import Optional, Any, Literal  # Typing for type annotations, enhancing code readability and static analysis

# Concurrency
import threading  # Threading for running concurrent operations

# Type Hinting and Annotations
from typing import Dict  # Typing for type annotations, enhancing code readability and static analysis

# Specialized Libraries
import dspy  # (Assumed to be a specialized data science or signal processing library)

# Configuration Parsing
import toml  # TOML for reading and parsing configuration files

# Color Management
import colorsys

# Add Logging
import logging

# Set up logging configuration
logging.basicConfig(
    filename='app.log',  # Log to a file called 'app.log'
    level=logging.DEBUG,  # Log detailed information for debugging
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp and log level
)

# Load environment variables from .env file
load_dotenv()

# Load secrets from .streamlit/secrets.toml
secrets = st.secrets

# Colors
primary_color = "#F63366"  # Streamlit's primary color
background_color = "#FFFFFF"  # Main background color
secondary_background_color = "#F0F2F6"  # Secondary background color
text_color = "#262730"  # Text color

# Salesforce credentials from secrets
SF_USERNAME = secrets["salesforce"]["SF_USERNAME"]
SF_PASSWORD = secrets["salesforce"]["SF_PASSWORD"]
SF_TOKEN = secrets["salesforce"]["SF_TOKEN"]
SF_CONSUMER_KEY = secrets["salesforce"]["SF_CONSUMER_KEY"]
SF_CONSUMER_SECRET = secrets["salesforce"]["SF_CONSUMER_SECRET"]

# OpenAI API key from secrets
OPENAI_API_KEY = secrets["openai"]["api_key"]

# Load Salesforce object and field data
with open('salesforce_object_data.json', 'r') as f:
    salesforce_data = json.load(f)

# Load the mapping from the JSON file
with open('salesforce_object_mapping.json', 'r') as f:
    object_aliases = json.load(f)

def check_password():
    """Returns `True` if the user had a correct password."""
    
    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "password" not in st.session_state:
        st.session_state.password = ""
    if "submit_clicked" not in st.session_state:
        st.session_state.submit_clicked = False

    def login():
        st.session_state.submit_clicked = True

    if st.session_state.authenticated:
        return True

    # Input placeholders for email and password
    email_placeholder = st.empty()
    password_placeholder = st.empty()
    submit_placeholder = st.empty()

    email = email_placeholder.text_input("Email", key="email_input", value=st.session_state.username)
    password = password_placeholder.text_input("Password", type="password", key="password_input", value=st.session_state.password)
    submit_button = submit_placeholder.button("Login", on_click=login)

    if st.session_state.submit_clicked:
        try:
            logging.debug(f"Login attempt for email: {email}")
            
            # Check if the email exists in the secrets file
            if email in st.secrets["credentials"]["usernames"]:
                stored_password = st.secrets["credentials"]["passwords"][email]
                logging.debug(f"Stored password for {email}: {stored_password}")
                
                # Verify the password by comparing directly
                if password == stored_password:
                    st.session_state.authenticated = True
                    st.session_state.username = email
                    st.session_state.password = ""
                    
                    # Clear input placeholders after successful login
                    email_placeholder.empty()
                    password_placeholder.empty()
                    submit_placeholder.empty()
                    
                    logging.info(f"User {email} successfully logged in.")
                    return True
                else:
                    st.error("😕 Incorrect password")
                    logging.warning(f"Incorrect password attempt for {email}")
            else:
                st.error("😕 User not found")
                logging.warning(f"Login attempt with non-existent user: {email}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error during login attempt for {email}: {str(e)}", exc_info=True)
        
        st.session_state.submit_clicked = False
        return False

    return False

def connect_to_salesforce():
    """
    Establish a connection to Salesforce using the provided credentials.

    Returns:
        A Salesforce connection object, or None if the connection fails.
    """
    try:
        sf = Salesforce(
            username=SF_USERNAME,
            password=SF_PASSWORD,
            security_token=SF_TOKEN,
            consumer_key=SF_CONSUMER_KEY,
            consumer_secret=SF_CONSUMER_SECRET,
        )
        return sf
    except Exception as e:
        st.error(f"Failed to connect to Salesforce: {str(e)}")
        return None


class OpenAIModel(dspy.OpenAI):
    """
    A wrapper class for dspy.OpenAI that adds token usage logging.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAIModel.

        Args:
            model: The name of the OpenAI model to use.
            api_key: The OpenAI API key.
            model_type: The type of the model ("chat" or "text").
            **kwargs: Additional keyword arguments to pass to dspy.OpenAI.
        """
        super().__init__(model=model, api_key=api_key, model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """
        Log the total tokens used from the OpenAI API response.

        Args:
            response: The response from the OpenAI API.
        """
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """
        Get the total tokens used and reset the token usage counters.

        Returns:
            A dictionary containing the prompt and completion token usage for the model.
        """
        usage = {
            self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Query the OpenAI model and track token usage.

        Args:
            prompt: The prompt to send to the OpenAI model.
            only_completed: Whether to return only completed choices.
            return_sorted: Whether to return the choices sorted by score.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.

        Returns:
            A list of completions from the OpenAI model.
        """
        response = self.request(prompt, **kwargs)
        self.log_usage(response)
        choices = response["choices"]
        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = sorted(
                [
                    (
                        sum(c["logprobs"]["token_logprobs"]) / len(c["logprobs"]["token_logprobs"]),
                        self._get_choice_text(c),
                    )
                    for c in choices
                ],
                reverse=True,
            )
            completions = [c for _, c in scored_completions]

        return completions


# Initialize the OpenAI model
openai_model = OpenAIModel(api_key=OPENAI_API_KEY)


def map_product_family(product_name):
    """
    Classify the product name into a predefined product family using OpenAI.

    Args:
        product_name: The name of the product to classify.

    Returns:
        The product family the product belongs to, or "Other" if it cannot be classified.
    """
    prompt = (
        f"Classify the product '{product_name}' into one of the following categories: People, Giving, "
        f"Mobile App, Websites, Streaming, Service Planning, Accounting, Safety, Media, Communications. "
        f"Important: Your reply will only contain the product category name."
    )
    response = openai_model(prompt)
    return response[0].strip() if response else "Other"

# --- SQL Queries --- 
def load_sql_query(filename):
    """Loads a SOQL query from a file, ignoring docstrings."""
    with open(os.path.join("sql", filename), 'r') as f:
        lines = f.readlines()
        query_lines = []
        in_docstring = False
        for line in lines:
            if line.strip().startswith('"""'):
                in_docstring = not in_docstring  # Toggle docstring flag
            elif not in_docstring:
                query_lines.append(line)
        return "".join(query_lines).strip()

def get_brand_id(salesforce, brand_name):
    if brand_name == "ALL":
        return None  # Don't filter by brand if "ALL" is selected
    query = f"SELECT Id FROM Brand__c WHERE Name = '{brand_name}'"
    result = salesforce.query(query)
    if result['records']:
        return result['records'][0]['Id']
    else:
        return None

def fetch_orders_data(sf, start_date, end_date, brand=None):
    """
    Fetch order data from Salesforce within a specified date range,
    optionally filtered by brand.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.
        brand: The brand to filter by (optional).

    Returns:
        A Pandas DataFrame containing the order data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    query = load_sql_query("Order.sql").format(start_date=start_date_str, end_date=end_date_str)

    if brand: 
        if isinstance(brand, list):  # Check if brand is a list (for "ALL" selection)
            brand_str = ", ".join([f"'{b}'" for b in brand])  # Format for IN operator
            query += f" AND Order.Brand__c IN ({brand_str})"
        else:
            query += f" AND Order.Brand__c = '{brand}'"

    result = sf.query_all(query)
    
    # Check if any Orders were found
    if result['totalSize'] == 0:
        st.warning(f"No Orders found for {brand if brand else 'ALL'} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()  # Return an empty DataFrame

    orders = pd.DataFrame(result["records"])

    # Flatten nested order items data
    order_items = []
    for order in result["records"]:
        for item in order.get("OrderItems", {}).get("records", []):
            order_items.append(
                {
                    "OrderId": order["Id"],
                    "SBQQ__Subscription__c": item.get("SBQQ__Subscription__c"),
                    "Product_Category__c": item.get("Product_Category__c"),
                    "ProductName": item.get("Product_Name__c"),
                }
            )

    # Create an empty DataFrame if no OrderItems are found
    if order_items:
        order_items_df = pd.DataFrame(order_items)
    else:
        order_items_df = pd.DataFrame(columns=['OrderId', 'SBQQ__Subscription__c', 'Product_Category__c', 'ProductName'])

    # Now merge, ensuring all orders are included even if they have no OrderItems
    df = orders.merge(order_items_df, left_on="Id", right_on="OrderId", how="left")

    # Clean up the DataFrame
    df = df.drop(columns=["attributes", "OrderItems"])
    df["Account.Time_Zone__c"] = df["Account"].apply(lambda x: x["Time_Zone__c"] if x else None)
    df["Account.Account_Primary_Contact_Email__c"] = df["Account"].apply(
        lambda x: x["Account_Primary_Contact_Email__c"] if x else None
    )
    df = df.drop(columns=["Account"])

    return df

def fetch_mql_data(sf, start_date, end_date, brand_ids=None):
    """
    Fetch MQL (Marketing Qualified Leads) data from Salesforce within a 
    specified date range, optionally filtered by a list of brand IDs.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.
        brand_ids: A list of Salesforce brand IDs to filter by (optional).

    Returns:
        A Pandas DataFrame containing the MQL data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
    query = load_sql_query("MQL_Record__c.sql").format(start_date=start_date_str, end_date=end_date_str)

    if brand_ids:
        brand_id_list_str = ', '.join([f"'{b}'" for b in brand_ids])
        query += f" AND Brand__c IN ({brand_id_list_str})"

    result = sf.query_all(query)
    df = pd.DataFrame(result["records"]).drop(columns=["attributes"])
    return df

def fetch_campaign_data(sf, start_date, end_date):
    """
    Fetch data from the Campaign object within the specified date range.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.

    Returns:
        A Pandas DataFrame containing the Campaign data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    query = load_sql_query("Campaign.sql").format(start_date=start_date_str, end_date=end_date_str)

    result = sf.query_all(query)
    df = pd.DataFrame(result["records"]).drop(columns=["attributes"])
    return df


def process_mql_data(df):
    """
    Process the MQL data and handle timezone issues.

    Args:
        df: Pandas DataFrame containing the MQL data.

    Returns:
        A Pandas DataFrame with processed MQL data.
    """
    # Convert dates to datetime objects
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    df["MQL_Converted_Date__c"] = pd.to_datetime(df["MQL_Converted_Date__c"])

    # Localize to UTC only if naive (no timezone information)
    if df["CreatedDate"].dt.tz is None:
        df["CreatedDate"] = df["CreatedDate"].dt.tz_localize("UTC")
    # Convert to UTC if already tz-aware
    else:
        df["CreatedDate"] = df["CreatedDate"].dt.tz_convert("UTC")

    if df["MQL_Converted_Date__c"].dt.tz is None:
        df["MQL_Converted_Date__c"] = df["MQL_Converted_Date__c"].dt.tz_localize("UTC")
    else:
        df["MQL_Converted_Date__c"] = df["MQL_Converted_Date__c"].dt.tz_convert("UTC")

    # Calculate the age of leads
    now = pd.Timestamp.now(tz="UTC")
    df["LeadAge"] = (now - df["CreatedDate"]).dt.total_seconds() / (24 * 60 * 60)  # in days

    return df


def process_data(df):
    """
    Process the fetched data to standardize and clean it for analysis.

    Args:
        df: Pandas DataFrame containing the fetched data.

    Returns:
        A Pandas DataFrame with processed data.
    """
    date_column = "CreatedDate" if "CreatedDate" in df.columns else "ORDERDATETIME"
    df[date_column] = pd.to_datetime(df[date_column])

    def get_utc_offset(timezone_str):
        """
        Get the UTC offset from a timezone string.

        Args:
            timezone_str: The timezone string.

        Returns:
            The UTC offset as a string, e.g., "UTC+5".
        """
        try:
            tz = pytz.timezone(timezone_str)
            offset = tz.utcoffset(datetime.now(tz)).total_seconds() / 3600
            return f"UTC{int(offset):+d}"
        except Exception:
            return "UTC+0"

    column_mapping = {
        "Id": "ORDERID",
        date_column: "ORDERDATETIME",
        "Account.Time_Zone__c": "ORDERTIMEZONE",
        "TotalAmount": "ORDERTOTAL",
        "Account.Account_Primary_Contact_Email__c": "CUSTOMEREMAIL",
        "BillingState": "CUSTOMERSTATE",
        "BillingCountry": "CUSTOMERCOUNTRY",
        "SBQQ__Subscription__c": "SUBSCRIPTIONID",
        "ProductId": "PRODUCTID",
        "ProductName": "PRODUCTNAME",
        "CurrencyIsoCode": "CURRENCYCODE",
        "BillingCity": "CITY",
    }
    df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
    df["ORDERSYSTEM"] = "SFDC-API"
    df["ISREFUND"] = "No"
    df["IPADDRESS"] = ""
    df["ORDERTIMEZONE"] = df["ORDERTIMEZONE"].apply(get_utc_offset)
    df = df.dropna(subset=["CUSTOMEREMAIL", "SUBSCRIPTIONID"])

    # Map ProductFamily to predefined categories using OpenAI model
    if "PRODUCTNAME" in df.columns:
        df["ProductFamily"] = df["PRODUCTNAME"].apply(map_product_family)

    return df


def fetch_wicked_reports_data(sf, start_date, end_date, brand=None):
    """
    Fetches order data from Salesforce specifically for Wicked Reports export.

    Args:
        sf (Salesforce): The Salesforce connection object.
        start_date (datetime): The start date for the data retrieval.
        end_date (datetime): The end date for the data retrieval.
        brand: The brand to filter by (optional).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the fetched order data for Wicked Reports,
                      or an empty DataFrame if no orders are found or 'Id' is not available.
    """
    try:
        start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

        query = load_sql_query("Order.sql").format(start_date=start_date_str, end_date=end_date_str)

        if brand:
            if isinstance(brand, list):  # Check if brand is a list (for "ALL" selection)
                brand_str = ", ".join([f"'{b}'" for b in brand])  # Format for IN operator
                query += f" AND Brand__c IN ({brand_str})"
            else:
                query += f" AND Brand__c = '{brand}'"
        query += " ORDER BY CreatedDate"

        result = sf.query_all(query)
        orders = pd.DataFrame(result["records"])

        # Check if 'Id' column is present in the 'orders' DataFrame
        if 'Id' not in orders.columns:
            st.warning(f"The 'Id' column is not available for brand: {brand}. Skipping Wicked Reports data fetch.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Flatten the nested OrderItems data
        order_items = []
        for order in orders.to_dict(orient='records'):
            for item in order.get("OrderItems", {}).get("records", []):
                order_items.append(
                    {
                        "OrderId": order["Id"],
                        "SBQQ__Subscription__c": item.get("SBQQ__Subscription__c"),
                        "Product2Id": item.get("Product2Id"),
                        "Product_Name__c": item.get("Product_Name__c"),
                    }
                )

        # Create an empty DataFrame if no OrderItems are found
        if order_items:
            order_items_df = pd.DataFrame(order_items)
        else:
            order_items_df = pd.DataFrame(columns=['OrderId', 'SBQQ__Subscription__c', 'Product2Id', 'Product_Name__c'])

        # Merge orders with order items using a left join
        df = orders.merge(order_items_df, left_on="Id", right_on="OrderId", how="left")

        # Clean up the DataFrame
        df = df.drop(columns=["attributes", "OrderItems"], errors="ignore")
        df["Time_Zone__c"] = df["Account"].apply(lambda x: x["Time_Zone__c"] if isinstance(x, dict) and "Time_Zone__c" in x else None)
        df["Account_Primary_Contact_Email__c"] = df["Account"].apply(
            lambda x: x["Account_Primary_Contact_Email__c"] if isinstance(x, dict) and "Account_Primary_Contact_Email__c" in x else None
        )
        df = df.drop(columns=["Account"])

        return df

    except Exception as e:
        st.error(f"An error occurred while fetching Wicked Reports data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame


def prepare_wicked_reports_export(df):
    """
    Prepares the Salesforce data for export in Wicked Reports format.

    Args:
        df (pd.DataFrame): The raw Salesforce data fetched for Wicked Reports.

    Returns:
        pd.DataFrame: A DataFrame formatted for Wicked Reports export.
    """

    def get_utc_offset(timezone_str):
        """
        Get the UTC offset from a timezone string.

        Args:
            timezone_str: The timezone string.

        Returns:
            The UTC offset as a string, e.g., "UTC+5".
        """
        try:
            tz = pytz.timezone(timezone_str)
            offset = tz.utcoffset(datetime.now(tz)).total_seconds() / 3600
            return f"UTC{int(offset):+d}"
        except:
            return "UTC+0"  # Default to UTC if timezone is not recognized

    wicked_columns = [
        "ORDERID",
        "ORDERDATETIME",
        "ORDERTIMEZONE",
        "ORDERTOTAL",
        "CUSTOMEREMAIL",
        "CUSTOMERSTATE",
        "CUSTOMERCOUNTRY",
        "SUBSCRIPTIONID",
        "ORDERSYSTEM",
        "ISREFUND",
        "PRODUCTID",
        "PRODUCTNAME",
        "CURRENCYCODE",
        "CITY",
        "IPADDRESS",
    ]

    wicked_df = pd.DataFrame(columns=wicked_columns)

    wicked_df["ORDERID"] = df["Id"]
    wicked_df["ORDERDATETIME"] = pd.to_datetime(df["CreatedDate"]).dt.strftime("%m/%d/%Y %H:%M")
    wicked_df["ORDERTIMEZONE"] = df["Time_Zone__c"].apply(get_utc_offset)
    wicked_df["ORDERTOTAL"] = df["TotalAmount"].fillna(0).round(2)
    wicked_df["CUSTOMEREMAIL"] = df["Account_Primary_Contact_Email__c"]
    wicked_df["CUSTOMERSTATE"] = df["BillingState"]
    wicked_df["CUSTOMERCOUNTRY"] = df["BillingCountry"]
    wicked_df["SUBSCRIPTIONID"] = df["SBQQ__Subscription__c"]
    wicked_df["ORDERSYSTEM"] = "SFDC-API"
    wicked_df["ISREFUND"] = "NO"
    wicked_df["PRODUCTID"] = df["Product2Id"]
    wicked_df["PRODUCTNAME"] = df["Product_Name__c"]
    wicked_df["CURRENCYCODE"] = df["CurrencyIsoCode"]
    wicked_df["CITY"] = df["BillingCity"]
    wicked_df["IPADDRESS"] = ""

    wicked_df.loc[wicked_df["CUSTOMERCOUNTRY"] == "US", "CUSTOMERCOUNTRY"] = "United States"

    # Remove rows with missing data (except IPADDRESS)
    columns_to_check = [col for col in wicked_columns if col != "IPADDRESS"]
    wicked_df = wicked_df.dropna(subset=columns_to_check)

    return wicked_df

def export_wicked_report_orders(df):
    """
    Export the Salesforce data to a CSV file format compatible with Wicked Reports.

    Args:
        df (pd.DataFrame): The raw Salesforce data fetched for Wicked Reports.

    Returns:
        pd.DataFrame: A DataFrame formatted for Wicked Reports export.
    """
    # Define the columns we want in our export, based on the provided template
    wicked_report_columns = [
        "ORDERID",
        "ORDERDATETIME",
        "ORDERTIMEZONE",
        "ORDERTOTAL",
        "CUSTOMEREMAIL",
        "CUSTOMERSTATE",
        "CUSTOMERCOUNTRY",
        "SUBSCRIPTIONID",
        "ORDERSYSTEM",
        "ISREFUND",
        "PRODUCTID",
        "PRODUCTNAME",
        "CURRENCYCODE",
        "CITY",
        "IPADDRESS",
    ]

    # Create a list to store individual order items
    order_items = []

    for _, row in df.iterrows():
        base_order = {
            "ORDERID": row["Id"],
            "ORDERDATETIME": pd.to_datetime(row["CreatedDate"]).strftime("%m/%d/%Y %H:%M"),
            "ORDERTIMEZONE": (
                row["Account"]["Time_Zone__c"]
                if isinstance(row["Account"], dict) and "Time_Zone__c" in row["Account"]
                else ""
            ),
            "ORDERTOTAL": round(row["TotalAmount"], 2) if pd.notna(row["TotalAmount"]) else "",
            "CUSTOMEREMAIL": (
                row["Account"]["Account_Primary_Contact_Email__c"]
                if isinstance(row["Account"], dict) and "Account_Primary_Contact_Email__c" in row["Account"]
                else ""
            ),
            "CUSTOMERSTATE": row["BillingState"],
            "CUSTOMERCOUNTRY": "United States" if row["BillingCountry"] == "US" else row["BillingCountry"],
            "ORDERSYSTEM": "SFDC-API",
            "ISREFUND": "NO",
            "CURRENCYCODE": row["CurrencyIsoCode"],
            "CITY": row["BillingCity"],
            "IPADDRESS": "",
        }

        # Handle nested OrderItems
        if isinstance(row["OrderItems"], dict) and "records" in row["OrderItems"]:
            for item in row["OrderItems"]["records"]:
                order_item = base_order.copy()
                order_item["SUBSCRIPTIONID"] = item.get("SBQQ__Subscription__c", "")
                order_item["PRODUCTID"] = item.get("Product2Id", "")
                order_item["PRODUCTNAME"] = item.get("Product_Name__c", "") or (item.get("Product2", {}) or {}).get(
                    "Name", ""
                )
                order_items.append(order_item)
        else:
            # If no OrderItems, still include the base order
            order_items.append(base_order)

    # Create DataFrame from order_items
    wicked_report_df = pd.DataFrame(order_items, columns=wicked_report_columns)

    # Fill any empty cells with an empty string
    wicked_report_df = wicked_report_df.fillna("")

    # Ensure only the specified columns are included and in the correct order
    wicked_report_df = wicked_report_df[wicked_report_columns]

    return wicked_report_df

def calculate_product_adoption(dataframe):
    """
    Calculate product adoption rates over time.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing order data.

    Returns:
        pd.DataFrame: A DataFrame with product adoption rates over time.
    """
    dataframe["ORDERDATETIME"] = pd.to_datetime(dataframe["ORDERDATETIME"], errors="coerce")
    dataframe = dataframe.dropna(subset=["ORDERDATETIME"])

    if "PRODUCTNAME" in dataframe.columns:
        dataframe["OrderYear"] = dataframe["ORDERDATETIME"].dt.year
        dataframe["OrderMonth"] = dataframe["ORDERDATETIME"].dt.month

        # Group by product, year, and month, then count orders
        product_orders = dataframe.groupby(["PRODUCTNAME", "OrderYear", "OrderMonth"])["ORDERID"].count().reset_index()

        # Rename the count column to AdoptionRate
        product_adoption = product_orders.rename(columns={"ORDERID": "AdoptionRate"})
        return product_adoption
    else:
        st.error("'ProductName' column is missing from the DataFrame.")
        return pd.DataFrame()


def apply_moving_average(series, window):
    """
    Apply a moving average to a time series.

    Args:
        series (pd.Series): The time series data.
        window (int): The window size for the moving average.

    Returns:
        pd.Series: The time series with the moving average applied.
    """
    return series.rolling(window=window).mean()

def calculate_target_and_mql_count(df, date_option, start_date, end_date):
    """
    Calculate the target MQL count and actual MQL count for the selected period.

    Args:
        df: Pandas DataFrame containing MQL data.
        date_option: Selected date range option.
        start_date: Start date for the data range.
        end_date: End date for the data range.

    Returns:
        A tuple containing the target MQL count, actual MQL count, and period name.
    """
    monthly_target = 1144

    # Ensure start_date and end_date are timezone-aware
    start_date = pd.Timestamp(start_date).tz_localize("UTC")
    end_date = pd.Timestamp(end_date).tz_localize("UTC")

    # Calculate the number of months, quarters, and years in the selected period
    months_in_period = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    quarters_in_period = (months_in_period + 2) // 3  # Round up to the nearest quarter
    years_in_period = (months_in_period + 11) // 12  # Round up to the nearest year

    # Calculate target based on the selected period
    if date_option == "Month to Date (Current Month)":
        target = monthly_target
        period_name = "Month"
    elif date_option == "Last Month":
        target = monthly_target
        period_name = "Month"
    elif date_option == "Last 7 days":
        target = monthly_target * (7 / 30)  # Approximate target for a week
        period_name = "Week"
    elif date_option in ["Last 3 months", "Last 6 months"]:
        target = monthly_target * quarters_in_period
        period_name = "Quarter"
    else:  # "Last 12 months", "Year to Date", "Custom"
        target = monthly_target * years_in_period
        period_name = "Year"

    mql_count = df[(df["CreatedDate"] >= start_date) & (df["CreatedDate"] <= end_date)].shape[0]

    return target, mql_count, period_name


def calculate_velocity(sf, start_date, end_date):
    """
    Calculates the average transition time (velocity) for different stages in the sales process.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.

    Returns:
        A dictionary containing the average transition times for each stage, with 'days' suffix.
    """

    # 1. MQL to SQL: Using MQL_Record__c object and MQL_Converted_Date__c field
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')

    mql_to_sql_query = load_sql_query("MQL_Record__c.sql").format(start_date=start_date_str, end_date=end_date_str)

    mql_to_sql_result = sf.query_all(mql_to_sql_query)
    mql_to_sql_df = pd.DataFrame(mql_to_sql_result["records"]).drop(columns=["attributes"])

    # Convert to datetime and set timezone to UTC
    mql_to_sql_df["CreatedDate"] = pd.to_datetime(mql_to_sql_df["CreatedDate"])
    mql_to_sql_df["MQL_Converted_Date__c"] = pd.to_datetime(mql_to_sql_df["MQL_Converted_Date__c"])

    if mql_to_sql_df["CreatedDate"].dt.tz is None:
        mql_to_sql_df["CreatedDate"] = mql_to_sql_df["CreatedDate"].dt.tz_localize("UTC")
    else:
        mql_to_sql_df["CreatedDate"] = mql_to_sql_df["CreatedDate"].dt.tz_convert("UTC")

    if mql_to_sql_df["MQL_Converted_Date__c"].dt.tz is None:
        mql_to_sql_df["MQL_Converted_Date__c"] = mql_to_sql_df["MQL_Converted_Date__c"].dt.tz_localize("UTC")
    else:
        mql_to_sql_df["MQL_Converted_Date__c"] = mql_to_sql_df["MQL_Converted_Date__c"].dt.tz_convert("UTC")

    mql_to_sql_df["MQLToSQL"] = (mql_to_sql_df["MQL_Converted_Date__c"] - mql_to_sql_df["CreatedDate"]).dt.days
    mql_to_sql = mql_to_sql_df["MQLToSQL"].mean()

    # 2. SQL to Won: Using Opportunity object and CreatedDate and CloseDate fields
    sql_to_won_query = load_sql_query("Opportunity.sql").format(start_date=start_date_str, end_date=end_date_str)

    sql_to_won_result = sf.query_all(sql_to_won_query)
    sql_to_won_df = pd.DataFrame(sql_to_won_result["records"]).drop(columns=["attributes"])

    # Convert to datetime and set timezone to UTC
    sql_to_won_df["CreatedDate"] = pd.to_datetime(sql_to_won_df["CreatedDate"]).dt.tz_convert("UTC")
    sql_to_won_df["CloseDate"] = pd.to_datetime(sql_to_won_df["CloseDate"])

    # Localize CloseDate to UTC if it's tz-naive
    if sql_to_won_df["CloseDate"].dt.tz is None:
        sql_to_won_df["CloseDate"] = sql_to_won_df["CloseDate"].dt.tz_localize("UTC")
    # Convert to UTC if already tz-aware
    else:
        sql_to_won_df["CloseDate"] = sql_to_won_df["CloseDate"].dt.tz_convert("UTC")

    sql_to_won_df["SQLToWon"] = (sql_to_won_df["CloseDate"] - sql_to_won_df["CreatedDate"]).dt.days
    sql_to_won = sql_to_won_df["SQLToWon"].mean()

    # Calculate total velocity
    total_velocity = mql_to_sql + sql_to_won

    return {
        "MQL to SQL": f"{mql_to_sql:.2f} days",
        "SQL to Won": f"{sql_to_won:.2f} days",
        "Total": f"{total_velocity:.2f} days",
    }


def create_velocity_funnel(velocity_data):
    """
    Creates a funnel chart visualizing velocity data.

    Args:
        velocity_data (dict): A dictionary where keys are sales stages and values are average transition times.

    Returns:
        plotly.graph_objects.Figure: The funnel chart figure.
    """
    stages = list(velocity_data.keys())
    values = [float(value.split()[0]) for value in velocity_data.values()]  # Extract numeric value for plotting
    labels = list(velocity_data.values())  # Use the full text including "days" suffix

    fig = go.Figure(
        go.Funnel(
            y=stages,
            x=values,
            textinfo="text",  # Show the text (e.g., "X days")
            text=labels,  # Use the labels with "days" suffix
            textfont={"size": 14},
            marker={"color": ["#0077CC", "#009933", "#66CCFF"]},  # Customize colors
            connector={"line": {"color": "gray", "width": 2}},
        )
    )

    fig.update_layout(title="How many days does it take to transition?")
    return fig


# MQL Card Functions
def display_mql_card(df, field, chart_function, metric_label="Count"):
    """
    Displays a card with headline stats and a visualization.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.
        field (str): The column name to analyze.
        chart_function (function): The function to create the visualization.
        metric_label (str): The label for the metric being displayed.
    """
    data = df[field].value_counts().reset_index()
    data.columns = [field, metric_label]

    total = len(data)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric(f"Average {metric_label}", data[metric_label].mean())
    st.plotly_chart(chart_function(df), use_container_width=True)

def MQLFunnelCard(df):
    """
    Creates a funnel chart visualizing the MQL conversion stages.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The funnel chart figure.
    """
    stages = ["MQL Created", "Contacted", "Qualified", "Demo Scheduled", "Opportunity Created", "Won"]
    
    # Calculate counts for each stage
    stage_counts = []
    stage_counts.append(len(df))  # Total MQLs Created
    stage_counts.append(len(df[df["Status__c"] != "New"])) # Assuming "New" means not contacted yet 
    stage_counts.append(len(df[df["Status__c"] == "Qualified"]))
    stage_counts.append(len(df[df["Status__c"] == "Demo Scheduled"])) 
    stage_counts.append(len(df[df["Opportunity__c"].notna()]))
    stage_counts.append(len(df[df["Status__c"] == "Won"]))

    fig = go.Figure(go.Funnel(
        y=stages,
        x=stage_counts,
        textinfo="value+percent initial", # Show count and percentage 
        marker={"color": ["#0077CC", "#66CCFF", "#009933", "#99FF99", "#FFCC99", "#FF9933"]}
    ))
    fig.update_layout(title="MQL Conversion Funnel")
    return fig

def create_top_campaigns_chart(df, campaign_df):
    """
    Creates a bar chart showing the top campaigns driving MQLs.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.
        campaign_df (pd.DataFrame): The DataFrame containing Campaign data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    # Merge MQL data with Campaign data
    merged_df = df.merge(campaign_df, left_on="Campaign__c", right_on="Id", how="left")

    campaign_counts = merged_df["Name_y"].value_counts().reset_index()
    campaign_counts.columns = ["Campaign", "MQL Count"]
    fig = px.bar(campaign_counts, x="Campaign", y="MQL Count", title="Top Campaigns Driving MQLs")
    return fig


def LeadSourcePerformanceCard(df):
    """
    Visualize lead source performance with MQL count and conversion rate.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    lead_source_data = df.groupby("Lead_Source__c")['Converted_to_Opportunity__c'].agg(['sum', 'count']).reset_index()
    lead_source_data.columns = ["Lead Source", "Conversions", "Count"]
    lead_source_data['Conversion Rate'] = lead_source_data['Conversions'] / lead_source_data['Count']

    fig = px.bar(lead_source_data, x="Lead Source", y=["Count", "Conversions"], 
                 title="Lead Source Performance - MQL Count and Conversions",
                 barmode='group')
    fig.update_yaxes(title_text="Count")

    # Add conversion rate as text annotations on each bar
    for i, row in lead_source_data.iterrows():
        fig.add_annotation(
            x=row["Lead Source"], 
            y=row["Count"],
            text=f"{row['Conversion Rate']:.2%}",
            showarrow=False,
            yshift=10 
        )
    return fig

def ConversionRateProductInterestCard(df):
    """
    Visualize conversion rates by product interest.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The pie chart figure.
    """
    product_interest_data = df.groupby("Product_Interest__c")["Converted_to_Opportunity__c"].mean().reset_index()
    product_interest_data.columns = ["Product Interest", "Conversion Rate"]
    return px.pie(
        product_interest_data,
        values="Conversion Rate",
        names="Product Interest",
        title="Conversion Rate by Product Interest",
    )


def MQLTrendOverTimeCard(df):
    """
    Visualize MQL trend over time, optionally grouped by Lead Source.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The line chart figure.
    """
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    mql_trend = df.groupby([df["CreatedDate"].dt.date, "Lead_Source__c"]).size().reset_index(name="Count")

    # Create line chart with Plotly Express
    fig = px.line(mql_trend, x="CreatedDate", y="Count", color="Lead_Source__c", 
                  title="MQL Trend Over Time by Lead Source") 
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="MQL Count")
    return fig


def MQLSourceConversionRateCard(df):
    """
    Visualize MQL conversion rate by lead source.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    conversion_rates = df.groupby("Lead_Source__c")["Converted_to_Opportunity__c"].mean().reset_index()
    conversion_rates.columns = ["Lead Source", "Conversion Rate"]
    conversion_rates["Average Conversion Rate"] = conversion_rates["Conversion Rate"].mean()
    return px.bar(
        conversion_rates,
        x="Lead Source",
        y="Conversion Rate",
        title="MQL Source Conversion Rate",
        text="Average Conversion Rate",
    )


def TimeToSQLCard(df):
    """
    Visualize time to SQL (in days), handling potential issues with small or empty datasets. 

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The histogram figure.
    """

    df['TimeToSQL'] = (df["MQL_Converted_Date__c"] - df["CreatedDate"]).dt.days

    # Ensure there are valid TimeToSQL values for the histogram
    df = df[df['TimeToSQL'] >= 0] 

    if df.empty:
        # Handle the case where there's no data to plot
        fig = go.Figure()
        fig.update_layout(
            xaxis =  { "visible": False },
            yaxis = { "visible": False },
            annotations = [
                {
                    "text": "No data available to calculate Time to SQL",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16}
                }
            ]
        )
        return fig

    median_value = df["TimeToSQL"].median()
    mean_value = df["TimeToSQL"].mean()

    fig = px.histogram(df, x="TimeToSQL", title="Time to SQL (in days)")
    fig.update_xaxes(title_text="Time to SQL (days)")
    fig.update_yaxes(title_text="Count")

    fig.add_vline(
        x=median_value,
        line_dash="dash",
        line_color="green",
        annotation_text="Median",
        annotation_position="top left",
    )
    fig.add_vline(
        x=mean_value, line_dash="dash", line_color="red", annotation_text="Average", annotation_position="top right"
    )
    return fig


def MQLDistributionByCampaignCard(df):
    """
    Visualize MQL distribution by campaign.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    campaign_distribution = df["Campaign__c"].value_counts().reset_index()
    campaign_distribution.columns = ["Campaign", "Count"]
    return px.bar(campaign_distribution, x="Campaign", y="Count", title="MQL Distribution by Campaign")

# SQL Workbench with AI
def load_json_data(filepath):
    """
    Loads JSON data from the provided file path, handles nested structures if present.
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        # Normalize the JSON structure if there are nested fields
        df = pd.json_normalize(data)
        return df
    except Exception as e:
        st.error(f"Error loading JSON data: {str(e)}")
        return None
    
def validate_salesforce_object_and_fields(sf, object_name: str, fields: list):
    """
    Validate whether a Salesforce object and its fields exist in the Salesforce schema.

    Args:
        sf: Salesforce connection object.
        object_name (str): The Salesforce object name to validate.
        fields (list): A list of fields to validate within the object.

    Returns:
        dict: A dictionary containing validation results with potential corrections.
    """
    try:
        # Fetch the object's metadata
        object_metadata = sf.describe()

        # Check if the object exists
        if object_name not in [obj['name'] for obj in object_metadata['sobjects']]:
            return {"status": "error", "message": f"Object '{object_name}' does not exist.", "suggestions": []}
        
        # Validate the fields within the object
        field_metadata = sf.__getattr__(object_name).describe()
        existing_fields = [f['name'] for f in field_metadata['fields']]
        
        invalid_fields = [field for field in fields if field not in existing_fields]
        
        if invalid_fields:
            return {
                "status": "error", 
                "message": f"Fields {invalid_fields} not found in object '{object_name}'.", 
                "suggestions": existing_fields[:5]  # Suggest first 5 fields
            }
        
        return {"status": "success", "message": "Object and fields are valid."}
    
    except Exception as e:
        return {"status": "error", "message": str(e), "suggestions": []}

def generate_next_soql_with_openai(openai_model, salesforce_data, failed_query):
    """
    Generate the next SOQL query using OpenAI by considering the entire object tree context.

    Args:
        openai_model: OpenAI model object to generate the SOQL query.
        salesforce_data: JSON data with available Salesforce objects and fields.
        failed_query: The query that returned empty results.

    Returns:
        str: The new SOQL query generated by OpenAI, with unnecessary spacing removed.
    """
    # Extract available objects and fields from the JSON data
    object_list = salesforce_data["object_list"]
    object_tree = salesforce_data["object_tree"]
    
    # Refine the prompt for OpenAI to generate the next query, considering all relevant fields
    prompt = f"""
    You are an expert in Salesforce Object Query Language (SOQL).
    The following objects are available: {object_list}.
    Here is the complete field list for the 'Opportunity' object: {object_tree['Opportunity']}.
    
    The following SOQL query returned no results:
    {failed_query}
    
    Using the above field list, generate a new SOQL query that can return results.
    Use appropriate conditions and filters for the 'Opportunity' object. Only return the SOQL query.
    """
    
    # Generate the new query using OpenAI
    response = openai_model(prompt)
    
    if response:
        # Clean up response: remove extra spaces and backticks, and ensure proper formatting
        new_query = response[0].strip().replace("  ", " ").replace("`", "")
        return new_query
    else:
        return None

def run_soql_query_until_not_empty(sf, openai_model, query, salesforce_data, max_retries=5, retry_delay=5):
    """
    Runs a SOQL query and retries until a non-empty result is returned or a result is confirmed empty.

    Args:
        sf: Salesforce connection object.
        openai_model: OpenAI model to generate new queries dynamically if needed.
        query: The initial SOQL query string.
        salesforce_data: Salesforce object schema for generating the next query if needed.
        max_retries: Maximum number of retries if a temporary issue occurs (default: 5).
        retry_delay: Time (in seconds) to wait between retries (default: 5 seconds).

    Returns:
        A Pandas DataFrame containing the query result or an error message after retries.
    """
    for retry_count in range(max_retries):
        try:
            # Run the SOQL query
            result = sf.query_all(query)
            result_df = pd.DataFrame(result["records"]).drop(columns=["attributes"], errors="ignore")
            
            # If results are found, return immediately
            if not result_df.empty:
                return result_df

            st.warning(f"Query returned empty result. Retrying... ({retry_count+1}/{max_retries})")

        except Exception as e:
            st.error(f"Error executing query: {e}")
            return None

        # If empty result, try generating a new query dynamically using OpenAI
        st.info("Retrying with a new query...")
        query = generate_next_soql_with_openai(openai_model, salesforce_data, query)

        # Wait for the retry delay before running again
        time.sleep(retry_delay)

    st.warning("No data returned after maximum retries and refinement.")
    return None

def generate_soql_query_with_validation(sf, salesforce_data: Dict, object_aliases: Dict, requirement: str) -> str:
    """
    Generates a SOQL query using OpenAI based on the Salesforce data and user-provided requirement, with validation.
    
    Args:
        sf: Salesforce connection object.
        salesforce_data (Dict): The Salesforce object and field data.
        object_aliases (Dict): Mapping of common terms to Salesforce objects.
        requirement (str): The user's requirement for the query.
        
    Returns:
        str: The generated SOQL query, or an error message if validation fails.
    """

    # Load the object list and tree from the Salesforce data schema
    object_list = salesforce_data["object_list"]
    object_tree = salesforce_data["object_tree"]

    # Step 1: Dynamically determine object based on user input or mapped terms
    possible_objects = [obj for obj in object_list if obj.lower() in requirement.lower()]

    # Step 2: If no exact match, try to use object alias mapping from the JSON file
    if not possible_objects:
        for mapping in object_aliases["object_aliases"]:
            if mapping["term"].lower() in requirement.lower():
                possible_objects.append(mapping["salesforce_object"])

    # Step 3: If still no match, apply fuzzy matching to find the closest Salesforce object
    if not possible_objects:
        closest_match, score = process.extractOne(requirement, object_list, scorer=fuzz.token_sort_ratio)
        
        # Set a similarity threshold (e.g., 70%)
        if score >= 70:
            possible_objects.append(closest_match)
        else:
            # Suggest similar objects if no close match is found
            return f"Error: No matching object found for the requirement. Did you mean '{closest_match}'?"

    if not possible_objects:
        return "Error: No matching object found for the requirement."

    # Step 4: Select the first matched object and retrieve fields
    object_name = possible_objects[0]
    object_fields = object_tree.get(object_name, [])
    
    if not object_fields:
        return f"Error: No fields found for object '{object_name}' in Salesforce data."

    # Step 5: Determine relevant fields mentioned in the requirement, or default to the first few fields
    relevant_fields = [field for field in object_fields if field.lower() in requirement.lower()]

    if not relevant_fields:
        relevant_fields = object_fields[:3]  # Default to the first 3 fields if none are mentioned

    # Step 6: Validate the object and fields
    validation_result = validate_salesforce_object_and_fields(sf, object_name, relevant_fields)
    if validation_result["status"] == "error":
        return f"Validation Error: {validation_result['message']}. Suggestions: {validation_result['suggestions']}"

    # Step 7: Construct the refined OpenAI prompt to generate a valid SOQL query
    prompt = f"""
    You are an expert at writing SOQL (Salesforce Object Query Language) queries for Salesforce.

    The available objects are: {', '.join(object_list)}

    The '{object_name}' object has the following example fields: {', '.join(object_fields)}

    Generate **only** a valid SOQL query that meets the following requirement: "{requirement}".

    Important considerations:
    1. Avoid unsupported SOQL functions like YEAR(), GROUP BY, or HAVING.
    2. Apply date filters like `CreatedDate >= 2024-01-01T00:00:00Z`.
    3. Do not include any explanations or keywords like 'sql'. Return only the query.
    4. Ensure the SOQL query only includes fields and objects from the validated Salesforce schema.
    """

    # Step 8: Call OpenAI to generate the SOQL query
    response = openai_model(prompt)

    # Ensure the response contains only SOQL code
    if response and isinstance(response, list) and response[0]:
        soql_query = response[0].strip()
    else:
        return "Error: No response from OpenAI."

    # Step 9: Clean up any markdown formatting or extraneous text from OpenAI output
    if "```" in soql_query:
        soql_query = soql_query.split("```")[1].strip()  # Extract only the query

    soql_query = soql_query.replace("sql", "").strip()  # Ensure no unexpected words appear

    return soql_query

def parse_relative_time(requirement: str):
    """
    Parse the user's request and determine the appropriate time range based on relative time phrases.
    
    Args:
        requirement (str): The user's requirement/query string.
        
    Returns:
        tuple: start_date (datetime), end_date (datetime), or None if no valid time period is found.
    """
    today = datetime.utcnow()
    
    if "last week" in requirement.lower():
        start_of_week = today - timedelta(days=today.weekday(), weeks=1)  # Previous Monday
        end_of_week = start_of_week + timedelta(days=6)  # End of previous Sunday
        return start_of_week, end_of_week
    
    elif "this month" in requirement.lower():
        start_of_month = today.replace(day=1)  # First day of the current month
        end_of_month = today  # Today is the end of the current period (to now)
        return start_of_month, end_of_month
    
    elif "last month" in requirement.lower():
        first_day_this_month = today.replace(day=1)
        last_month_end = first_day_this_month - timedelta(days=1)  # Last day of the previous month
        last_month_start = last_month_end.replace(day=1)  # First day of the previous month
        return last_month_start, last_month_end
    
    elif "this year" in requirement.lower():
        start_of_year = today.replace(month=1, day=1)  # First day of the current year
        end_of_year = today  # Today is the end of the current period (to now)
        return start_of_year, end_of_year
    
    elif "last year" in requirement.lower():
        start_of_last_year = today.replace(year=today.year - 1, month=1, day=1)  # First day of last year
        end_of_last_year = start_of_last_year.replace(month=12, day=31)  # Last day of last year
        return start_of_last_year, end_of_last_year

    return None, None

def identify_salesforce_objects(query: str, salesforce_data: Dict, object_aliases: Dict) -> list:
    """
    Identify Salesforce objects mentioned in a user's query.

    Args:
        query (str): The user's query in natural language.
        salesforce_data (Dict): Salesforce object and field data.
        object_aliases (Dict): Mapping of common terms to Salesforce objects.

    Returns:
        list: A list of possible Salesforce objects mentioned in the query.
    """
    possible_objects = []
    for alias in object_aliases['object_aliases']:
        if alias['term'].lower() in query.lower():
            possible_objects.append(alias['salesforce_object'])
    return possible_objects

def infer_relevant_fields(query: str, available_fields: list) -> list:
    """
    Infer relevant fields based on keywords in a user's query.

    Args:
        query (str): The user's query.
        available_fields (list): List of available fields for the object.

    Returns:
        list: List of inferred relevant fields.
    """
    if "name" in query.lower():
        return [field for field in available_fields if "Name" in field]
    if "amount" in query.lower() or "revenue" in query.lower():
        return [field for field in available_fields if "Amount" in field or "Revenue" in field]
    return available_fields[:5]  # Default to the first 5 fields

def generate_soql_query_with_dynamic_dates(user_query: str, salesforce_data: Dict, object_aliases: Dict) -> str:
    """
    Generate a SOQL query based on the user's natural language query, 
    dynamically handling date-based and non-date-based questions.

    Args:
        user_query (str): The user's query in natural language.
        salesforce_data (Dict): Salesforce object and field data.
        object_aliases (Dict): Mapping for object aliases in the Salesforce schema.

    Returns:
        str: The generated SOQL query or an error message if validation fails.
    """
    # 1. Identify the main object(s) from the user query 
    matched_objects = identify_salesforce_objects(user_query, salesforce_data, object_aliases)
    
    if not matched_objects:
        return "Error: No matching Salesforce object found for your query."
    
    # 2. Analyze the user query for context
    date_range = parse_relative_time(user_query)  
    start_date, end_date = date_range
    
    # 3. Select appropriate fields based on the matched object
    selected_object = matched_objects[0]  
    available_fields = salesforce_data['object_tree'].get(selected_object, [])
    
    if not available_fields:
        return f"Error: No fields available for the object '{selected_object}'."
    
    # Dynamically select the fields 
    relevant_fields = infer_relevant_fields(user_query, available_fields)
    field_str = ', '.join(relevant_fields)
    
    # 4. Generate the SOQL query dynamically
    soql_query = f"SELECT {field_str} FROM {selected_object}"
    
    # 5. Add time-based filters if dates are inferred
    if start_date and end_date:
        start_date_str = start_date.strftime('%Y-%m-%dT00:00:00Z')
        end_date_str = end_date.strftime('%Y-%m-%dT23:59:59Z')
        soql_query += f" WHERE CreatedDate >= {start_date_str} AND CreatedDate <= {end_date_str}"
    
    return soql_query.strip().replace("  ", " ") 

def validate_required_columns(sf, object_name: str, required_fields: list):
    """
    Validate whether the required columns exist in the Salesforce object schema.
    
    Args:
        sf: Salesforce connection object.
        object_name (str): The Salesforce object name.
        required_fields (list): A list of required fields for validation.

    Returns:
        tuple: A tuple containing the valid fields and the missing fields.
    """
    try:
        # Describe the object to get its metadata
        object_metadata = sf.__getattr__(object_name).describe()
        existing_fields = [f['name'] for f in object_metadata['fields']]

        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in existing_fields]
        valid_fields = [field for field in required_fields if field in existing_fields]

        return valid_fields, missing_fields
    except Exception as e:
        logging.error(f"Error validating columns for {object_name}: {str(e)}")
        return [], required_fields

def generate_soql_query(object_name: str, required_fields: list, sf):
    """
    Generate a SOQL query based on available columns for a given object.

    Args:
        object_name (str): The name of the Salesforce object.
        required_fields (list): The list of fields required for the query.
        sf: Salesforce connection object.

    Returns:
        str: The generated SOQL query or an error message.
    """
    # Validate the fields
    valid_fields, missing_fields = validate_required_columns(sf, object_name, required_fields)

    if missing_fields:
        st.warning(f"Required columns missing: {missing_fields}")

    # If no valid fields are found, return an error
    if not valid_fields:
        return f"Error: No valid fields found for object '{object_name}'."

    # Generate the SOQL query string based on available fields
    fields_str = ', '.join(valid_fields)
    query = f"SELECT {fields_str} FROM {object_name}"

    return query

def fetch_salesforce_data_with_validation(sf, object_name, required_fields):
    """
    Fetch Salesforce data, validating the existence of required columns.

    Args:
        sf: Salesforce connection object.
        object_name (str): The Salesforce object to query.
        required_fields (list): A list of required columns for the query.

    Returns:
        pd.DataFrame: The fetched data as a Pandas DataFrame or an error message.
    """
    # Generate the SOQL query with dynamic field checking
    soql_query = generate_soql_query(object_name, required_fields, sf)
    
    if soql_query.startswith("Error"):
        return None, soql_query

    # Fetch data from Salesforce
    try:
        result = sf.query_all(soql_query)
        data_df = pd.DataFrame(result["records"]).drop(columns=["attributes"], errors="ignore")
        return data_df, None
    except Exception as e:
        error_message = f"Error fetching data: {str(e)}"
        logging.error(error_message)
        return None, error_message

# Opportunities 
def fetch_open_opportunities_data(sf, start_date, end_date, brand=None):
    """
    Fetch open opportunities data from Salesforce within a specified date range,
    optionally filtered by brand.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.
        brand: The brand to filter by (optional).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the fetched open opportunities data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    query = load_sql_query("Opportunity.sql").format(start_date=start_date_str, end_date=end_date_str)

    if brand:
        query += f" AND Brand__c = '{brand}'"

    result = sf.query_all(query)
    df = pd.DataFrame(result["records"]).drop(columns=["attributes"])
    return df

def process_open_opportunities_data(df):
    """
    Process the open opportunities data.

    Args:
        df (pd.DataFrame): The raw open opportunities data.

    Returns:
        pd.DataFrame: Processed open opportunities data.
    """
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"]).dt.tz_localize(None)
    df["CloseDate"] = pd.to_datetime(df["CloseDate"]).dt.tz_localize(None)
    df["DaysOpen"] = (datetime.now(timezone.utc).replace(tzinfo=None) - df["CreatedDate"]).dt.days
    return df

def OpenOpportunitiesByStageCard(df):
    """
    Visualize open opportunities by stage.

    Args:
        df (pd.DataFrame): The processed open opportunities data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    stage_data = df["StageName"].value_counts().reset_index()
    stage_data.columns = ["Stage", "Count"]
    return px.bar(stage_data, x="Stage", y="Count", title="Open Opportunities by Stage")

def OpenOpportunitiesTrendCard(df):
    """
    Visualize open opportunities trend over time.

    Args:
        df (pd.DataFrame): The processed open opportunities data.

    Returns:
        plotly.graph_objects.Figure: The line chart figure.
    """
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    trend_data = df.groupby(df["CreatedDate"].dt.date).size().reset_index(name="Count")
    return px.line(trend_data, x="CreatedDate", y="Count", title="Open Opportunities Trend")

# Leads
def fetch_lead_data(sf, start_date, end_date, brand=None):
    """
    Fetches lead data from Salesforce within a specified date range,
    optionally filtered by brand.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.
        brand: The brand to filter by (optional).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the fetched lead data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
    query = load_sql_query("Lead.sql").format(start_date=start_date_str, end_date=end_date_str)

    if brand:
        query += f" AND Brand__c = '{brand}'"

    result = sf.query_all(query)
    df = pd.DataFrame(result["records"]).drop(columns=["attributes"])
    return df

def process_lead_data(df):
    """
    Processes lead data, including handling timezones and calculating lead age.

    Args:
        df (pd.DataFrame): The raw lead data.

    Returns:
        pd.DataFrame: Processed lead data.
    """
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    df["ConvertedDate"] = pd.to_datetime(df["ConvertedDate"])
    df["LeadAge"] = (datetime.now(timezone.utc) - df["CreatedDate"]).dt.days
    return df

def LeadsBySourceCard(df):
    """
    Visualizes leads by source using a bar chart.

    Args:
        df (pd.DataFrame): The processed lead data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    lead_source_counts = df["LeadSource"].value_counts().reset_index()
    lead_source_counts.columns = ["Lead Source", "Count"]
    return px.bar(lead_source_counts, x="Lead Source", y="Count", title="Leads by Source")

def LeadConversionOverTimeCard(df):
    """
    Visualizes lead conversion trend over time with a line chart.

    Args:
        df (pd.DataFrame): The processed lead data.

    Returns:
        plotly.graph_objects.Figure: The line chart figure.
    """
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    converted_leads = df[df["IsConverted"] == True]
    conversion_trend = converted_leads.groupby(converted_leads["CreatedDate"].dt.date).size().reset_index(name="Conversions")
    return px.line(conversion_trend, x="CreatedDate", y="Conversions", title="Lead Conversion Trend Over Time")

def LeadAgeDistributionCard(df):
    """
    Visualizes lead age distribution using a histogram.

    Args:
        df (pd.DataFrame): The processed lead data.

    Returns:
        plotly.graph_objects.Figure: The histogram figure.
    """
    return px.histogram(df, x="LeadAge", title="Lead Age Distribution")

def LeadConversionFunnelCard(df):
    """
    Visualizes the lead conversion funnel with a funnel chart.

    Args:
        df (pd.DataFrame): The processed lead data.

    Returns:
        plotly.graph_objects.Figure: The funnel chart figure.
    """
    stages = ["Lead Created", "Contacted", "Qualified", "Demo Scheduled", "Opportunity Created", "Won"]

    # You need to adjust these conditions based on your Salesforce data
    stage_counts = [
        len(df),  # Total Leads Created
        len(df[df["Status"] != "New"]),  # Assuming "New" means not contacted
        len(df[df["Status"] == "Qualified"]),
        len(df[df["Status"] == "Demo Scheduled"]),  # Customize based on your process
        len(df[df["IsConverted"] == True]),  # Converted to Opportunity
        len(df[df["Status"] == "Won"])  # Opportunities Won (you might need to join with Opportunity data)
    ]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=stage_counts,
        textinfo="value+percent initial",  # Show count and percentage
        marker={"color": ["#0077CC", "#66CCFF", "#009933", "#99FF99", "#FFCC99", "#FF9933"]}
    ))
    fig.update_layout(title="Lead Conversion Funnel")
    return fig

def calculate_cltv(df, time_window_months=12):
    """
    Calculate Customer Lifetime Value (CLTV) for each customer over a specified time window.

    Args:
        df (pd.DataFrame): The DataFrame containing order data.
        time_window_months (int, optional): The time window in months for CLTV calculation. 
                                           Defaults to 12 months.

    Returns:
        pd.DataFrame: The DataFrame with added 'CLTV' and 'AVG_ORDER_VALUE' columns, 
                      or None if no CLTV data can be calculated.
    """
    df['ORDERDATETIME'] = pd.to_datetime(df['ORDERDATETIME'])

    if df.empty or 'CUSTOMEREMAIL' not in df.columns or 'ORDERTOTAL' not in df.columns:
        st.warning("Not enough data to calculate CLTV. Please ensure you have orders with customer emails and order totals.")
        return None

    # Calculate average order value
    customer_avg_order_value = df.groupby("CUSTOMEREMAIL")["ORDERTOTAL"].mean()

    # Calculate the time difference between the first and last order for each customer
    customer_time_diff = df.groupby("CUSTOMEREMAIL")['ORDERDATETIME'].apply(
        lambda x: (x.max() - x.min()).days
    )

    # Calculate number of unique orders per customer
    customer_unique_orders = df.groupby("CUSTOMEREMAIL")['ORDERID'].nunique()

    # Create DataFrame for CLTV
    cltv_df = pd.DataFrame({
        "CUSTOMEREMAIL": customer_avg_order_value.index, 
        "AVG_ORDER_VALUE": customer_avg_order_value.values,
        "TIME_DIFF": customer_time_diff.values,
        "UNIQUE_ORDERS": customer_unique_orders.values
    }).reset_index(drop=True)

    # Calculate purchase frequency, handling zero time difference
    cltv_df['PURCHASE_FREQUENCY'] = cltv_df.apply(
        lambda row: row['UNIQUE_ORDERS'] / (row['TIME_DIFF'] / 30) if row['TIME_DIFF'] > 0 else row['UNIQUE_ORDERS'], 
        axis=1
    )

    # Calculate CLTV 
    cltv_df["CLTV"] = cltv_df["AVG_ORDER_VALUE"] * cltv_df["PURCHASE_FREQUENCY"] * time_window_months

    return df.merge(cltv_df, on="CUSTOMEREMAIL", how="left")

def create_cltv_histogram(df):
    """
    Creates a histogram to visualize the distribution of CLTV values.

    Args:
        df (pd.DataFrame): The DataFrame containing CLTV data.

    Returns:
        plotly.graph_objects.Figure: The histogram figure.
    """
    fig = px.histogram(df, x="CLTV", title="Customer Lifetime Value (CLTV) Distribution")
    fig.update_layout(
        xaxis_title="CLTV",
        yaxis_title="Number of Customers"
    )
    return fig

def create_cltv_by_product_category_chart(df):
    """
    Creates a bar chart to show average CLTV by product category.

    Args:
        df (pd.DataFrame): The DataFrame containing CLTV and product category data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    cltv_by_category = df.groupby('Product_Category__c')['CLTV'].mean().reset_index()
    fig = px.bar(cltv_by_category, x='Product_Category__c', y='CLTV', 
                 title='Average CLTV by Product Category')
    fig.update_layout(
        xaxis_title="Product Category",
        yaxis_title="Average CLTV"
    )
    return fig

def fetch_and_process_bookings_data(sf, start_date, end_date, brand=None):
    """
    Fetch and process bookings data from Salesforce within a specified date range,
    optionally filtered by brand.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.
        brand: The brand to filter by (optional).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the processed bookings data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    # Fetch Order data
    order_query = load_sql_query("Order.sql").format(start_date=start_date_str, end_date=end_date_str)
    if brand:
        if isinstance(brand, list):
            brand_str = ", ".join([f"'{b}'" for b in brand])
            order_query += f" AND Brand__c IN ({brand_str})"
        else:
            order_query += f" AND Brand__c = '{brand}'"
    order_result = sf.query_all(order_query)
    orders_df = pd.DataFrame(order_result["records"]).drop(columns=["attributes"])

    # Fetch OrderItem data
    orderitem_query = f"SELECT Id, OrderId, Product2Id, TotalPrice FROM OrderItem WHERE OrderId IN (SELECT Id FROM Order WHERE CreatedDate >= {start_date_str} AND CreatedDate <= {end_date_str})"
    orderitem_result = sf.query_all(orderitem_query)
    orderitems_df = pd.DataFrame(orderitem_result["records"]).drop(columns=["attributes"])

    # Fetch Product data
    product_query = "SELECT Id, Product_Category__c FROM Product2"
    product_result = sf.query_all(product_query)
    products_df = pd.DataFrame(product_result["records"]).drop(columns=["attributes"])

    # Merge the DataFrames
    merged_df = orderitems_df.merge(products_df, left_on='Product2Id', right_on='Id', how='left')
    merged_df = merged_df.merge(orders_df, left_on='OrderId', right_on='Id', how='left')

    # Process the data
    merged_df['CreatedDate'] = pd.to_datetime(merged_df['CreatedDate'])
    merged_df['OrderMonth'] = merged_df['CreatedDate'].dt.strftime('%Y-%m')

    # Group the orders by month and product category and sum the total price
    bookings_by_month = merged_df.groupby(['OrderMonth', 'Product_Category__c'])['TotalPrice'].sum().reset_index()

    # Create a pivot table to show bookings by month and product category
    bookings_pivot = bookings_by_month.pivot(index='OrderMonth', columns='Product_Category__c', values='TotalPrice').fillna(0)

    # Rename the columns for clarity
    bookings_pivot = bookings_pivot.rename(columns={'Services': 'Services Bookings', 'Software': 'Software Bookings'})

    return bookings_pivot

def generate_color_palette(n):
    """
    Generate a palette of n distinct colors.
    
    Args:
        n (int): Number of colors to generate.
    
    Returns:
        list: List of hex color codes.
    """
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.9) for x in range(n)]
    hex_colors = ['#%02x%02x%02x' % tuple(int(x*255) for x in colorsys.hsv_to_rgb(*hsv)) for hsv in HSV_tuples]
    return hex_colors

def create_bookings_chart(df):
    """
    Create a line chart to visualize bookings over time.

    Args:
        df (pd.DataFrame): The processed bookings data.

    Returns:
        plotly.graph_objects.Figure: The line chart figure.
    """
    df_melted = df.reset_index().melt(id_vars=['OrderMonth'], var_name='Category', value_name='Bookings')
    
    # Generate colors based on unique categories
    categories = df_melted['Category'].unique()
    colors = generate_color_palette(len(categories))
    color_map = dict(zip(categories, colors))
    
    fig = px.line(df_melted, x='OrderMonth', y='Bookings', color='Category',
                  title='Bookings by Month and Category',
                  color_discrete_map=color_map)
    fig.update_layout(xaxis_title='Month', yaxis_title='Bookings ($)')
    return fig

def create_bookings_stacked_bar_chart(df):
    """
    Create a stacked bar chart to visualize bookings over time by product family.

    Args:
        df (pd.DataFrame): The processed bookings data.

    Returns:
        plotly.graph_objects.Figure: The stacked bar chart figure.
    """
    # Reset index to make 'OrderMonth' a column
    df_reset = df.reset_index()
    
    # Identify the column names dynamically
    date_col = df_reset.columns[0]  # Assuming the first column is the date/month
    product_cols = df_reset.columns[1:]  # All other columns are product categories
    
    # Generate colors based on product categories
    colors = generate_color_palette(len(product_cols))
    
    # Create the stacked bar chart
    fig = go.Figure()
    
    for i, col in enumerate(product_cols):
        fig.add_trace(go.Bar(
            name=col,
            x=df_reset[date_col],
            y=df_reset[col],
            marker_color=colors[i]
        ))
    
    # Customize the layout
    fig.update_layout(
        title='Bookings by Month and Group',
        xaxis_title='Month',
        yaxis_title='Sum of Net Price',
        barmode='stack',
        legend_title='Product Family (groups)',
        xaxis_tickangle=-45,
        yaxis=dict(
            tickformat='$,.0f',
            tickmode='auto',
            nticks=6,
        ),
        plot_bgcolor='white',
    )
    
    # Add total labels on top of each stacked bar
    for i in range(len(df_reset)):
        total = df_reset[product_cols].iloc[i].sum()
        fig.add_annotation(
            x=df_reset[date_col].iloc[i],
            y=total,
            text=f"${total/1000:.0f}K",
            showarrow=False,
            yshift=10,
        )
    
    return fig

def create_bookings_pie_chart(df):
    """
    Create a pie chart to visualize the sum of net price by product category.

    Args:
        df (pd.DataFrame): The processed bookings data.

    Returns:
        plotly.graph_objects.Figure: The pie chart figure.
    """
    # Sum the total bookings for each product category
    category_totals = df.sum().sort_values(ascending=False)
    
    # Calculate percentages
    total = category_totals.sum()
    category_percentages = (category_totals / total * 100).round(2)
    
    # Generate colors based on the number of categories
    colors = generate_color_palette(len(category_totals))

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=category_totals.index,
        values=category_totals,
        textposition='outside',
        textinfo='none',
        hoverinfo='label+percent+value',
        marker=dict(colors=colors, line=dict(color='#ffffff', width=1)),
        showlegend=False
    )])

    # Update the layout
    fig.update_layout(
        title='Net Bookings by Product Category',
        annotations=[dict(text=f'Total<br>${total/1000:,.0f}K', x=0.5, y=0.5, font_size=20, showarrow=False, font_color='#333333')],
        legend=dict(
            title="Product Family (groups)",
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1
        )
    )

    # Add a custom legend
    for i, (category, value, percentage) in enumerate(zip(category_totals.index, category_totals, category_percentages)):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=colors[i]),
            showlegend=True,
            name=f"{category}<br>${value/1000:,.0f}K ({percentage:.2f}%)"
        ))

    # Remove axes
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    return fig

def create_bookings_count_pie_chart(df):
    """
    Create a pie chart to visualize the count of IDs by product category,
    showing the dollar amount for each slice and abbreviating the total.

    Args:
        df (pd.DataFrame): The processed bookings data (can be multi-index).

    Returns:
        plotly.graph_objects.Figure: The pie chart figure.
    """
    # For multi-index DataFrames, the categories are the column names
    category_amounts = df.sum()
    
    # Calculate the total dollar amount
    total_amount = category_amounts.sum()

    # Abbreviate total amount (MM for millions)
    if total_amount >= 1_000_000:
        total_amount_str = f"${total_amount / 1_000_000:.1f}MM"
    else:
        total_amount_str = f"${total_amount:,.0f}"

    # Generate colors based on the number of categories
    colors = generate_color_palette(len(category_amounts))

    # Create labels with dollar amount
    labels = [f"{cat}\n${amount:,.0f}" 
              for cat, amount in zip(category_amounts.index, category_amounts)]

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=category_amounts.index,
        values=category_amounts,
        text=labels,
        textposition='inside', 
        textinfo='text',  # Show label and custom text (dollar amount)
        hoverinfo='label+percent+value',
        marker=dict(colors=colors, line=dict(color='#ffffff', width=1)),
        showlegend=True
    )])

    # Update the layout
    fig.update_layout(
        title=f'Net Price by Product Category (Total: {total_amount_str})',
        legend_title="Product Category",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1
        )
    )

    # Remove axes
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    # Improve text readability inside the slices 
    fig.update_traces(textfont_size=12, textfont_color='black', insidetextorientation='radial')

    return fig

def create_bookings_count_id_pie_chart(df):
    """
    Create a pie chart to visualize the count of bookings by product category.

    Args:
        df (pd.DataFrame): The processed bookings data with MultiIndex.

    Returns:
        plotly.graph_objects.Figure: The pie chart figure.
    """
    # Sum the bookings for each product category across all months
    category_counts = df.sum().sort_values(ascending=False)
    
    # Calculate percentages
    total = category_counts.sum()
    category_percentages = (category_counts / total * 100).round(2)
    
    # Generate colors based on the number of categories
    colors = generate_color_palette(len(category_counts))

    # Create labels with count and percentage
    labels = [f"{cat}<br>${count:,.0f} ({percentage:.2f}%)" 
              for cat, count, percentage in zip(category_counts.index, category_counts, category_percentages)]

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts,
        text=labels,
        textposition='inside',
        textinfo='label+percent',
        hoverinfo='label+percent+value',
        marker=dict(colors=colors, line=dict(color='#ffffff', width=1)),
        showlegend=True
    )])

    # Update the layout
    fig.update_layout(
        title='Net Bookings by Product Category',
        legend_title="Product Category",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1
        )
    )

    # Remove axes
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    # Improve text readability inside the slices - Dark color for text
    fig.update_traces(textfont_size=12, textfont_color='black', insidetextorientation='radial')

    return fig

# Sales Pipeline
def fetch_sales_pipeline_data(sf, start_date, end_date, brand=None):
    """
    Fetch sales pipeline data from Salesforce within a specified date range,
    optionally filtered by brand.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    query = load_sql_query("SalesPipeline.sql").format(start_date=start_date_str, end_date=end_date_str)

    if brand:
        if isinstance(brand, list):
            brand_str = ", ".join([f"'{b}'" for b in brand])
            query += f" AND Brand__c IN ({brand_str})"
        else:
            query += f" AND Brand__c = '{brand}'"

    result = sf.query_all(query)
    df = pd.DataFrame(result["records"]).drop(columns=["attributes"])
    return df

def process_sales_pipeline_data(df):
    """
    Process the sales pipeline data.
    """
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
    df['CloseDate'] = pd.to_datetime(df['CloseDate'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    return df

def create_pipeline_stage_chart(df):
    """
    Create a bar chart showing the number of opportunities by stage.
    """
    stage_counts = df['StageName'].value_counts().reset_index()
    stage_counts.columns = ['Stage', 'Count']
    fig = px.bar(stage_counts, x='Stage', y='Count', title='Number of Opportunities by Stage')
    return fig

def create_win_loss_chart(df):
    """
    Create a pie chart showing the win/loss ratio.
    """
    win_loss = df['IsWon'].value_counts().reset_index()
    win_loss.columns = ['Status', 'Count']
    win_loss['Status'] = win_loss['Status'].map({True: 'Won', False: 'Lost'})
    fig = px.pie(win_loss, values='Count', names='Status', title='Win/Loss Ratio')
    return fig

def create_average_deal_size_chart(df):
    """
    Create a bar chart showing the average deal size by stage.
    """
    avg_deal_size = df.groupby('StageName')['Amount'].mean().reset_index()
    fig = px.bar(avg_deal_size, x='StageName', y='Amount', title='Average Deal Size by Stage')
    fig.update_layout(yaxis_title='Average Amount')
    return fig

def main():
    """Main function to run the Streamlit application."""

    # Load brands from brands.json
    with open('brands.json', 'r') as f:
        brands_by_vertical = json.load(f)

    # First, check if the user is authenticated
    if check_password():
        # Sidebar for navigation
        with st.sidebar:
            st.markdown("### Select Business Vertical")
            selected_vertical = st.selectbox("Choose", list(brands_by_vertical.keys()))

            # Dynamically create the brand selection based on the chosen vertical
            available_brands = brands_by_vertical[selected_vertical]
            selected_brand = st.selectbox("Select Brand", ["ALL"] + available_brands)

            with st.sidebar:
                st.markdown("### Select Report")
            report_type = st.radio(
                "Choose Report Type", 
                ("Orders", "MQLs", "Leads", "Open Opportunities", "Customer Lifetime Value", "Bookings", "Sales Pipeline", "SQL Bench")
            )

        # Main dashboard title
        st.title("Salesforce Data Dashboard")

        if report_type == "SQL Bench":
            st.markdown("### SQL Bench")
            user_question = st.text_area("Ask a question to generate a SOQL query:", value=st.session_state.get("last_question", ""))

            if st.button("Generate Query"):
                if user_question:
                    with st.spinner("Generating SOQL query..."):
                        salesforce = connect_to_salesforce()
                        if salesforce:
                            # Generate the SOQL query based on the user's question
                            soql_query = generate_soql_query_with_dynamic_dates(user_question, salesforce_data, object_aliases)
                            st.session_state.generated_query = soql_query  # Save to session state
                            st.session_state.last_question = user_question  # Track last question
                        else:
                            st.error("Salesforce connection failed.")
                else:
                    st.warning("Please enter a question.")

            # Prepopulate the text area with the generated query
            sql_input = st.text_area("Enter your SOQL query here:", value=st.session_state.get("generated_query", ""))
            
            if st.button("Run Query"):
                if sql_input:
                    with st.spinner("Running query..."):
                        salesforce = connect_to_salesforce()  # Connect to Salesforce

                        if salesforce:
                            try:
                                # Run query with retry mechanism and OpenAI refinement
                                dataframe = run_soql_query_until_not_empty(salesforce, openai_model, sql_input, salesforce_data)

                                # Handle the case where the result is None (error occurred) or empty (no data)
                                if dataframe is not None:
                                    if not dataframe.empty:
                                        st.dataframe(dataframe)  # Display the DataFrame in the app

                                        # CSV download button for query results
                                        query_results_csv = dataframe.to_csv(index=False)
                                        st.download_button(
                                            label="Download Query Results CSV",
                                            data=query_results_csv,
                                            file_name="query_results.csv",
                                            mime="text/csv",
                                        )
                                    else:
                                        st.warning("The query returned no data.")
                                else:
                                    st.error("An error occurred during the query execution.")

                            except Exception as e:
                                st.error(f"Error executing query: {e}")
                        else:
                            st.error("Not connected to Salesforce.")
                else:
                    st.warning("Please enter a SOQL query.")

        else:
            # Display date range selection and "Fetch Data" button for other reports
            column1, column2 = st.columns(2)
            with column1:
                date_option = st.selectbox(
                    "Select Date Range",
                    (
                        "Month to Date (Current Month)",
                        "Last 7 days",
                        "Last Month",
                        "Last 3 months",
                        "Last 6 months",
                        "Last 12 months",
                        "Year to Date",
                        "Custom",
                    ),
                )

                if date_option == "Custom":
                    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
                    end_date = st.date_input("End Date", value=datetime.now())
                elif date_option == "Year to Date":
                    start_date = datetime(datetime.now().year, 1, 1)
                    end_date = datetime.now()
                elif date_option == "Month to Date (Current Month)":
                    start_date = datetime.now().replace(day=1)
                    end_date = datetime.now()
                elif date_option == "Last Month":
                    today = datetime.now()
                    last_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
                    last_month_end = today.replace(day=1) - timedelta(days=1)
                    start_date = last_month_start
                    end_date = last_month_end
                else:
                    end_date = datetime.now()
                    start_date_map = {"Last 7 days": 7, "Last 3 months": 90, "Last 6 months": 180, "Last 12 months": 365}
                    start_date = end_date - timedelta(days=start_date_map[date_option])

            # Fetch and display data on button click
            if st.button("Fetch Data"):
                with st.spinner("Fetching data from Salesforce..."):
                    salesforce = connect_to_salesforce()
                    if not salesforce:
                        return

                    # --- Handling Report Types based on Vertical and Brand ---

                    if selected_brand == "ALL":
                        brand_filter = available_brands  # Fetch data for all brands in the vertical
                    else:
                        brand_filter = selected_brand

                    if report_type == "Orders":
                        dataframe = fetch_orders_data(salesforce, start_date, end_date, brand=brand_filter)
                        wicked_reports_dataframe = fetch_wicked_reports_data(salesforce, start_date, end_date, brand=brand_filter)
                        
                        if dataframe.empty:
                            st.warning("No data found for the selected date range.")
                        else:
                            processed_dataframe = process_data(dataframe)
                            processed_dataframe = calculate_cltv(processed_dataframe)

                            # Check if processed_dataframe is None (no CLTV data)
                            if processed_dataframe is not None:
                                # Display key metrics and visualizations
                                column1, column2, column3 = st.columns(3)
                                with column1:
                                    st.metric("Total Orders", len(processed_dataframe))
                                with column2:
                                    if "ORDERTOTAL" in processed_dataframe.columns:
                                        total_revenue = processed_dataframe["ORDERTOTAL"].sum()
                                        st.metric("Total Revenue", f"${total_revenue:,.2f}")
                                    else:
                                        st.metric("Total Revenue", "N/A")
                                with column3:
                                    if "ORDERTOTAL" in processed_dataframe.columns:
                                        average_order_value = processed_dataframe["ORDERTOTAL"].mean()
                                        st.metric("Average Order Value", f"${average_order_value:,.2f}")
                                    else:
                                        st.metric("Average Order Value", "N/A")

                                # Orders by State visualization
                                state_counts = processed_dataframe["CUSTOMERSTATE"].value_counts().reset_index()
                                state_counts.columns = ["State", "Order Count"]
                                st.plotly_chart(
                                    px.bar(state_counts, x="State", y="Order Count", title="Orders by State"),
                                    use_container_width=True,
                                )

                                # Orders by Product Category (Pie Chart)
                                category_counts = processed_dataframe["Product_Category__c"].value_counts().reset_index()
                                category_counts.columns = ["Product Category", "Order Count"]
                                st.plotly_chart(
                                    px.pie(
                                        category_counts,
                                        values="Order Count",
                                        names="Product Category",
                                        title=f"Orders by Product Category ({selected_brand if selected_brand != 'ALL' else selected_vertical})",
                                    ),
                                    use_container_width=True,
                                )

                                # Product Adoption Chart Section
                                st.markdown(f'<h3 style="font-family: \'Open Sans\', sans-serif; font-size: 14px; font-weight: 600; margin-bottom: 10px;">Product Adoption Rates Over Time ({selected_brand if selected_brand != "ALL" else selected_vertical})</h3>', unsafe_allow_html=True)
                                
                                # Check if the date range is at least 3 months
                                date_difference = (end_date - start_date).days
                                if date_difference >= 90:  # Approximately 3 months
                                    product_adoption = calculate_product_adoption(processed_dataframe)

                                    # Display product adoption chart
                                    if not product_adoption.empty:
                                        st.plotly_chart(
                                            px.line(
                                                product_adoption,
                                                x="OrderMonth",
                                                y="AdoptionRate",
                                                color="PRODUCTNAME",
                                            ).update_layout(legend_title_text=f"{selected_brand if selected_brand != 'ALL' else selected_vertical} Products"),
                                            use_container_width=True,
                                        )
                                else:
                                    st.warning("Please select a date range of at least 3 months to view data.")

                                # Display data table
                                st.subheader("Salesforce Data Table")
                                st.dataframe(processed_dataframe)

                                # CSV download buttons
                                complete_csv = processed_dataframe.to_csv(index=False)
                                st.download_button(
                                    label="Download Complete CSV",
                                    data=complete_csv,
                                    file_name="salesforce_export.csv",
                                    mime="text/csv",
                                )

                                if not wicked_reports_dataframe.empty:
                                    wicked_dataframe = prepare_wicked_reports_export(wicked_reports_dataframe)
                                    st.subheader("Wicked Reports Data Table")
                                    st.dataframe(wicked_dataframe)
                                    wicked_csv = wicked_dataframe.to_csv(index=False)
                                    st.download_button(
                                        label="Download Wicked Reports CSV",
                                        data=wicked_csv,
                                        file_name="wicked_reports_export.csv",
                                        mime="text/csv",
                                    )
                                else:
                                    st.warning("No data available for Wicked Reports export.")
                            else:
                                st.warning("Unable to calculate CLTV.")

                    elif report_type == "MQLs":
                        # --- Handling Report Types based on Vertical and Brand ---
                        if selected_brand == "ALL":
                            # Get the Salesforce IDs of all brands in the vertical
                            brand_filter = [get_brand_id(salesforce, b) for b in available_brands]
                        else:
                            # Get the Salesforce ID of the selected brand
                            brand_filter = [get_brand_id(salesforce, selected_brand)]

                        # Filter out None values from brand_filter
                        brand_filter = [brand_id for brand_id in brand_filter if brand_id is not None]

                        dataframe = fetch_mql_data(salesforce, start_date, end_date, brand_ids=brand_filter)  # Pass as brand_ids
                        campaign_dataframe = fetch_campaign_data(salesforce, start_date, end_date)

                        if dataframe.empty:
                            st.warning("No MQL data found for the selected date range.")
                        else:
                            dataframe = process_mql_data(dataframe)
                            st.subheader("MQL Dashboard")

                            # Display MQL metrics and charts
                            total_mqls = len(dataframe)
                            average_lead_age = dataframe["LeadAge"].mean()

                            column1, column2, column3 = st.columns(3)
                            with column1:
                                st.metric("How many MQLs did we generate?", total_mqls)
                            with column2:
                                st.metric("What's the average new lead to MQL age?", f"{average_lead_age:.1f} days")
                            with column3:
                                target, mql_count, period_name = calculate_target_and_mql_count(
                                    dataframe, date_option, start_date, end_date
                                )
                                st.metric(f"MQLs This {period_name} vs Target", f"{mql_count}/{target}")

                            # Enhanced charts and visualizations:
                            st.plotly_chart(LeadSourcePerformanceCard(dataframe))
                            st.plotly_chart(MQLTrendOverTimeCard(dataframe))
                            st.plotly_chart(TimeToSQLCard(dataframe))
                            st.plotly_chart(MQLFunnelCard(dataframe))

                            st.subheader("MQL Data Table")
                            st.dataframe(dataframe)
                            st.download_button(
                                label="Download MQL CSV",
                                data=dataframe.to_csv(index=False),
                                file_name="mql_export.csv",
                                mime="text/csv",
                            )

                    elif report_type == "Leads":
                        dataframe = fetch_lead_data(salesforce, start_date, end_date, brand=brand_filter)

                        if dataframe.empty:
                            st.warning("No lead data found for the selected date range.")
                        else:
                            dataframe = process_lead_data(dataframe)
                            st.subheader("Lead Dashboard")

                            # Display Lead metrics and charts
                            total_leads = len(dataframe)
                            average_lead_age = dataframe["LeadAge"].mean()

                            column1, column2 = st.columns(2)
                            with column1:
                                st.metric("Total Leads", total_leads)
                            with column2:
                                st.metric("Average Lead Age", f"{average_lead_age:.1f} days")

                            # Display Lead charts
                            st.plotly_chart(LeadsBySourceCard(dataframe))
                            st.plotly_chart(LeadConversionOverTimeCard(dataframe))
                            st.plotly_chart(LeadAgeDistributionCard(dataframe))
                            st.plotly_chart(LeadConversionFunnelCard(dataframe))

                            # Display data table
                            st.subheader("Lead Data Table")
                            st.dataframe(dataframe)

                            # CSV download button
                            csv_data = dataframe.to_csv(index=False)
                            st.download_button(
                                label="Download Leads CSV",
                                data=csv_data,
                                file_name="leads_export.csv",
                                mime="text/csv",
                            )

                    elif report_type == "Open Opportunities":
                        dataframe = fetch_open_opportunities_data(salesforce, start_date, end_date, brand=brand_filter)

                        if dataframe.empty:
                            st.warning("No open opportunities found for the selected date range.")
                        else:
                            dataframe = process_open_opportunities_data(dataframe)
                            st.subheader("Open Opportunities Dashboard")

                            # Display Open Opportunities metrics and charts
                            total_open_opportunities = len(dataframe)
                            average_days_open = dataframe["DaysOpen"].mean()
                            total_value = dataframe["Amount"].sum()

                            column1, column2, column3 = st.columns(3)
                            with column1:
                                st.metric("Total Open Opportunities", total_open_opportunities)
                            with column2:
                                st.metric("Average Days Open", f"{average_days_open:.1f} days")
                            with column3:
                                st.metric("Total Opportunity Value", f"${total_value:,.2f}")

                            # Display Open Opportunities charts
                            st.plotly_chart(OpenOpportunitiesByStageCard(dataframe))
                            st.plotly_chart(OpenOpportunitiesTrendCard(dataframe))

                            st.subheader("Open Opportunities Data Table")
                            st.dataframe(dataframe)
                            st.download_button(
                                label="Download Open Opportunities CSV",
                                data=dataframe.to_csv(index=False),
                                file_name="open_opportunities_export.csv",
                                mime="text/csv",
                            )

                    elif report_type == "Customer Lifetime Value":
                        dataframe = fetch_orders_data(salesforce, start_date, end_date, brand=brand_filter)

                        if dataframe.empty:
                            st.warning("No data found for the selected date range.")
                        else:
                            processed_dataframe = process_data(dataframe)
                            cltv_df = calculate_cltv(processed_dataframe)

                            # Check if cltv_df is None
                            if cltv_df is not None:
                                st.subheader("Customer Lifetime Value")

                                # Display CLTV metrics
                                avg_cltv = cltv_df["CLTV"].mean()
                                st.metric("Average CLTV", f"${avg_cltv:,.2f}")

                                # Display CLTV histogram
                                st.plotly_chart(create_cltv_histogram(cltv_df), use_container_width=True)

                                # Display CLTV by product category chart
                                st.plotly_chart(create_cltv_by_product_category_chart(cltv_df), use_container_width=True)

                                # Display data table
                                st.dataframe(cltv_df)

                                # CSV download button
                                cltv_csv = cltv_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CLTV Report CSV",
                                    data=cltv_csv,
                                    file_name="cltv_report.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.warning("Unable to calculate CLTV due to insufficient data.")

                    elif report_type == "Bookings":
                        bookings_df = fetch_and_process_bookings_data(salesforce, start_date, end_date, brand=brand_filter)

                        if bookings_df.empty:
                            st.warning("No bookings data found for the selected date range.")
                        else:
                            st.subheader("Bookings Dashboard")

                            # Display total bookings
                            total_bookings = bookings_df.sum().sum()
                            st.metric("Total Bookings", f"${total_bookings:,.2f}")

                            # Display count of IDs by Product Category pie chart
                            count_pie_chart = create_bookings_count_pie_chart(bookings_df)
                            if count_pie_chart:
                                st.plotly_chart(count_pie_chart, use_container_width=True)

                            # Display count of bookings by Product Category pie chart
                            count_pie_chart = create_bookings_count_id_pie_chart(bookings_df)
                            st.plotly_chart(count_pie_chart, use_container_width=True)

                            # Display stacked bar chart
                            st.plotly_chart(create_bookings_stacked_bar_chart(bookings_df), use_container_width=True)

                            # Display line chart
                            st.plotly_chart(create_bookings_chart(bookings_df), use_container_width=True)

                            # Display data table
                            st.subheader("Bookings Data Table")
                            st.dataframe(bookings_df)

                            # CSV download button
                            bookings_csv = bookings_df.to_csv()
                            st.download_button(
                                label="Download Bookings CSV",
                                data=bookings_csv,
                                file_name="bookings_export.csv",
                                mime="text/csv",
                            )

                    elif report_type == "Sales Pipeline":
                        dataframe = fetch_sales_pipeline_data(salesforce, start_date, end_date, brand=brand_filter)

                        if dataframe.empty:
                            st.warning("No sales pipeline data found for the selected date range.")
                        else:
                            dataframe = process_sales_pipeline_data(dataframe)
                            st.subheader("Sales Pipeline Overview")

                            # Display key metrics
                            total_opportunities = len(dataframe)
                            total_pipeline_value = dataframe['Amount'].sum()
                            average_deal_size = dataframe['Amount'].mean()

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Opportunities", total_opportunities)
                            with col2:
                                st.metric("Total Pipeline Value", f"${total_pipeline_value:,.2f}")
                            with col3:
                                st.metric("Average Deal Size", f"${average_deal_size:,.2f}")

                            # Display charts
                            st.plotly_chart(create_pipeline_stage_chart(dataframe), use_container_width=True)
                            st.plotly_chart(create_win_loss_chart(dataframe), use_container_width=True)
                            st.plotly_chart(create_average_deal_size_chart(dataframe), use_container_width=True)

                            # Display data table
                            st.subheader("Sales Pipeline Data Table")
                            st.dataframe(dataframe)

                            # CSV download button
                            csv_data = dataframe.to_csv(index=False)
                            st.download_button(
                                label="Download Sales Pipeline CSV",
                                data=csv_data,
                                file_name="sales_pipeline_export.csv",
                                mime="text/csv",
                            )

if __name__ == "__main__":
    main()