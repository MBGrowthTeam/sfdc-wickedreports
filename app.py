# Frontend/UI Libraries
import streamlit as st  # Streamlit for building interactive web apps

# Set the page configuration as the first Streamlit command
st.set_page_config(layout="wide", page_title="Salesforce Data Dashboard")

# Data Manipulation/Analysis
import pandas as pd  # Pandas for data manipulation and analysis

# Data Visualization
import plotly.express as px  # Plotly Express for quick, interactive plots
import plotly.graph_objects as go  # Plotly Graph Objects for detailed, customizable visualizations

# Salesforce Integration
from simple_salesforce import Salesforce  # Simple Salesforce for interacting with Salesforce API

# Date and Time Handling
from datetime import datetime, timedelta  # DateTime and Timedelta for manipulating dates and times
import pytz  # Pytz for timezone handling

# Environment Management
from dotenv import load_dotenv  # Load environment variables from a .env file
import os  # OS for interacting with the operating system, managing environment variables, file paths, etc.

# Type Hinting and Annotations
from typing import Optional, Any, Literal  # Typing for type annotations, enhancing code readability and static analysis

# Concurrency
import threading  # Threading for running concurrent operations

# Specialized Libraries
import dspy  # (Assumed to be a specialized data science or signal processing library)

# Configuration Parsing
import toml  # TOML for reading and parsing configuration files


# Load environment variables from .env file
load_dotenv()

# Load secrets from .streamlit/secrets.toml
secrets = st.secrets

# Salesforce credentials from secrets
SF_USERNAME = secrets["salesforce"]["SF_USERNAME"]
SF_PASSWORD = secrets["salesforce"]["SF_PASSWORD"]
SF_TOKEN = secrets["salesforce"]["SF_TOKEN"]
SF_CONSUMER_KEY = secrets["salesforce"]["SF_CONSUMER_KEY"]
SF_CONSUMER_SECRET = secrets["salesforce"]["SF_CONSUMER_SECRET"]

# OpenAI API key from secrets
OPENAI_API_KEY = secrets["openai"]["api_key"]


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


def fetch_salesforce_data(sf, start_date, end_date):
    """
    Fetch order data from Salesforce within a specified date range.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.

    Returns:
        A Pandas DataFrame containing the order data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    query = f"""
    SELECT 
        Order.Id,
        Order.CreatedDate,
        Account.Time_Zone__c,
        Order.TotalAmount,
        Account.Account_Primary_Contact_Email__c,
        Order.BillingState,
        Order.BillingCountry,
        Order.CurrencyIsoCode,
        Order.BillingCity,
        Order.Brand__c,
        (SELECT 
            SBQQ__Subscription__c,
            Product_Category__c,
            Product2Id,
            Product_Name__c,
            Product2.Name,
            Product2.Family
         FROM OrderItems)
    FROM Order
    WHERE Order.Brand__c IN ('Ministry Brands', 'Amplify')
    AND Order.CreatedDate >= {start_date_str} 
    AND Order.CreatedDate <= {end_date_str}
    ORDER BY Order.CreatedDate
    """

    result = sf.query_all(query)
    orders = pd.DataFrame(result["records"])

    # Flatten nested order items data
    order_items = []
    for order in result["records"]:
        for item in order.get("OrderItems", {}).get("records", []):
            order_items.append(
                {
                    "OrderId": order["Id"],
                    "SBQQ__Subscription__c": item.get("SBQQ__Subscription__c"),
                    "Product_Category__c": item.get("Product_Category__c"),  # Correct column name
                    "ProductName": item.get("Product_Name__c"),
                }
            )

    order_items_df = pd.DataFrame(order_items)

    df = orders.merge(order_items_df, left_on="Id", right_on="OrderId", how="left")

    df = df.drop(columns=["attributes", "OrderItems"])
    df["Account.Time_Zone__c"] = df["Account"].apply(lambda x: x["Time_Zone__c"] if x else None)
    df["Account.Account_Primary_Contact_Email__c"] = df["Account"].apply(
        lambda x: x["Account_Primary_Contact_Email__c"] if x else None
    )
    df = df.drop(columns=["Account"])

    return df


def fetch_mql_data(sf, start_date, end_date):
    """
    Fetch MQL (Marketing Qualified Leads) data from Salesforce within a specified date range.

    Args:
        sf: Salesforce connection object.
        start_date: Start date for the data range.
        end_date: End date for the data range.

    Returns:
        A Pandas DataFrame containing the MQL data.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
    query = f"""
    SELECT 
        Id, OwnerId, IsDeleted, Name, CurrencyIsoCode, CreatedDate, CreatedById, 
        LastModifiedDate, LastModifiedById, SystemModstamp, LastActivityDate, 
        LastViewedDate, LastReferencedDate, Account__c, Address__c, Age__c, 
        Annual_Budget__c, Brand__c, Campaign__c, Company__c, Contact__c, 
        Description__c, Disqualification_Date__c, E_Mail_Address__c, 
        E_Mail_Opt_Out__c, HS_Owner__c, Households__c, Hubspot_Notes__c, 
        Industry__c, Lead_Conversion_Date__c, Lead_Source__c, Lead__c, 
        MQL_AutoNumber__c, Mobile__c, Name__c, DEPRECATED_Name_of_the_referrer__c, 
        Number_of_Employees__c, Opportunity__c, Phone__c, Product_Interest__c, 
        Status__c, TWA__c, Title__c, Website__c, Salesloft_Cadence_Name__c, 
        Parent_Account__c, Latest_Source_Drill_Down_1__c, 
        Number_of_Activities_Converted__c, Number_of_Activities__c, 
        Converted_to_Opportunity__c, MQL_Converted_Date__c, Name_of_the_referrer__c, 
        Latest_Source_Drill_Down_2__c, Latest_Source__c, MQL_Reason__c, 
        Google_ad_click_id__c, Last_Page_Seen__c, Screenings_Per_Year__c, 
        Referring_Page__c, Account_Engagement_Comments__c, Prospect_Zip_Code__c, 
        Self_Declared_Customer_From_Form__c, Account_Engagement_Score__c, 
        Partner_Form_Date__c, Partner_Name_Company__c
    FROM MQL_Record__c
    WHERE CreatedDate >= {start_date_str} 
    AND CreatedDate <= {end_date_str}
    """
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
    query = f"""
    SELECT 
        Id, Name, CreatedDate
    FROM Campaign
    WHERE CreatedDate >= {start_date_str} 
    AND CreatedDate <= {end_date_str}
    """
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


def fetch_wicked_reports_data(sf, start_date, end_date):
    """
    Fetches order data from Salesforce specifically for Wicked Reports export.

    Args:
        sf (Salesforce): The Salesforce connection object.
        start_date (datetime): The start date for the data retrieval.
        end_date (datetime): The end date for the data retrieval.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the fetched order data for Wicked Reports.
    """
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    query = f"""
    SELECT 
        Id,
        CreatedDate,
        Account.Time_Zone__c,
        TotalAmount,
        Account.Account_Primary_Contact_Email__c,
        BillingState,
        BillingCountry,
        CurrencyIsoCode,
        BillingCity,
        Brand__c,
        (SELECT 
            SBQQ__Subscription__c,
            Product2Id,
            Product_Name__c
         FROM OrderItems)
    FROM Order
    WHERE Brand__c IN ('Ministry Brands', 'Amplify')
    AND CreatedDate >= {start_date_str} 
    AND CreatedDate <= {end_date_str}
    ORDER BY CreatedDate
    """

    result = sf.query_all(query)
    orders = pd.DataFrame(result["records"])

    # Flatten the nested OrderItems data
    order_items = []
    for order in result["records"]:
        for item in order.get("OrderItems", {}).get("records", []):
            order_items.append(
                {
                    "OrderId": order["Id"],
                    "SBQQ__Subscription__c": item.get("SBQQ__Subscription__c"),
                    "Product2Id": item.get("Product2Id"),
                    "Product_Name__c": item.get("Product_Name__c"),
                }
            )

    order_items_df = pd.DataFrame(order_items)

    # Merge orders with order items
    df = orders.merge(order_items_df, left_on="Id", right_on="OrderId", how="left")

    # Clean up the DataFrame
    df = df.drop(columns=["attributes", "OrderItems"], errors="ignore")
    df["Time_Zone__c"] = df["Account"].apply(lambda x: x["Time_Zone__c"] if x else None)
    df["Account_Primary_Contact_Email__c"] = df["Account"].apply(
        lambda x: x["Account_Primary_Contact_Email__c"] if x else None
    )
    df = df.drop(columns=["Account"])

    return df


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


def calculate_cltv(df):
    """
    Calculate Customer Lifetime Value (CLTV) for each customer.

    Args:
        df (pd.DataFrame): The DataFrame containing order data.

    Returns:
        pd.DataFrame: The DataFrame with an added 'CLTV' column.
    """
    customer_orders = df.groupby("CUSTOMEREMAIL")
    customer_avg_order_value = customer_orders["ORDERTOTAL"].mean()
    customer_purchase_frequency = customer_orders["ORDERID"].count()

    # Assume 12 months CLTV
    cltv = customer_avg_order_value * customer_purchase_frequency * 12
    cltv_df = pd.DataFrame({"CUSTOMEREMAIL": cltv.index, "CLTV": cltv.values})

    return df.merge(cltv_df, on="CUSTOMEREMAIL", how="left")


def calculate_product_adoption(df):
    """
    Calculate product adoption rates over time, particularly for Amplify products.

    Args:
        df (pd.DataFrame): The DataFrame containing order data.

    Returns:
        pd.DataFrame: A DataFrame with product adoption rates over time.
    """
    df["ORDERDATETIME"] = pd.to_datetime(df["ORDERDATETIME"], errors="coerce")
    df = df.dropna(subset=["ORDERDATETIME"])

    if "PRODUCTNAME" in df.columns:
        df["OrderYear"] = df["ORDERDATETIME"].dt.year
        df["OrderMonth"] = df["ORDERDATETIME"].dt.month
        amplify_df = df[df["PRODUCTNAME"].str.contains("Amplify", case=False, na=False)]

        # Group by product, year, and month, then count orders
        product_orders = amplify_df.groupby(["PRODUCTNAME", "OrderYear", "OrderMonth"])["ORDERID"].count().reset_index()

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
    mql_to_sql_query = f"""
    SELECT MQL_Converted_Date__c, CreatedDate
    FROM MQL_Record__c
    WHERE CreatedDate >= {start_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')}
    AND CreatedDate <= {end_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')}
    AND MQL_Converted_Date__c != NULL
    """
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
    sql_to_won_query = f"""
    SELECT CloseDate, CreatedDate
    FROM Opportunity
    WHERE CreatedDate >= {start_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')}
    AND CreatedDate <= {end_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')}
    AND IsWon = TRUE
    """
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


# Helper Function to Display MQL Cards
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


# MQL Card Functions
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
    Visualize lead source performance with MQL count.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    lead_source_data = df["Lead_Source__c"].value_counts().reset_index()
    lead_source_data.columns = ["Lead Source", "Count"]
    return px.bar(lead_source_data, x="Lead Source", y="Count", title="Lead Source Performance")


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
    Visualize MQL trend over time.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure: The line chart figure.
    """
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    mql_trend = df.groupby(df["CreatedDate"].dt.date).size().reset_index(name="Count")
    return px.line(mql_trend, x="CreatedDate", y="Count", title="MQL Trend Over Time")


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
    Visualize time to SQL in days with adjusted labels and handling of small data sets.

    Args:
        df (pd.DataFrame): The DataFrame containing MQL data.

    Returns:
        plotly.graph_objects.Figure or None: The histogram figure, or None if the dataset is too small.
    """

    # Calculate TimeToSQL
    df["TimeToSQL"] = (df["MQL_Converted_Date__c"] - df["CreatedDate"]).dt.days

    # Remove negative values (if any)
    df = df[df["TimeToSQL"] >= 0]

    # Handle small datasets by checking if there are enough data points
    if df["TimeToSQL"].nunique() > 1:
        median_value = df["TimeToSQL"].median()
        mean_value = df["TimeToSQL"].mean()

        fig = px.histogram(df, x="TimeToSQL", title="Time to SQL (in days)")
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
    else:
        return None


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


def main():
    """
    Main function to run the Streamlit application.
    """

    # Sidebar for navigation
    with st.sidebar:

        st.markdown("### Select Business Vertical")
        vertical = st.selectbox("Choose", ("Protestant (Default)", "Catholic", "Protection", "Non-Profit"))

        st.markdown("### Select Report")
        report_type = st.radio("Choose Report Type", ("Orders", "Lead Source Pipeline (MQLs)"))

    # Main dashboard title
    st.title("Salesforce Data Dashboard")

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
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

    if st.button("Fetch Data"):
        with st.spinner("Fetching data from Salesforce..."):
            sf = connect_to_salesforce()
            if not sf:
                return

            if report_type == "Orders":
                df = fetch_salesforce_data(sf, start_date, end_date)
                wicked_reports_df = fetch_wicked_reports_data(sf, start_date, end_date)

                if df.empty:
                    st.warning("No data found for the selected date range.")
                else:
                    processed_df = process_data(df)
                    processed_df = calculate_cltv(processed_df)
                    product_adoption = calculate_product_adoption(processed_df)

                    # Display key metrics and visualizations
                    if not product_adoption.empty:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Orders", len(processed_df))
                        with col2:
                            if "ORDERTOTAL" in processed_df.columns:
                                total_revenue = processed_df["ORDERTOTAL"].sum()
                                st.metric("Total Revenue", f"${total_revenue:,.2f}")
                            else:
                                st.metric("Total Revenue", "N/A")
                        with col3:
                            if "ORDERTOTAL" in processed_df.columns:
                                avg_order_value = processed_df["ORDERTOTAL"].mean()
                                st.metric("Average Order Value", f"${avg_order_value:,.2f}")
                            else:
                                st.metric("Average Order Value", "N/A")

                        # Corrected Orders by State visualization
                        state_counts = processed_df["CUSTOMERSTATE"].value_counts().reset_index()
                        state_counts.columns = ["State", "Order Count"]
                        st.plotly_chart(
                            px.bar(state_counts, x="State", y="Order Count", title="Orders by State"),
                            use_container_width=True,
                        )

                        # Orders by Product Category (Pie Chart)
                        category_counts = processed_df["Product_Category__c"].value_counts().reset_index()
                        category_counts.columns = ["Product Category", "Order Count"]
                        st.plotly_chart(
                            px.pie(
                                category_counts,
                                values="Order Count",
                                names="Product Category",
                                title="Orders by Product Category (Amplify)",
                            ),
                            use_container_width=True,
                        )

                        window = {
                            "Last 7 days": 7,
                            "Last Month": 14,
                            "Last 3 months": 30,
                            "Last 6 months": 45,
                            "Last 12 months": 60,
                            "Year to Date": 60,
                        }.get(date_option, 7)
                        trend_type = (
                            "monthly"
                            if date_option in ["Last 3 months", "Last 6 months", "Last 12 months", "Year to Date"]
                            else "daily"
                        )

                        # Convert ORDERDATETIME to monthly periods without dropping timezone
                        if trend_type == "monthly":
                            processed_df["ORDERDATETIME"] = (
                                processed_df["ORDERDATETIME"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
                            )
                        else:
                            processed_df["ORDERDATETIME"] = processed_df["ORDERDATETIME"].dt.tz_convert(
                                None
                            )  # Remove timezone for daily data

                        fig_daily = go.Figure()
                        fig_daily.add_trace(
                            go.Scatter(
                                x=processed_df["ORDERDATETIME"],
                                y=processed_df["ORDERTOTAL"],
                                mode="lines",
                                name=f"{trend_type.capitalize()} Total",
                            )
                        )
                        fig_daily.add_trace(
                            go.Scatter(
                                x=processed_df["ORDERDATETIME"],
                                y=apply_moving_average(processed_df["ORDERTOTAL"], window),
                                mode="lines",
                                name=f"{window}-day Moving Average",
                            )
                        )
                        fig_daily.update_layout(title="Order Totals", xaxis_title="Date", yaxis_title="Order Total")
                        st.plotly_chart(fig_daily, use_container_width=True)

                        # Order Frequency Heatmap
                        processed_df["DayOfWeek"] = processed_df["ORDERDATETIME"].dt.dayofweek
                        processed_df["HourOfDay"] = processed_df["ORDERDATETIME"].dt.hour
                        heatmap_data = processed_df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
                        heatmap_data = (
                            heatmap_data.reindex(range(7), fill_value=0, axis=0)
                            .reindex(range(24), fill_value=0, axis=1)
                            .sort_index()
                        )
                        st.plotly_chart(
                            px.imshow(
                                heatmap_data,
                                labels=dict(x="Hour of Day", y="Day of Week", color="Order Count"),
                                x=list(range(24)),
                                y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                aspect="auto",
                            ).update_layout(title="Order Frequency Heatmap"),
                            use_container_width=True,
                        )

                        # Product Adoption Rate for Amplify Products
                        st.plotly_chart(
                            px.line(
                                product_adoption,
                                x="OrderMonth",
                                y="AdoptionRate",
                                color="PRODUCTNAME",
                                title="Amplify Product Adoption Rates Over Time",
                            ).update_layout(legend_title_text="Amplify Products"),
                            use_container_width=True,
                        )

                        # Display the data table
                        st.subheader("Salesforce Data Table")
                        st.dataframe(processed_df)

                        # Export button for complete data
                        complete_csv = processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Complete CSV",
                            data=complete_csv,
                            file_name="salesforce_export.csv",
                            mime="text/csv",
                        )

                        # Prepare and export Wicked Reports data
                        if not wicked_reports_df.empty:
                            wicked_df = prepare_wicked_reports_export(wicked_reports_df)

                            # Display Wicked Reports data table for verification
                            st.subheader("Wicked Reports Data Table")
                            st.dataframe(wicked_df)

                            # Wicked Reports download button
                            wicked_csv = wicked_df.to_csv(index=False)
                            st.download_button(
                                label="Download Wicked Reports CSV",
                                data=wicked_csv,
                                file_name="wicked_reports_export.csv",
                                mime="text/csv",
                            )
                        else:
                            st.warning("No data available for Wicked Reports export.")

            if report_type == "Lead Source Pipeline (MQLs)":
                df = fetch_mql_data(sf, start_date, end_date)
                campaign_df = fetch_campaign_data(sf, start_date, end_date)
                
                if vertical == "Protestant (Default)":
                    df = df[df["Product_Interest__c"] == "Amplify"]

                if df.empty:
                    st.warning("No MQL data found for the selected date range.")
                else:
                    df = process_mql_data(df)

                    st.subheader("MQL Dashboard")

                    # Calculate metrics
                    total_mqls = len(df)
                    avg_lead_age = df["LeadAge"].mean()

                    # Create layout
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("How many MQLs did we generate?", total_mqls)

                        # MQLs by month
                        df["Month"] = df["CreatedDate"].dt.tz_localize(None).dt.to_period("M")
                        monthly_mqls = df.groupby("Month").size().reset_index(name="Count")
                        monthly_mqls["Month"] = monthly_mqls["Month"].astype(str)
                        st.plotly_chart(px.bar(monthly_mqls, x="Month", y="Count", title="MQLs by month"))

                        # MQLs by Status
                        if "Status__c" in df.columns:
                            status_counts = df["Status__c"].value_counts()
                            fig = go.Figure(
                                data=[go.Pie(labels=status_counts.index, values=status_counts.values, hole=0.3)]
                            )
                            fig.update_layout(title="MQLs by Status")
                            st.plotly_chart(fig)
                        else:
                            st.warning("Status information is not available in the dataset.")

                        # Top Campaigns
                        st.subheader("Top Campaigns")
                        fig_campaigns = create_top_campaigns_chart(df, campaign_df)
                        st.plotly_chart(fig_campaigns, use_container_width=True)

                    with col2:
                        st.metric("What's the average new lead to MQL age?", f"{avg_lead_age:.1f} days")

                        # How many MQLs converted?
                        converted_df = df[df["Converted_to_Opportunity__c"] == True]  # Fixed line
                        conversions = (
                            converted_df.groupby(converted_df["CreatedDate"].dt.tz_localize(None).dt.to_period("M"))
                            .size()
                            .reset_index(name="Count")
                        )
                        conversions["Month"] = conversions["CreatedDate"].astype(str)
                        st.plotly_chart(px.bar(conversions, x="Month", y="Count", title="How many MQLs converted?"))

                        # MQLs vs Target for the selected period
                        target, mql_count, period_name = calculate_target_and_mql_count(
                            df, date_option, start_date, end_date
                        )

                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number+delta",
                                value=mql_count,
                                number={"suffix": " MQLs", "font": {"color": "#F8F8F8"}},
                                delta={
                                    "reference": target,
                                    "relative": True,
                                    "valueformat": ".1%",
                                    "font": {"color": "#F8F8F8"},
                                },
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={
                                    "text": f"MQLs This {period_name} vs Target ({target:.0f})",
                                    "font": {"size": 24, "color": "#F8F8F8"},
                                },
                                gauge={
                                    "axis": {"range": [None, target * 1.5], "tickwidth": 1, "tickcolor": "#F8F8F8"},
                                    "bar": {"color": "#4CAF50"},
                                    "bgcolor": "rgba(255, 255, 255, 0.1)",
                                    "borderwidth": 2,
                                    "bordercolor": "#F8F8F8",
                                    "steps": [
                                        {"range": [0, target * 0.5], "color": "#FF5252"},
                                        {"range": [target * 0.5, target * 0.75], "color": "#FFC107"},
                                        {"range": [target * 0.75, target], "color": "#FFEB3B"},
                                        {"range": [target, target * 1.5], "color": "#4CAF50"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "#F8F8F8", "width": 4},
                                        "thickness": 0.75,
                                        "value": target,
                                    },
                                },
                            )
                        )

                        fig.update_layout(
                            height=400,
                            font={"color": "#F8F8F8", "family": "Arial"},
                            paper_bgcolor="#0F1116",
                            plot_bgcolor="#0F1116",
                        )

                        st.plotly_chart(fig)

                    with col3:
                        # Sales Velocity Funnel
                        st.subheader("Sales Velocity Funnel")
                        velocity_data = calculate_velocity(sf, start_date, end_date)
                        fig_funnel = create_velocity_funnel(velocity_data)
                        st.plotly_chart(fig_funnel, use_container_width=True)

                        # Median and Average days from MQL to SQL
                        if "MQL_Converted_Date__c" in df.columns:
                            df["DaysToSQL"] = (df["MQL_Converted_Date__c"] - df["CreatedDate"]).dt.total_seconds() / (
                                24 * 60 * 60
                            )
                            median_days_to_sql = df["DaysToSQL"].median()
                            avg_days_to_sql = df["DaysToSQL"].mean()

                            fig = go.Figure()
                            fig.add_trace(
                                go.Indicator(
                                    mode="number+delta",
                                    value=median_days_to_sql,
                                    title={"text": "Median Days from MQL to SQL"},
                                    domain={"y": [0, 0.5], "x": [0, 1]},
                                )
                            )
                            fig.add_trace(
                                go.Indicator(
                                    mode="number+delta",
                                    value=avg_days_to_sql,
                                    title={"text": "Average Days from MQL to SQL"},
                                    domain={"y": [0.5, 1], "x": [0, 1]},
                                )
                            )
                            fig.update_layout(title="MQL to SQL Conversion Time")
                            st.plotly_chart(fig)
                        else:
                            st.warning("MQL to SQL conversion time information is not available in the dataset.")

                    st.subheader("MQL Data Table")
                    st.dataframe(df)
                    st.download_button(
                        label="Download MQL CSV",
                        data=df.to_csv(index=False),
                        file_name="mql_export.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()
