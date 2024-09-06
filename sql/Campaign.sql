"""
Fetch data from the Campaign object within the specified date range.

Parameters:
    start_date (str): Start date for the data range in the format 'YYYY-MM-DDT00:00:00Z'.
    end_date (str): End date for the data range in the format 'YYYY-MM-DDT23:59:59Z'.

Returns:
    str: The SOQL query string to fetch Campaign data.
"""
SELECT 
    Id, Name, CreatedDate
FROM Campaign
WHERE CreatedDate >= {start_date} 
AND CreatedDate <= {end_date}