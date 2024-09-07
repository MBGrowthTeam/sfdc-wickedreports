"""
Fetch open opportunities data from Salesforce within a specified date range.

Parameters:
    start_date (str): Start date for the data range in the format 'YYYY-MM-DDT00:00:00Z'.
    end_date (str): End date for the data range in the format 'YYYY-MM-DDT23:59:59Z'.

Returns:
    str: The SOQL query string to fetch open opportunities data.
"""
SELECT Id, Name, Amount, StageName, CloseDate, LeadSource, CreatedDate
FROM Opportunity
WHERE IsClosed = FALSE 
AND CloseDate >= {start_date}
AND CloseDate <= {end_date}