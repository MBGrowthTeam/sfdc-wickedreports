"""
Fetches lead data from Salesforce within a specified date range.

Parameters:
    start_date (str): Start date for the data range in the format 'YYYY-MM-DDT00:00:00Z'.
    end_date (str): End date for the data range in the format 'YYYY-MM-DDT23:59:59Z'.

Returns:
    str: The SOQL query string to fetch Lead data.
"""
SELECT 
    Id, IsDeleted, Name, CreatedDate, LeadSource, Status, Industry, Rating, 
    AnnualRevenue, NumberOfEmployees, OwnerId, IsConverted, ConvertedDate, 
    Age__c, Brand__c, Lead_Assigned_Date__c, Lead_Conversion_Date__c, 
    Marketo_Lead_Score__c, HS_Owner__c, TWA__c, Hubspot_Notes__c, 
    MQL_Date__c, MQL_Record__c, Annual_Budget__c, Product_Interest__c, 
    SalesLoft1__Most_Recent_Cadence_Name__c, Latest_Source__c, 
    Latest_Source_Drill_Down_1__c, Latest_Source_Drill_Down_2__c 
FROM Lead
WHERE CreatedDate >= {start_date}
AND CreatedDate <= {end_date}