"""
Fetches subscription data from Salesforce within a specified date range.

Parameters:
    start_date (str): Start date for the data range in the format 'YYYY-MM-DDT00:00:00Z'.
    end_date (str): End date for the data range in the format 'YYYY-MM-DDT23:59:59Z'.

Returns:
    str: The SOQL query string to fetch subscription data.
"""
SELECT
    Id,
    SBQQ__Account__c, 
    SBQQ__StartDate__c,
    SBQQ__EndDate__c, 
    SBQQ__Product__r.Name,
    SBQQ__Product__r.Family,
    SBQQ__Product__r.Product_Category__c,
    SBQQ__RegularPrice__c,
    SBQQ__NetPrice__c,
    SBQQ__Contract__c  
FROM SBQQ__Subscription__c 
WHERE SBQQ__StartDate__c >= {start_date}
AND SBQQ__StartDate__c <= {end_date}