"""
Fetches Order Item data from Salesforce within a specified date range, including
related Order information. This query leverages the relationship between OrderItem
and Order objects.

Parameters:
    start_date (str): Start date for the data range in the format 'YYYY-MM-DDT00:00:00Z'.
    end_date (str): End date for the data range in the format 'YYYY-MM-DDT23:59:59Z'.

Returns:
    str: The SOQL query string to fetch Order Item data with related Order details.
"""
SELECT 
    Order.Id,
    Order.Brand__c,
    Order.CreatedDate,
    OrderItem.SBQQ__Subscription__c, 
    OrderItem.Product2Id, 
    OrderItem.Product_Category__c,
    OrderItem.Product_Name__c
FROM OrderItem
WHERE Order.CreatedDate >= {start_date} AND Order.CreatedDate <= {end_date}