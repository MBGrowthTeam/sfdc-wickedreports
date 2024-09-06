"""
Fetches order data from Salesforce within a specified date range.

Parameters:
    start_date (str): Start date for the data range in the format 'YYYY-MM-DDT00:00:00Z'.
    end_date (str): End date for the data range in the format 'YYYY-MM-DDT23:59:59Z'.

Returns:
    str: The SOQL query string to fetch order data.
"""
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
WHERE Order.CreatedDate >= {start_date} 
AND Order.CreatedDate <= {end_date} 