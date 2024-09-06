"""
Fetch MQL (Marketing Qualified Leads) data from Salesforce within a specified date range.

Parameters:
    start_date (str): Start date for the data range in the format 'YYYY-MM-DDT00:00:00Z'.
    end_date (str): End date for the data range in the format 'YYYY-MM-DDT23:59:59Z'.

Returns:
    str: The SOQL query string to fetch MQL data.
"""
SELECT 
    Id, OwnerId, IsDeleted, Name, CurrencyIsoCode, CreatedDate, CreatedById, 
    LastModifiedDate, LastModifiedById, SystemModstamp, LastActivityDate, 
    LastViewedDate, LastReferencedDate, Account__c, Address__c, Age__c, 
    Annual_Budget__c, Brand__c, Campaign__c, Company__c, Contact__c, 
    Description__c, E_Mail_Address__c, 
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
WHERE CreatedDate >= {start_date} 
AND CreatedDate <= {end_date}