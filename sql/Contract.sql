"""
Fetches Contract data from Salesforce within a specified date range.

This query retrieves a comprehensive set of fields from the Contract object, 
including standard fields, custom fields, and fields related to Salesforce CPQ (SteelBrick). 
It filters the results based on the CreatedDate, allowing you to retrieve contracts created 
within a specific time period.

Parameters:
    {start_date}: The start date for the data range (inclusive).
    {end_date}: The end date for the data range (inclusive).

Returns:
    A result set containing Contract records with the following fields:

    - Id: Unique identifier of the Contract.
    - AccountId: ID of the related Account.
    - CurrencyIsoCode: Currency code for the Contract.
    - Pricebook2Id: ID of the Pricebook associated with the Contract.
    - OwnerExpirationNotice: Number of days before the Contract expires that the owner is notified.
    - StartDate: Contract start date.
    - EndDate: Contract end date.
    - BillingStreet, BillingCity, BillingState, BillingPostalCode, BillingCountry, BillingCountryCode: Billing address details.
    - BillingLatitude, BillingLongitude, BillingGeocodeAccuracy: Billing address geolocation data.
    - BillingAddress: Full billing address as a compound field.
    - ShippingStreet, ShippingCity, ShippingState, ShippingPostalCode, ShippingCountry, ShippingCountryCode: Shipping address details.
    - ShippingLatitude, ShippingLongitude, ShippingGeocodeAccuracy: Shipping address geolocation data.
    - ShippingAddress: Full shipping address as a compound field.
    - ContractTerm: Contract term in months.
    - OwnerId: ID of the Contract owner (User).
    - Status: Contract status (e.g., Draft, Activated).
    - CompanySignedId: ID of the Contact who signed on behalf of the company.
    - CompanySignedDate: Date the Contract was signed by the company.
    - CustomerSignedId: ID of the Contact who signed on behalf of the customer.
    - CustomerSignedTitle: Title of the customer signatory.
    - CustomerSignedDate: Date the Contract was signed by the customer.
    - SpecialTerms: Special terms of the Contract.
    - ActivatedById: ID of the User who activated the Contract.
    - ActivatedDate: Date the Contract was activated.
    - StatusCode: Status code of the Contract.
    - Description: Description of the Contract.
    - IsDeleted: Indicates if the Contract has been deleted.
    - ContractNumber: Contract number.
    - LastApprovedDate: Date the Contract was last approved.
    - CreatedDate: Date the Contract was created.
    - CreatedById: ID of the User who created the Contract.
    - LastModifiedDate: Date the Contract was last modified.
    - LastModifiedById: ID of the User who last modified the Contract.
    - SystemModstamp: Timestamp of the last system modification.
    - LastActivityDate: Date of the last activity related to the Contract.
    - LastViewedDate: Date the Contract was last viewed.
    - LastReferencedDate: Date the Contract was last referenced.
    - SBQQ__ActiveContract__c: Indicates if the Contract is active (Salesforce CPQ).
    - SBQQ__AmendmentOpportunityRecordTypeId__c: Record Type ID for Amendment Opportunities (Salesforce CPQ).
    - SBQQ__AmendmentOpportunityStage__c: Stage of Amendment Opportunities (Salesforce CPQ).
    - SBQQ__AmendmentOwner__c: Owner of Amendment Opportunities (Salesforce CPQ).
    - SBQQ__AmendmentPricebookId__c: Pricebook ID for Amendments (Salesforce CPQ).
    - SBQQ__AmendmentRenewalBehavior__c: Renewal behavior for Amendments (Salesforce CPQ).
    - SBQQ__AmendmentStartDate__c: Start date for Amendments (Salesforce CPQ).
    - SBQQ__DefaultRenewalContactRoles__c: Default Contact Roles for Renewals (Salesforce CPQ).
    - SBQQ__DefaultRenewalPartners__c: Default Partners for Renewals (Salesforce CPQ).
    - SBQQ__DisableAmendmentCoTerm__c: Indicates if Amendment Co-Terming is disabled (Salesforce CPQ).
    - SBQQ__ExpirationDate__c: Contract expiration date (Salesforce CPQ).
    - SBQQ__MDQRenewalBehavior__c: Renewal behavior for Multi-Dimensional Quoting (Salesforce CPQ).
    - SBQQ__MasterContract__c: ID of the Master Contract (Salesforce CPQ).
    - SBQQ__OpportunityPricebookId__c: Pricebook ID for the related Opportunity (Salesforce CPQ).
    - SBQQ__Opportunity__c: ID of the related Opportunity (Salesforce CPQ).
    - SBQQ__Order__c: ID of the related Order (Salesforce CPQ).
    - SBQQ__PreserveBundleStructureUponRenewals__c: Indicates if Bundle structure is preserved upon renewal (Salesforce CPQ).
    - SBQQ__Quote__c: ID of the related Quote (Salesforce CPQ).
    - SBQQ__RenewalForecast__c: Renewal forecast amount (Salesforce CPQ).
    - SBQQ__RenewalOpportunityRecordTypeId__c: Record Type ID for Renewal Opportunities (Salesforce CPQ).
    - SBQQ__RenewalOpportunityStage__c: Stage of Renewal Opportunities (Salesforce CPQ).
    - SBQQ__RenewalOpportunity__c: ID of the related Renewal Opportunity (Salesforce CPQ).
    - SBQQ__RenewalOwner__c: Owner of Renewal Opportunities (Salesforce CPQ).
    - SBQQ__RenewalPricebookId__c: Pricebook ID for Renewals (Salesforce CPQ).
    - SBQQ__RenewalQuoted__c: Indicates if the Renewal has been quoted (Salesforce CPQ).
    - SBQQ__RenewalTerm__c: Renewal term in months (Salesforce CPQ).
    - SBQQ__RenewalUpliftRate__c: Renewal uplift rate (Salesforce CPQ).
    - SBQQ__SubscriptionQuantitiesCombined__c: Combined quantity of subscriptions (Salesforce CPQ).
    - ATG_Cohort_ID__c: Custom field for ATG Cohort ID.
    - ATG_Contract_External_ID__c: Custom field for ATG Contract External ID.
    - ATG_Migrated_Record__c: Indicates if the Contract has been migrated.
    - Brand__c: Custom field for Brand.
    - Contract_Terminated__c: Indicates if the Contract has been terminated.
    - Contract_Type__c: Custom field for Contract Type.
    - Evergreen_Auto_Renewal__c: Indicates if the Contract has Evergreen Auto Renewal.
    - Intra_term_Auto_Renewal__c: Indicates if the Contract has Intra-Term Auto Renewal.
    - Primary_Contact_Email__c: Email address of the primary contact.
    - Renewal_Date__c: Custom field for Renewal Date.
    - Touched_Renewal__c: Indicates if the Renewal has been touched.
    - atg_Amendment_Contract_Value__c: Custom field for Amendment Contract Value.
    - atg_Amendment_Non_Recurring_Revenue__c: Custom field for Amendment Non-Recurring Revenue.
    - atg_Amendment_Recurring_Revenue__c: Custom field for Amendment Recurring Revenue.
    - atg_Original_Contract_Value__c: Custom field for Original Contract Value.
    - atg_Original_Non_Recurring_Revenue__c: Custom field for Original Non-Recurring Revenue.
    - atg_Original_Recurring_Revenue__c: Custom field for Original Recurring Revenue.
    - atg_TCV__c: Custom field for Total Contract Value.
    - atg_Total_Non_Recurring_Revenue__c: Custom field for Total Non-Recurring Revenue.
    - atg_Total_Recurring_Revenue__c: Custom field for Total Recurring Revenue.
    - easyTithe_ID__c: Custom field for easyTithe ID.
    - Primary_Contact__c: ID of the primary contact.
"""
SELECT 
    Id,
    AccountId,
    CurrencyIsoCode,
    Pricebook2Id,
    OwnerExpirationNotice,
    StartDate,
    EndDate,
    BillingStreet,
    BillingCity,
    BillingState,
    BillingPostalCode,
    BillingCountry,
    BillingCountryCode,
    BillingLatitude,
    BillingLongitude,
    BillingGeocodeAccuracy,
    BillingAddress,
    ShippingStreet,
    ShippingCity,
    ShippingState,
    ShippingPostalCode,
    ShippingCountry,
    ShippingCountryCode,
    ShippingLatitude,
    ShippingLongitude,
    ShippingGeocodeAccuracy,
    ShippingAddress,
    ContractTerm,
    OwnerId,
    Status,
    CompanySignedId,
    CompanySignedDate,
    CustomerSignedId,
    CustomerSignedTitle,
    CustomerSignedDate,
    SpecialTerms,
    ActivatedById,
    ActivatedDate,
    StatusCode,
    Description,
    IsDeleted,
    ContractNumber,
    LastApprovedDate,
    CreatedDate,
    CreatedByid,
    LastModifiedDate,
    LastModifiedById,
    SystemModstamp,
    LastActivityDate,
    LastViewedDate,
    LastReferencedDate,
    SBQQ__ActiveContract__c,
    SBQQ__AmendmentOpportunityRecordTypeId__c,
    SBQQ__AmendmentOpportunityStage__c,
    SBQQ__AmendmentOwner__c,
    SBQQ__AmendmentPricebookId__c,
    SBQQ__AmendmentRenewalBehavior__c,
    SBQQ__AmendmentStartDate__c,
    SBQQ__DefaultRenewalContactRoles__c,
    SBQQ__DefaultRenewalPartners__c,
    SBQQ__DisableAmendmentCoTerm__c,
    SBQQ__ExpirationDate__c,
    SBQQ__MDQRenewalBehavior__c,
    SBQQ__MasterContract__c,
    SBQQ__OpportunityPricebookId__c,
    SBQQ__Opportunity__c,
    SBQQ__Order__c,
    SBQQ__PreserveBundleStructureUponRenewals__c,
    SBQQ__Quote__c,
    SBQQ__RenewalForecast__c,
    SBQQ__RenewalOpportunityRecordTypeId__c,
    SBQQ__RenewalOpportunityStage__c,
    SBQQ__RenewalOpportunity__c,
    SBQQ__RenewalOwner__c,
    SBQQ__RenewalPricebookId__c,
    SBQQ__RenewalQuoted__c,
    SBQQ__RenewalTerm__c,
    SBQQ__RenewalUpliftRate__c,
    SBQQ__SubscriptionQuantitiesCombined__c,
    ATG_Cohort_ID__c,
    ATG_Contract_External_ID__c,
    ATG_Migrated_Record__c,
    Brand__c,
    Contract_Terminated__c,
    Contract_Type__c,
    Evergreen_Auto_Renewal__c,
    Intra_term_Auto_Renewal__c,
    Primary_Contact_Email__c,
    Renewal_Date__c,
    Touched_Renewal__c,
    atg_Amendment_Contract_Value__c,
    atg_Amendment_Non_Recurring_Revenue__c,
    atg_Amendment_Recurring_Revenue__c,
    atg_Original_Contract_Value__c,
    atg_Original_Non_Recurring_Revenue__c,
    atg_Original_Recurring_Revenue__c,
    atg_TCV__c,
    atg_Total_Non_Recurring_Revenue__c,
    atg_Total_Recurring_Revenue__c,
    easyTithe_ID__c,
    Primary_Contact__c
FROM 
    Contract
WHERE 
    WHERE CreatedDate >= {start_date} AND CreatedDate <= {end_date}
    AND (Brand__c = '{brand_ids}')