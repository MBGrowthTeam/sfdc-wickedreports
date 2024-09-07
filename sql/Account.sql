"""
Fetches Account data, including various fields related to customer information.

This query retrieves a comprehensive set of fields from the Account object, 
including standard fields, custom fields, and fields related to various 
integrations and business processes.

Returns:
    A result set containing Account records with the following fields:

    - Id: Unique identifier of the Account.
    - IsDeleted: Indicates if the Account has been deleted.
    - MasterRecordId: ID of the master record for this Account (if merged).
    - Name: Account name.
    - Type: Account type (e.g., Prospect, Customer).
    - RecordTypeId: ID of the record type for this Account.
    - ParentId: ID of the parent Account (if any).
    - BillingStreet, BillingCity, BillingState, BillingPostalCode, BillingCountry, BillingStateCode, BillingCountryCode: Billing address details.
    - BillingLatitude, BillingLongitude, BillingGeocodeAccuracy: Billing address geolocation data.
    - BillingAddress: Full billing address as a compound field.
    - ShippingStreet, ShippingCity, ShippingState, ShippingPostalCode, ShippingCountry, ShippingStateCode, ShippingCountryCode: Shipping address details.
    - ShippingLatitude, ShippingLongitude, ShippingGeocodeAccuracy: Shipping address geolocation data.
    - ShippingAddress: Full shipping address as a compound field.
    - Phone: Account phone number.
    - Fax: Account fax number.
    - AccountNumber: Account number.
    - Website: Account website.
    - PhotoUrl: URL of the Account's photo.
    - Sic: Standard Industrial Classification (SIC) code.
    - Industry: Industry of the Account.
    - AnnualRevenue: Annual revenue of the Account.
    - NumberOfEmployees: Number of employees at the Account.
    - Ownership: Ownership type of the Account.
    - TickerSymbol: Stock ticker symbol for the Account.
    - Description: Description of the Account.
    - Rating: Account rating.
    - Site: Account site.
    - CurrencyIsoCode: Currency code for the Account.
    - OwnerId: ID of the Account owner (User).
    - CreatedDate: Date the Account was created.
    - CreatedById: ID of the User who created the Account.
    - LastModifiedDate: Date the Account was last modified.
    - LastModifiedById: ID of the User who last modified the Account.
    - SystemModstamp: Timestamp of the last system modification.
    - LastActivityDate: Date of the last activity related to the Account.
    - LastViewedDate: Date the Account was last viewed.
    - LastReferencedDate: Date the Account was last referenced.
    - IsExcludedFromRealign: Indicates if the Account is excluded from territory realignment.
    - IsCustomerPortal: Indicates if the Account has access to the customer portal.
    - Jigsaw: Data.com key for the Account.
    - JigsawCompanyId: Data.com Company ID for the Account.
    - AccountSource: Source of the Account.
    - SicDesc: Description of the SIC code.
    - OperatingHoursId: ID of the operating hours for the Account.
    - ActivityMetricId: ID of the activity metric for the Account.
    - SBQQ__AssetQuantitiesCombined__c: Combined quantity of assets for the Account (Salesforce CPQ).
    - SBQQ__CoTermedContractsCombined__c: Indicates if the Account has co-termed contracts (Salesforce CPQ).
    - SBQQ__CoTerminationEvent__c: Co-termination event for the Account (Salesforce CPQ).
    - SBQQ__ContractCoTermination__c: Indicates if the Account has contract co-termination (Salesforce CPQ).
    - SBQQ__DefaultOpportunity__c: ID of the default Opportunity for the Account (Salesforce CPQ).
    - SBQQ__IgnoreParentContractedPrices__c: Indicates if parent contracted prices should be ignored (Salesforce CPQ).
    - SBQQ__PreserveBundle__c: Indicates if bundle structure should be preserved (Salesforce CPQ).
    - SBQQ__PriceHoldEnd__c: End date for price hold (Salesforce CPQ).
    - SBQQ__RenewalModel__c: Renewal model for the Account (Salesforce CPQ).
    - SBQQ__RenewalPricingMethod__c: Renewal pricing method for the Account (Salesforce CPQ).
    - SBQQ__TaxExempt__c: Indicates if the Account is tax exempt (Salesforce CPQ).
    - AM_Status__c: Custom field for AM Status.
    - Billing_Account_ID__c: Custom field for Billing Account ID.
    - Account_Primary_Contact_Email__c: Email address of the primary contact for the Account.
    - Account_Primary_Contact_Phone__c: Phone number of the primary contact for the Account.
    - Dynamics_Account_ID__c: Custom field for Dynamics Account ID.
    - CloudingoAgent__BAR__c: Cloudingo Agent field for Billing Address Rollup.
    - Acct_Serial_No__c: Custom field for Account Serial Number.
    - CloudingoAgent__BAS__c: Cloudingo Agent field for Billing Address State.
    - CloudingoAgent__BAV__c: Cloudingo Agent field for Billing Address Validation.
    - Annual_Budget__c: Custom field for Annual Budget.
    - CloudingoAgent__BRDI__c: Cloudingo Agent field for Billing Rollup Duplicate Indicator.
    - CloudingoAgent__BTZ__c: Cloudingo Agent field for Billing Time Zone.
    - Arena__c: Custom field for Arena.
    - CloudingoAgent__SAR__c: Cloudingo Agent field for Shipping Address Rollup.
    - Basecamp_Project_Link__c: Custom field for Basecamp Project Link.
    - CloudingoAgent__SAS__c: Cloudingo Agent field for Shipping Address State.
    - Cancellation_Date__c: Custom field for Cancellation Date.
    - ChMS_ID__c: Custom field for ChMS ID.
    - CloudingoAgent__SAV__c: Cloudingo Agent field for Shipping Address Validation.
    - CloudingoAgent__SRDI__c: Cloudingo Agent field for Shipping Rollup Duplicate Indicator.
    - CloudingoAgent__STZ__c: Cloudingo Agent field for Shipping Time Zone.
    - EDW_Customer_Master_ID__c: Custom field for EDW Customer Master ID.
    - Current_Inactive_Date__c: Custom field for Current Inactive Date.
    - ChOO_Freshbooks_ID__c: Custom field for ChOO Freshbooks ID.
    - easyTithe_ID__c: Custom field for easyTithe ID.
    - e360_WHMCS_ID__c: Custom field for e360 WHMCS ID.
    - CustomerEffectiveDate__c: Custom field for Customer Effective Date.
    - CustomerId__c: Custom field for Customer ID.
    - Elexio_ID__c: Custom field for Elexio ID.
    - F1_WHMCS_ID__c: Custom field for F1 WHMCS ID.
    - Denomination__c: Denomination of the church or organization.
    - Diocesan_ID__c: Custom field for Diocesan ID.
    - Diocese__c: Diocese of the church.
    - EIN__c: Employer Identification Number (EIN).
    - Elexio_Financials_ID__c: Custom field for Elexio Financials ID.
    - ParishSOFT_Quickbooks_ID__c: Custom field for ParishSOFT Quickbooks ID.
    - F1_Church_Code__c: Custom field for F1 Church Code.
    - F1_Edition__c: Custom field for F1 Edition.
    - Shelby_Arena_Select_Customer__c: Custom field for Shelby Arena Select Customer.
    - FRS_ID__c: Custom field for FRS ID.
    - Give_Church_Link__c: Custom field for Give Church Link.
    - Giving_ID__c: Custom field for Giving ID.
    - ServiceU_ID__c: Custom field for ServiceU ID.
    - Shelby_V5_ID__c: Custom field for Shelby V5 ID.
    - SimpleChurch_SCCRM_ID__c: Custom field for SimpleChurch SCCRM ID.
    - Stripe_ID__c: Custom field for Stripe ID.
    - Merged_EDW_IDs__c: Custom field for Merged EDW IDs.
    - KeyedIn_Customer__c: Custom field for Keyed In Customer.
    - LastSupport_CSAT__c: Custom field for Last Support CSAT.
    - Check_Scans__c: Custom field for Check Scans.
    - Average_Weekly_Mailings__c: Custom field for Average Weekly Mailings.
    - Account_Entity_Type__c: Custom field for Account Entity Type.
    - Membership_Information_System__c: Custom field for Membership Information System.
    - F1_Risk__c: Custom field for F1 Risk.
    - MostRecent_CSAT_Score__c: Custom field for Most Recent CSAT Score.
    - Multi_Campus__c: Indicates if the Account is a multi-campus organization.
    - Shelby_HQ_Customer__c: Custom field for Shelby HQ Customer.
    - Merged_Customer_IDs__c: Custom field for Merged Customer IDs.
    - Org_ID__c: Custom field for Org ID.
    - PMM__c: Custom field for PMM.
    - PSG_ID__c: Custom field for PSG ID.
    - Account_Type__c: Custom field for Account Type.
    - Top_Account__c: Indicates if the Account is a top account.
    - Power_of_1__c: Custom field for Power of 1.
    - SalesLoft1__Active_Account__c: Indicates if the Account is active in SalesLoft.
    - SalesLoft1__SalesLoft_Domain__c: SalesLoft domain for the Account.
    - ChMS_Domain_search__c: Custom field for ChMS Domain Search.
    - Admin_Training__c: Custom field for Admin Training.
    - ChMS_Solution__c: Custom field for ChMS Solution.
    - Class_Management_for_Paid_Registrations__c: Custom field for Class Management for Paid Registrations.
    - Project_Stage_PM__c: Custom field for Project Stage PM.
    - Record_Count__c: Custom field for Record Count.
    - Redesign_Site_ID__c: Custom field for Redesign Site ID.
    - Community_Builder_Training__c: Custom field for Community Builder Training.
    - ChMS_Domain__c: Custom field for ChMS Domain.
    - Community_Events_for_Paid_Ticketing__c: Custom field for Community Events for Paid Ticketing.
    - SU_Org_ID__c: Custom field for SU Org ID.
    - Community_Giving__c: Custom field for Community Giving.
    - Community_Ministries__c: Custom field for Community Ministries.
    - Current_Publisher__c: Custom field for Current Publisher.
    - Donor_Growth_Strategist__c: Custom field for Donor Growth Strategist.
    - ShareFaith_ID__c: Custom field for ShareFaith ID.
    - Shelby_Financials_ID__c: Custom field for Shelby Financials ID.
    - Site_ID__c: Custom field for Site ID.
    - Status__c: Custom field for Account Status.
    - Engagement_Manager__c: Custom field for Engagement Manager.
    - Facility_Management__c: Custom field for Facility Management.
    - Sync_From_Desk__c: Custom field for Sync From Desk.
    - Sync_From_Hubspot__c: Custom field for Sync From Hubspot.
    - Feature_Comments__c: Custom field for Feature Comments.
    - TWA__c: Custom field for TWA.
    - Theme__c: Custom field for Theme.
    - Federal_Tax_ID__c: Custom field for Federal Tax ID.
    - Federal_Tax_Name__c: Custom field for Federal Tax Name.
    - Ministry_Scheduling__c: Custom field for Ministry Scheduling.
    - Other_ChMS_Solution__c: Custom field for Other ChMS Solution.
    - Submitted_W_9__c: Custom field for Submitted W-9.
    - WL_Giving_ID_CID__c: Custom field for WL Giving ID CID.
    - WS_WG_Customer_Type__c: Custom field for WS WG Customer Type.
    - WeGather_Database__c: Custom field for WeGather Database.
    - Website_Product__c: Custom field for Website Product.
    - e360_Status__c: Custom field for e360 Status.
    - e360_Type__c: Custom field for e360 Type.
    - ia_crm_Bill_to_Contact__c: Custom field for ia crm Bill to Contact.
    - WeGather_Pledging__c: Custom field for WeGather Pledging.
    - WeShare_Customer__c: Custom field for WeShare Customer.
    - WeShare_Organization_Type__c: Custom field for WeShare Organization Type.
    - e_Giving_Solution__c: Custom field for e-Giving Solution.
    - ia_crm_Sync_With_Intacct__c: Custom field for ia crm Sync With Intacct.
    - Accounting_Solution__c: Custom field for Accounting Solution.
    - Count_of_Users__c: Custom field for Count of Users.
    - Detailed_Denomination__c: Custom field for Detailed Denomination.
    - Est_Annual_Giving__c: Custom field for Estimated Annual Giving.
    - Generated_From_Leads__c: Custom field for Generated From Leads.
    - Households__c: Custom field for Households.
    - Primary_Denomination__c: Custom field for Primary Denomination.
    - Office_Hours__c: Custom field for Office Hours.
    - Time_Zone__c: Time zone of the Account.
    - Diocesan_Account_Executive__c: Custom field for Diocesan Account Executive.
    - Parish_Account_Executive__c: Custom field for Parish Account Executive.
    - Text_Messages__c: Custom field for Text Messages.
    - ChMS_Provider__c: Custom field for ChMS Provider.
    - Contact_Group__c: Custom field for Contact Group.
    - Last_Health_Review_Date__c: Custom field for Last Health Review Date.
    - NPS_Score__c: Custom field for NPS Score.
    - Primary_Health_Score__c: Custom field for Primary Health Score.
    - Reference__c: Custom field for Reference.
    - Secondary_Health_Score__c: Custom field for Secondary Health Score.
    - Last_Health_Attempt_Date__c: Custom field for Last Health Attempt Date.
    - Ethnicity__c: Custom field for Ethnicity.
    - Website_Provider__c: Custom field for Website Provider.
    - EUID__c: Custom field for EUID.
    - Merged_EUIDs__c: Custom field for Merged EUIDs.
    - maps__AssignmentRule__c: Custom field for maps Assignment Rule.
    - Client_Success_Rep__c: Custom field for Client Success Rep.
    - Asset_Maintenance_Date__c: Custom field for Asset Maintenance Date.
    - Maintenance_Date__c: Custom field for Maintenance Date.
    - Customer_Message__c: Custom field for Customer Message.
    - NeedtoMergeID__c: Custom field for Need to Merge ID.
    - Winner__c: Custom field for Winner.
    - Customer_Category__c: Custom field for Customer Category.
    - Customer_Lifecycle_Stage__c: Custom field for Customer Lifecycle Stage.
    - Customer_Point_of_Contact__c: Custom field for Customer Point of Contact.
    - Lead_Source__c: Custom field for Lead Source.
    - Whitespace_Opportunity__c: Custom field for Whitespace Opportunity.
    - Campuses__c: Custom field for Campuses.
    - Parish_or_Diocese__c: Custom field for Parish or Diocese.
    - Parishes__c: Custom field for Parishes.
    - Screenings_Per_Year__c: Custom field for Screenings Per Year.
    - Customer_Integrated_Date__c: Custom field for Customer Integrated Date.
    - Religious_Education_Enrollment__c: Custom field for Religious Education Enrollment.
    - Hillsite_ID__c: Custom field for Hillsite ID.
    - DD_AccountNumber__c: Custom field for DD Account Number.
    - DD_AccountStatus__c: Custom field for DD Account Status.
    - DD_AccountTypeDescription__c: Custom field for DD Account Type Description.
    - DD_AccountType__c: Custom field for DD Account Type.
    - DD_FamilyId__c: Custom field for DD Family ID.
    - DD_MinistryTypeDescription__c: Custom field for DD Ministry Type Description.
    - DD_MinistryType__c: Custom field for DD Ministry Type.
    - DD_System_AccountType__c: Custom field for DD System Account Type.
    - Workday_ID__c: Custom field for Workday ID.
    - Amplify_Account_Id__c: Custom field for Amplify Account ID.
    - Accounting__c: Custom field for Accounting.
    - Background_Check__c: Custom field for Background Check.
    - ChMS__c: Custom field for ChMS.
    - Church_Mobile_App__c: Custom field for Church Mobile App.
    - Contact_Mode__c: Custom field for Contact Mode.
    - Facility_Mgmt__c: Custom field for Facility Mgmt.
    - Financials_Accounting_Domain__c: Custom field for Financials Accounting Domain.
    - Financials_Billing_Acct_ID__c: Custom field for Financials Billing Account ID.
    - Mandate__c: Custom field for Mandate.
    - Mandate_End_Date__c: Custom field for Mandate End Date.
    - Mass_Messaging__c: Custom field for Mass Messaging.
    - Online_Giving__c: Custom field for Online Giving.
    - Other_Software_PL__c: Custom field for Other Software PL.
    - Payment_Processing_Annual_Revenue__c: Custom field for Payment Processing Annual Revenue.
    - Payment_Processing_Annual_Volume__c: Custom field for Payment Processing Annual Volume.
    - Software_Annual_Recurring_Revenue__c: Custom field for Software Annual Recurring Revenue.
    - Streaming_Service__c: Custom field for Streaming Service.
    - Volunteer_Mgmt__c: Custom field for Volunteer Mgmt.
    - Website_Billing_Acct_ID__c: Custom field for Website Billing Account ID.
    - Communication_Solution__c: Custom field for Communication Solution.
    - Website_ID__c: Custom field for Website ID.
    - Website_Provider_Picklist__c: Custom field for Website Provider Picklist.
    - Worship_Planning__c: Custom field for Worship Planning.
    - Total_MB_Annual_Revenue_Deprecated__c: Custom field for Total MB Annual Revenue (Deprecated).
    - Other_Software__c: Custom field for Other Software.
    - Current_Vendor__c: Custom field for Current Vendor.
    - Enterprise__c: Custom field for Enterprise.
    - Headquarters__c: Custom field for Headquarters.
    - Market_Type__c: Custom field for Market Type.
    - Number_of_Locations__c: Custom field for Number of Locations.
    - Number_of_Volunteers__c: Custom field for Number of Volunteers.
    - Protection_ACT_Customer_Id__c: Custom field for Protection ACT Customer ID.
    - Protection_Account__c: Custom field for Protection Account.
    - Financials_Products__c: Custom field for Financials Products.
    - Protection_Client_Account_Number__c: Custom field for Protection Client Account Number.
    - Protection_ID__c: Custom field for Protection ID.
    - Referred_By_Other__c: Custom field for Referred By Other.
    - Referred_By__c: Custom field for Referred By.
    - Toll_Free_Phone__c: Custom field for Toll Free Phone.
    - Volume_Category__c: Custom field for Volume Category.
    - Shipping_Address_Same_as_Billing_Address__c: Indicates if the shipping address is the same as the billing address.
    - DOZISF__ZoomInfo_Enrich_Status__c: ZoomInfo enrichment status for the Account.
    - DOZISF__ZoomInfo_First_Updated__c: Date and time of the first ZoomInfo update.
    - DOZISF__ZoomInfo_Id__c: ZoomInfo ID for the Account.
    - DOZISF__ZoomInfo_InboxAI_ID__c: ZoomInfo InboxAI ID for the Account.
    - DOZISF__ZoomInfo_Last_Updated__c: Date and time of the last ZoomInfo update.
    - ZoomInfo_Account_Name__c: ZoomInfo account name.
    - ZoomInfo_Audience__c: ZoomInfo audience for the Account.
    - ZoomInfo_City__c: ZoomInfo city for the Account.
    - ZoomInfo_Phone__c: ZoomInfo phone number for the Account.
    - ZoomInfo_Signal_Score__c: ZoomInfo signal score for the Account.
    - ZoomInfo_State__c: ZoomInfo state for the Account.
    - ZoomInfo_Street_Address__c: ZoomInfo street address for the Account.
    - ZoomInfo_Website__c: ZoomInfo website for the Account.
    - ZoomInfo_ZipPostal_Code__c: ZoomInfo zip/postal code for the Account.
    - Amplify_Purchase_Billing_Method__c: Custom field for Amplify Purchase Billing Method.
    - ChMS_ARR__c: Custom field for ChMS ARR.
    - ChMS_Billing_Account_ID__c: Custom field for ChMS Billing Account ID.
    - ChMS_Cancel_Date__c: Custom field for ChMS Cancel Date.
    - ChMS_Status__c: Custom field for ChMS Status.
    - Digital_Content_Annual_Revenue__c: Custom field for Digital Content Annual Revenue.
    - Digital_Content_Billing_Account_ID__c: Custom field for Digital Content Billing Account ID.
    - ChMS_Brand__c: Custom field for ChMS Brand.
    - Digital_Content_Cancel_Date__c: Custom field for Digital Content Cancel Date.
    - Digital_Content_ID__c: Custom field for Digital Content ID.
    - Digital_Content_Status__c: Custom field for Digital Content Status.
    - Financials_ARR__c: Custom field for Financials ARR.
    - Financials_Cancel_Date__c: Custom field for Financials Cancel Date.
    - Financials_Status__c: Custom field for Financials Status.
    - Giving_Cancel_Date__c: Custom field for Giving Cancel Date.
    - Giving_Status__c: Custom field for Giving Status.
    - Messaging_Annual_Revenue__c: Custom field for Messaging Annual Revenue.
    - Messaging_Billing_Account_ID__c: Custom field for Messaging Billing Account ID.
    - Top_25_Account__c: Indicates if the Account is a top 25 account.
    - Messaging_Cancel_Date__c: Custom field for Messaging Cancel Date.
    - Messaging_ID__c: Custom field for Messaging ID.
    - Messaging_Status__c: Custom field for Messaging Status.
    - Protection_Annual_Revenue__c: Custom field for Protection Annual Revenue.
    - Protection_Billing_Account_ID__c: Custom field for Protection Billing Account ID.
    - Protection_Cancel_Date__c: Custom field for Protection Cancel Date.
    - Protection_Status__c: Custom field for Protection Status.
    - Streaming_Annual_Revenue__c: Custom field for Streaming Annual Revenue.
    - Streaming_Billing_Account_ID__c: Custom field for Streaming Billing Account ID.
    - Integrated_with_Monday_com__c: Custom field for Integrated with Monday.com.
    - Streaming_Cancel_Date__c: Custom field for Streaming Cancel Date.
    - Streaming_ID__c: Custom field for Streaming ID.
    - Streaming_Status__c: Custom field for Streaming Status.
    - Website_ARR__c: Custom field for Website ARR.
    - Website_Cancel_Date__c: Custom field for Website Cancel Date.
    - Website_Status__c: Custom field for Website Status.
    - FaithDirect_ID__c: Custom field for FaithDirect ID.
    - Partner_Name_Company__c: Custom field for Partner Name Company.
    - ChMS_Last_Invoice_Date__c: Custom field for ChMS Last Invoice Date.
    - ChMS_First_Invoice_Date__c: Custom field for ChMS First Invoice Date.
    - ChMS_Move_to__c: Custom field for ChMS Move to.
    - Financials_First_Invoice_Date__c: Custom field for Financials First Invoice Date.
    - ChMS_Cancel_Reason__c: Custom field for ChMS Cancel Reason.
    - Financials_Last_Invoice_Date__c: Custom field for Financials Last Invoice Date.
    - Financials_Cancel_Reason__c: Custom field for Financials Cancel Reason.
    - Financials_Move_to__c: Custom field for Financials Move to.
    - Website_First_Invoice_Date__c: Custom field for Website First Invoice Date.
    - Website_Last_Invoice_Date__c: Custom field for Website Last Invoice Date.
    - Website_Cancel_Reason__c: Custom field for Website Cancel Reason.
    - Website_Move_to__c: Custom field for Website Move to.
    - Giving_First_Invoice_Date__c: Custom field for Giving First Invoice Date.
    - Giving_Last_Invoice_Date__c: Custom field for Giving Last Invoice Date.
    - Giving_Cancel_Reason__c: Custom field for Giving Cancel Reason.
    - Giving_Move_to__c: Custom field for Giving Move to.
    - Protection_First_Invoice_
"""
SELECT
    Id,
    IsDeleted,
    MasterRecordId,
    Name,
    Type,
    RecordTypeId,
    ParentId,
    BillingStreet,
    BillingCity,
    BillingState,
    BillingPostalCode,
    BillingCountry,
    BillingStateCode,
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
    ShippingStateCode,
    ShippingCountryCode,
    ShippingLatitude,
    ShippingLongitude,
    ShippingGeocodeAccuracy,
    ShippingAddress,
    Phone,
    Fax,
    AccountNumber,
    Website,
    PhotoUrl,
    Sic,
    Industry,
    AnnualRevenue,
    NumberOfEmployees,
    Ownership,
    TickerSymbol,
    Description,
    Rating,
    Site,
    CurrencyIsoCode,
    OwnerId,
    CreatedDate,
    CreatedById,
    LastModifiedDate,
    LastModifiedById,
    SystemModstamp,
    LastActivityDate,
    LastViewedDate,
    LastReferencedDate,
    IsExcludedFromRealign,
    IsCustomerPortal,
    Jigsaw,
    JigsawCompanyId,
    AccountSource,
    SicDesc,
    OperatingHoursId,
    ActivityMetricId,
    SBQQ__AssetQuantitiesCombined__c,
    SBQQ__CoTermedContractsCombined__c,
    SBQQ__CoTerminationEvent__c,
    SBQQ__ContractCoTermination__c,
    SBQQ__DefaultOpportunity__c,
    SBQQ__IgnoreParentContractedPrices__c,
    SBQQ__PreserveBundle__c,
    SBQQ__PriceHoldEnd__c,
    SBQQ__RenewalModel__c,
    SBQQ__RenewalPricingMethod__c,
    SBQQ__TaxExempt__c,
    AM_Status__c,
    Billing_Account_ID__c,
    Account_Primary_Contact_Email__c,
    Account_Primary_Contact_Phone__c,
    Dynamics_Account_ID__c,
    CloudingoAgent__BAR__c,
    Acct_Serial_No__c,
    CloudingoAgent__BAS__c,
    CloudingoAgent__BAV__c,
    Annual_Budget__c,
    CloudingoAgent__BRDI__c,
    CloudingoAgent__BTZ__c,
    Arena__c,
    CloudingoAgent__SAR__c,
    Basecamp_Project_Link__c,
    CloudingoAgent__SAS__c,
    Cancellation_Date__c,
    ChMS_ID__c,
    CloudingoAgent__SAV__c,
    CloudingoAgent__SRDI__c,
    CloudingoAgent__STZ__c,
    EDW_Customer_Master_ID__c,
    Current_Inactive_Date__c,
    ChOO_Freshbooks_ID__c,
    easyTithe_ID__c,
    e360_WHMCS_ID__c,
    CustomerEffectiveDate__c,
    CustomerId__c,
    Elexio_ID__c,
    F1_WHMCS_ID__c,
    Denomination__c,
    Diocesan_ID__c,
    Diocese__c,
    EIN__c,
    Elexio_Financials_ID__c,
    ParishSOFT_Quickbooks_ID__c,
    F1_Church_Code__c,
    F1_Edition__c,
    Shelby_Arena_Select_Customer__c,
    FRS_ID__c,
    Give_Church_Link__c,
    Giving_ID__c,
    ServiceU_ID__c,
    Shelby_V5_ID__c,
    SimpleChurch_SCCRM_ID__c,
    Stripe_ID__c,
    Merged_EDW_IDs__c,
    KeyedIn_Customer__c,
    LastSupport_CSAT__c,
    Check_Scans__c,
    Average_Weekly_Mailings__c,
    Account_Entity_Type__c,
    Membership_Information_System__c,
    F1_Risk__c,
    MostRecent_CSAT_Score__c,
    Multi_Campus__c,
    Shelby_HQ_Customer__c,
    Merged_Customer_IDs__c,
    Org_ID__c,
    PMM__c,
    PSG_ID__c,
    Account_Type__c,
    Top_Account__c,
    Power_of_1__c,
    SalesLoft1__Active_Account__c,
    SalesLoft1__SalesLoft_Domain__c,
    ChMS_Domain_search__c,
    Admin_Training__c,
    ChMS_Solution__c,
    Class_Management_for_Paid_Registrations__c,
    Project_Stage_PM__c,
    Record_Count__c,
    Redesign_Site_ID__c,
    Community_Builder_Training__c,
    ChMS_Domain__c,
    Community_Events_for_Paid_Ticketing__c,
    SU_Org_ID__c,
    Community_Giving__c,
    Community_Ministries__c,
    Current_Publisher__c,
    Donor_Growth_Strategist__c,
    ShareFaith_ID__c,
    Shelby_Financials_ID__c,
    Site_ID__c,
    Status__c,
    Engagement_Manager__c,
    Facility_Management__c,
    Sync_From_Desk__c,
    Sync_From_Hubspot__c,
    Feature_Comments__c,
    TWA__c,
    Theme__c,
    Federal_Tax_ID__c,
    Federal_Tax_Name__c,
    Ministry_Scheduling__c,
    Other_ChMS_Solution__c,
    Submitted_W_9__c,
    WL_Giving_ID_CID__c,
    WS_WG_Customer_Type__c,
    WeGather_Database__c,
    Website_Product__c,
    e360_Status__c,
    e360_Type__c,
    ia_crm_Bill_to_Contact__c,
    WeGather_Pledging__c,
    WeShare_Customer__c,
    WeShare_Organization_Type__c,
    e_Giving_Solution__c,
    ia_crm_Sync_With_Intacct__c,
    Accounting_Solution__c,
    Count_of_Users__c,
    Detailed_Denomination__c,
    Est_Annual_Giving__c,
    Generated_From_Leads__c,
    Households__c,
    Primary_Denomination__c,
    Office_Hours__c,
    Time_Zone__c,
    Diocesan_Account_Executive__c,
    Parish_Account_Executive__c,
    Text_Messages__c,
    ChMS_Provider__c,
    Contact_Group__c,
    Last_Health_Review_Date__c,
    NPS_Score__c,
    Primary_Health_Score__c,
    Reference__c,
    Secondary_Health_Score__c,
    Last_Health_Attempt_Date__c,
    Ethnicity__c,
    Website_Provider__c,
    EUID__c,
    Merged_EUIDs__c,
    maps__AssignmentRule__c,
    Client_Success_Rep__c,
    Asset_Maintenance_Date__c,
    Maintenance_Date__c,
    Customer_Message__c,
    NeedtoMergeID__c,
    Winner__c,
    Customer_Category__c,
    Customer_Lifecycle_Stage__c,
    Customer_Point_of_Contact__c,
    Lead_Source__c,
    Whitespace_Opportunity__c,
    Campuses__c,
    Parish_or_Diocese__c,
    Parishes__c,
    Screenings_Per_Year__c,
    Customer_Integrated_Date__c,
    Religious_Education_Enrollment__c,
    Hillsite_ID__c,
    DD_AccountNumber__c,
    DD_AccountStatus__c,
    DD_AccountTypeDescription__c,
    DD_AccountType__c,
    DD_FamilyId__c,
    DD_MinistryTypeDescription__c,
    DD_MinistryType__c,
    DD_System_AccountType__c,
    Workday_ID__c,
    Amplify_Account_Id__c,
    Accounting__c,
    Background_Check__c,
    ChMS__c,
    Church_Mobile_App__c,
    Contact_Mode__c,
    Facility_Mgmt__c,
    Financials_Accounting_Domain__c,
    Financials_Billing_Acct_ID__c,
    Mandate__c,
    Mandate_End_Date__c,
    Mass_Messaging__c,
    Online_Giving__c,
    Other_Software_PL__c,
    Payment_Processing_Annual_Revenue__c,
    Payment_Processing_Annual_Volume__c,
    Software_Annual_Recurring_Revenue__c,
    Streaming_Service__c,
    Volunteer_Mgmt__c,
    Website_Billing_Acct_ID__c,
    Communication_Solution__c,
    Website_ID__c,
    Website_Provider_Picklist__c,
    Worship_Planning__c,
    Total_MB_Annual_Revenue_Deprecated__c,
    Other_Software__c,
    Current_Vendor__c,
    Enterprise__c,
    Headquarters__c,
    Market_Type__c,
    Number_of_Locations__c,
    Number_of_Volunteers__c,
    Protection_ACT_Customer_Id__c,
    Protection_Account__c,
    Financials_Products__c,
    Protection_Client_Account_Number__c,
    Protection_ID__c,
    Referred_By_Other__c,
    Referred_By__c,
    Toll_Free_Phone__c,
    Volume_Category__c,
    Shipping_Address_Same_as_Billing_Address__c,
    DOZISF__ZoomInfo_Enrich_Status__c,
    DOZISF__ZoomInfo_First_Updated__c,
    DOZISF__ZoomInfo_Id__c,
    DOZISF__ZoomInfo_InboxAI_ID__c,
    DOZISF__ZoomInfo_Last_Updated__c,
    ZoomInfo_Account_Name__c,
    ZoomInfo_Audience__c,
    ZoomInfo_City__c,
    ZoomInfo_Phone__c,
    ZoomInfo_Signal_Score__c,
    ZoomInfo_State__c,
    ZoomInfo_Street_Address__c,
    ZoomInfo_Website__c,
    ZoomInfo_ZipPostal_Code__c,
    Amplify_Purchase_Billing_Method__c,
    ChMS_ARR__c,
    ChMS_Billing_Account_ID__c,
    ChMS_Cancel_Date__c,
    ChMS_Status__c,
    Digital_Content_Annual_Revenue__c,
    Digital_Content_Billing_Account_ID__c,
    ChMS_Brand__c,
    Digital_Content_Cancel_Date__c,
    Digital_Content_ID__c,
    Digital_Content_Status__c,
    Financials_ARR__c,
    Financials_Cancel_Date__c,
    Financials_Status__c,
    Giving_Cancel_Date__c,
    Giving_Status__c,
    Messaging_Annual_Revenue__c,
    Messaging_Billing_Account_ID__c,
    Top_25_Account__c,
    Messaging_Cancel_Date__c,
    Messaging_ID__c,
    Messaging_Status__c,
    Protection_Annual_Revenue__c,
    Protection_Billing_Account_ID__c,
    Protection_Cancel_Date__c,
    Protection_Status__c,
    Streaming_Annual_Revenue__c,
    Streaming_Billing_Account_ID__c,
    Integrated_with_Monday_com__c,
    Streaming_Cancel_Date__c,
    Streaming_ID__c,
    Streaming_Status__c,
    Website_ARR__c,
    Website_Cancel_Date__c,
    Website_Status__c,
    FaithDirect_ID__c,
    Partner_Name_Company__c,
    ChMS_Last_Invoice_Date__c,
    ChMS_First_Invoice_Date__c,
    ChMS_Move_to__c,
    Financials_First_Invoice_Date__c,
    ChMS_Cancel_Reason__c,
    Financials_Last_Invoice_Date__c,
    Financials_Cancel_Reason__c,
    Financials_Move_to__c,
    Website_First_Invoice_Date__c,
    Website_Last_Invoice_Date__c,
    Website_Cancel_Reason__c,
    Website_Move_to__c,
    Giving_First_Invoice_Date__c,
    Giving_Last_Invoice_Date__c,
    Giving_Cancel_Reason__c,
    Giving_Move_to__c
FROM
    Account