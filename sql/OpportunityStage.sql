"""
Fetches Opportunity Stage data.

This query retrieves all fields from the OpportunityStage object, which 
represents the different stages in the sales process for opportunities.

Returns:
    str: The SOQL query string to fetch Opportunity Stage data.
"""
SELECT
    Id,
    MasterLabel,
    ApiName,
    IsActive,
    SortOrder,
    IsClosed,
    IsWon,
    ForecastCategory,
    ForecastCategoryName,
    DefaultProbability,
    Description,
    CreatedById,
    CreatedDate,
    LastModifiedById,
    LastModifiedDate,
    SystemModstamp
FROM
    OpportunityStage