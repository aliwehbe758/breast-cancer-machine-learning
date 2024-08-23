BoltTorqueInspectionRequest
BoltTorqueInspectionRequestDate
repairReportNo
repairStatus
repairDate
defectType
repairLocation
finalBoxUp
finalBoxUpDate
hammerTestNumber
hammerTestStatus
hammerTestDate
hammerTestPercentage;

TorqueWrench

SET DATESTYLE = 'ISO,DMY';
COPY app_user FROM 'C:/cccdev/apps/talisman/talismanserver/src/main/resources/config/liquibase/data/import/test_data/app_user.csv' CSV ENCODING 'windows-1251' HEADER;
COPY app_user_role FROM 'C:/cccdev/apps/talisman/talismanserver/src/main/resources/config/liquibase/data/import/test_data/app_user_role.csv' CSV ENCODING 'windows-1251' HEADER;
truncate table storage_service_parameter;
COPY storage_service_parameter FROM 'C:/cccdev/apps/talisman/talismanserver/src/main/resources/config/liquibase/data/import/test_data/storage_service_parameter.csv' CSV ENCODING 'windows-1251' HEADER;
truncate table storage_service;
COPY storage_service FROM 'C:/cccdev/apps/talisman/talismanserver/src/main/resources/config/liquibase/data/import/test_data/storage_service.csv' CSV ENCODING 'windows-1251' HEADER;
