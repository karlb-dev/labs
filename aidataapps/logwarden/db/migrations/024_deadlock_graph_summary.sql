SET XACT_ABORT ON;
GO

-- Return the bounded semantic summary required by the replay tool contract.
-- Raw graph XML remains protected in ingest.raw_events and is never returned
-- through the agent principal.
CREATE OR ALTER PROCEDURE agent.usp_tool_get_deadlock_graph
  @max_rows int = 5
AS
BEGIN
  SET NOCOUNT ON;
  IF @max_rows NOT BETWEEN 1 AND 20 THROW 51714, 'max_rows must be between 1 and 20', 1;
  ;WITH source AS
  (
    SELECT timestamp_utc,CONVERT(xml,event_data) AS event_xml
    FROM sys.fn_xe_file_target_read_file(N'/var/opt/mssql/log/logwarden_capture*.xel',NULL,NULL,NULL)
    WHERE object_name=N'xml_deadlock_report'
  ), selected AS
  (
    SELECT TOP (@max_rows) timestamp_utc,event_xml,
      LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',CONVERT(nvarchar(max),event_xml)),2)) AS graph_sha256
    FROM source
    ORDER BY timestamp_utc DESC,graph_sha256
  )
  SELECT
    graph.timestamp_utc AS occurred_at_utc,
    graph.event_xml.value('(/event/data/value/deadlock/victim-list/victimProcess/@id)[1]','nvarchar(128)') AS victim_process_id,
    graph.event_xml.value('count((/event/data/value/deadlock/process-list/process))','int') AS process_count,
    graph.event_xml.value('count((/event/data/value/deadlock/resource-list/*))','int') AS resource_count,
    JSON_QUERY
    ((
      SELECT
        process.node.value('(@id)[1]','nvarchar(128)') AS process_id,
        TRY_CONVERT(int,process.node.value('(@spid)[1]','nvarchar(20)')) AS session_id,
        process.node.value('(@loginname)[1]','nvarchar(256)') AS login_name,
        process.node.value('(@clientapp)[1]','nvarchar(256)') AS client_app_name,
        process.node.value('(@isolationlevel)[1]','nvarchar(120)') AS isolation_level,
        process.node.value('(@waitresource)[1]','nvarchar(512)') AS wait_resource,
        process.node.value('(@status)[1]','nvarchar(80)') AS process_status,
        LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',
          process.node.value('(inputbuf/text())[1]','nvarchar(4000)')),2)) AS statement_sha256
      FROM graph.event_xml.nodes('/event/data/value/deadlock/process-list/process') AS process(node)
      ORDER BY process.node.value('(@id)[1]','nvarchar(128)')
      FOR JSON PATH
    )) AS processes_json,
    JSON_QUERY
    ((
      SELECT
        resource.node.value('local-name(.)','nvarchar(80)') AS resource_type,
        TRY_CONVERT(int,NULLIF(resource.node.value('(@dbid)[1]','nvarchar(20)'),N'')) AS database_id,
        LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',
          resource.node.value('(@objectname)[1]','nvarchar(512)')),2)) AS object_name_sha256,
        resource.node.value('count(owner-list/owner)','int') AS owner_count,
        resource.node.value('count(waiter-list/waiter)','int') AS waiter_count,
        LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',CONVERT(nvarchar(max),resource.node.query('.'))),2)) AS resource_sha256
      FROM graph.event_xml.nodes('/event/data/value/deadlock/resource-list/*') AS resource(node)
      ORDER BY resource.node.value('local-name(.)','nvarchar(80)'),
        LOWER(CONVERT(char(64),HASHBYTES('SHA2_256',CONVERT(nvarchar(max),resource.node.query('.'))),2))
      FOR JSON PATH
    )) AS resources_json,
    graph.graph_sha256
  FROM selected AS graph
  ORDER BY graph.timestamp_utc DESC,graph.graph_sha256;
END;
GO

IF NOT EXISTS
(
  SELECT 1 FROM sys.crypt_properties
  WHERE class_desc='OBJECT_OR_COLUMN' AND major_id=OBJECT_ID(N'agent.usp_tool_get_deadlock_graph')
    AND thumbprint=(SELECT thumbprint FROM sys.certificates WHERE name=N'LogWardenToolCertificate')
)
  ADD SIGNATURE TO OBJECT::agent.usp_tool_get_deadlock_graph BY CERTIFICATE [LogWardenToolCertificate];
