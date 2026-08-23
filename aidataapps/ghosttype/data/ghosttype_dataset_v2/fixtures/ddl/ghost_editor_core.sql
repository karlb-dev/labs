-- Candidate fixture DDL for ghost_editor_core.
-- The Lab 04 implementation must replace or verify this against the pinned SQL Server build before freeze.
-- Catalog source follows.
/*
-- connection: ghosttype / GhostEditorCore, default schema dbo, engine: SQL Server 2025 Developer
-- inferred system query: no
-- schema size medium: 7 tables, 1 view, 1 routine
-- schema budget: profile ghosttype-v2, columns verbose, max prompt chars 18000
TABLE dbo.Customers (CustomerId int NOT NULL PK, Name nvarchar(120) NOT NULL, RegionId int NULL FK->dbo.Regions.RegionId, CreatedAt datetime2 NOT NULL)
TABLE dbo.Regions (RegionId int NOT NULL PK, RegionName nvarchar(80) NOT NULL)
TABLE dbo.Orders (OrderId bigint NOT NULL PK, CustomerId int NOT NULL FK->dbo.Customers.CustomerId, OrderDate datetime2 NOT NULL, Status nvarchar(30) NOT NULL, TotalAmount decimal(18,2) NOT NULL)
TABLE dbo.OrderItems (OrderItemId bigint NOT NULL PK, OrderId bigint NOT NULL FK->dbo.Orders.OrderId, ProductId int NOT NULL FK->dbo.Products.ProductId, Quantity int NOT NULL, UnitPrice decimal(18,2) NOT NULL)
TABLE dbo.Products (ProductId int NOT NULL PK, ProductName nvarchar(160) NOT NULL, CategoryId int NOT NULL FK->dbo.Categories.CategoryId, IsActive bit NOT NULL)
TABLE dbo.Categories (CategoryId int NOT NULL PK, CategoryName nvarchar(100) NOT NULL)
TABLE dbo.Employees (EmployeeId int NOT NULL PK, ManagerId int NULL FK->dbo.Employees.EmployeeId, DisplayName nvarchar(120) NOT NULL, Department nvarchar(80) NOT NULL)
VIEW reporting.MonthlySales (MonthStart date, RegionName nvarchar(80), Revenue decimal(18,2), OrderCount bigint)
PROCEDURE dbo.usp_GetCustomerOrders (@CustomerId int, @FromDate date = NULL)
SYSTEM OBJECTS: sys.tables, sys.columns, sys.indexes, INFORMATION_SCHEMA.TABLES, INFORMATION_SCHEMA.COLUMNS
*/
