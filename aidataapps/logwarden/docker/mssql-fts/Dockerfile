ARG SQLSERVER_BASE_IMAGE=mcr.microsoft.com/mssql/server@sha256:4bab24f36c1ecd48e85f7d37df26e6bf301641d84c3fe652f9a0dcc947d512e1
FROM ${SQLSERVER_BASE_IMAGE}

ARG MSSQL_FTS_PACKAGE_VERSION=17.0.4075.5-1

USER root

# The SQL Server 2025 base image has the Microsoft key but only the general
# product repository. Register the official engine repository and pin FTS to
# the engine build encoded by the base-image digest.
RUN wget -qO /etc/apt/sources.list.d/mssql-server-2025.list \
      https://packages.microsoft.com/config/ubuntu/24.04/mssql-server-2025.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      "mssql-server-fts=${MSSQL_FTS_PACKAGE_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

USER mssql
