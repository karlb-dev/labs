ARG SQLSERVER_BASE_IMAGE=mcr.microsoft.com/mssql/server@sha256:4bab24f36c1ecd48e85f7d37df26e6bf301641d84c3fe652f9a0dcc947d512e1
FROM ${SQLSERVER_BASE_IMAGE}

USER root

# SQL Server's base image already configures Microsoft's matching package
# repository. Keep the package transaction in one layer and fail the build if
# the engine-compatible full-text package cannot be installed.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends mssql-server-fts \
    && rm -rf /var/lib/apt/lists/*

USER mssql
