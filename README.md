# QGraph AI Service

AI backend for QGraph, responsible for:

- model orchestration
- inference pipelines
- analysis workflows

For production deployment, see [docs/deployment.md](docs/deployment.md).

Production containers use `restart: unless-stopped`, so they restart after a host reboot or transient crash; manually stopped containers stay stopped until started again.
