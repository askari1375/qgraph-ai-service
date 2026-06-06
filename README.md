# QGraph AI Service

AI backend for QGraph, responsible for:

- model orchestration
- inference pipelines
- analysis workflows

For production deployment, see [docs/deployment.md](docs/deployment.md).

Production deployment is intended to be private to the host:

- Public clients call the Django backend.
- Django calls this FastAPI service over `http://127.0.0.1:8001` on the VPS.
- `ai.qgraph.org` should not expose this service directly.

Production containers use `restart: unless-stopped`, so they restart after a host reboot or transient crash; manually stopped containers stay stopped until started again.
