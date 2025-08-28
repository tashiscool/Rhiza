# Deployment Guide

## Local Development

```bash
# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Start a local mesh node
python -c "
import asyncio
from mesh_core import create_complete_palm_slab

async def main():
    slab = create_complete_palm_slab('dev_node')
    await slab.initialize()
    print('Mesh node ready!')
    
asyncio.run(main())
"
```

## Production Deployment

```bash
# Install with all integrations
pip install .[all]

# Use production configuration
export MESH_CONFIG_PATH=/path/to/production_config.json

# Deploy mesh node
python scripts/deploy_mesh.py
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install .[all]

CMD ["python", "scripts/deploy_mesh.py"]
```

## Configuration

The Mesh uses JSON configuration files with environment variable overrides:

```json
{
    "node_id": "production_node_001",
    "privacy_level": "selective",
    "network": {
        "port": 8000,
        "discovery_enabled": true
    },
    "integrations": {
        "leon": true,
        "axiom": true,
        "sentient": true
    }
}
```
