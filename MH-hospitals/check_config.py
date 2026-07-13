import json
import os
from config.config import EMERGENCY_ENABLED

print(f"EMERGENCY_ENABLED from config.config: {EMERGENCY_ENABLED}")

with open('config/config.json', 'r') as f:
    config = json.load(f)
    print(f"Enabled from config/config.json: {config['emergencies']['enabled']}")
