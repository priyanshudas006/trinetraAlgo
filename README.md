# TRINETRA

TRINETRA is a three-layer autonomous robotics prototype:
- `layer1_drone`: one-shot drone image + metadata source
- `layer2_laptop`: AI planning + mission control dashboard
- `layer3_rover`: rover firmware placeholders (ESP32)
- `backend`: Node.js hazard ingestion API (`/api/hazards`)

## Implemented Capabilities
- Terrain segmentation (`open_mask`, `terrain_map`)
- 20x20 grid generation with obstacle inflation and heuristics
- Pixel-to-GPS mapping per grid node
- A* planning + dynamic replanning
- Runtime navigation control loop with state machine:
  - `IDLE`, `PLANNING`, `NAVIGATING`, `REPLANNING`, `VISION_LOCK`, `ACTUATING`, `COMPLETE`, `ERROR`
- Vision lock using ORB feature matching with confidence threshold
- Hazard classification and backend posting with retry + payload validation
- Rover and drone simulation contracts when hardware is unavailable

## JSON Contracts
- Rover state: `{ "lat": number, "lon": number, "heading": number }`
- Sensor data: `{ "metal": number, "gas": number, "obstacle": boolean }`
- Command payload: `{ "cmd": "FORWARD|LEFT|RIGHT|STOP" }`
- Hazard payload: `{ "lat": number, "lon": number, "status": "GREEN|YELLOW|RED", ... }`

## Run
1. Python deps:
   - `pip install -r requirements.txt`
2. Backend deps:
   - `cd backend && npm install`
3. Start backend:
   - `node backend/server.js`
4. Start laptop UI:
   - `python layer2_laptop/main.py`

## UI Workflow
1. `Load Drone Snapshot`
2. Select mode (`Target` or `Surveillance`)
3. For target mode:
   - `Select Target Node`
   - `Upload Target Images` (2-5)
4. `Start Mission`
5. Use `Stop Mission` for manual abort