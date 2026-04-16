import json
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force deterministic simulation mode for automated flow validation.
os.environ["DRONE_SIMULATION"] = "true"
os.environ["ROVER_SIMULATION"] = "true"

results = {
    "module_tests": {},
    "connection_tests": {},
    "surveillance_flow": [],
    "target_flow": [],
    "runtime_errors": [],
    "critical_breakpoints": [],
    "notes": []
}

def record_error(scope, exc):
    results["runtime_errors"].append({
        "scope": scope,
        "error": str(exc),
        "trace": traceback.format_exc(limit=2)
    })

try:
    from TRINETRA.layer1_drone.drone_stream import DroneStream
    from TRINETRA.layer2_laptop.model1_surveillance.terrain_detector import TerrainDetector
    from TRINETRA.layer2_laptop.model1_surveillance.grid_heuristics import GridHeuristics
    from TRINETRA.layer2_laptop.model1_surveillance.boundary_extractor import BoundaryExtractor
    from TRINETRA.layer2_laptop.model1_surveillance.node_latlon import NodeLatLon
    from TRINETRA.layer2_laptop.model1_surveillance.path_planner import PathPlanner
    from TRINETRA.layer2_laptop.model2_navigation.heading_calculator import HeadingCalculator
    from TRINETRA.layer2_laptop.model2_navigation.visual_lock import VisualLock
    from TRINETRA.layer2_laptop.model3_sensor.threshold_checker import ThresholdChecker
    from TRINETRA.layer2_laptop.model3_sensor.backend_poster import BackendPoster
    from TRINETRA.layer2_laptop.main import TrinetraSystem
    import numpy as np
    import cv2
except Exception as exc:
    record_error("import_phase", exc)
    print(json.dumps(results, indent=2))
    raise SystemExit(1)

# ---------- Module-level tests ----------
try:
    drone = DroneStream(simulation=True)
    snap = drone.capture_snapshot()
    ok = all(k in snap for k in ["lat","lon","altitude","pitch","roll","yaw","image"]) and snap["image"] is not None
    results["module_tests"]["drone_stream"] = {"status": "PASS" if ok else "FAIL", "details": {"keys": list(snap.keys())}}
except Exception as exc:
    record_error("drone_stream", exc)
    results["module_tests"]["drone_stream"] = {"status": "FAIL", "details": "exception"}

try:
    td = TerrainDetector()
    terrain = td.detect(snap["image"], pitch_deg=snap["pitch"], roll_deg=snap["roll"])
    ok = set(["open_mask","terrain_map"]).issubset(set(terrain.keys())) and terrain["open_mask"].shape[:2] == terrain["terrain_map"].shape[:2]
    results["module_tests"]["terrain_detector"] = {"status": "PASS" if ok else "FAIL", "details": {"keys": list(terrain.keys())}}
except Exception as exc:
    record_error("terrain_detector", exc)
    results["module_tests"]["terrain_detector"] = {"status": "FAIL", "details": "exception"}

try:
    gh = GridHeuristics(grid_size=20, obstacle_inflation_px=12)
    grid = gh.build(terrain["open_mask"])
    sample = grid[0][0]
    required = ["row","col","status","heuristic","lat","lon","center_px"]
    ok = len(grid)==20 and len(grid[0])==20 and all(k in sample for k in required)
    results["module_tests"]["grid_heuristics"] = {"status": "PASS" if ok else "FAIL", "details": {"sample_keys": list(sample.keys())}}
except Exception as exc:
    record_error("grid_heuristics", exc)
    results["module_tests"]["grid_heuristics"] = {"status": "FAIL", "details": "exception"}

try:
    mapper = NodeLatLon(drone_lat=snap["lat"], drone_lon=snap["lon"], altitude=snap["altitude"], grid_size=20, image_w=1000, image_h=1000, yaw_deg=snap["yaw"])
    mapper.calculate(grid[0][0])
    ok = isinstance(grid[0][0]["lat"], float) and isinstance(grid[0][0]["lon"], float)
    results["module_tests"]["node_latlon"] = {"status": "PASS" if ok else "FAIL", "details": {"lat": grid[0][0]["lat"], "lon": grid[0][0]["lon"]}}
except Exception as exc:
    record_error("node_latlon", exc)
    results["module_tests"]["node_latlon"] = {"status": "FAIL", "details": "exception"}

try:
    for row in grid:
        for node in row:
            if node["lat"] is None:
                mapper.calculate(node)
    be = BoundaryExtractor(grid_size=20)
    boundary = be.extract(grid)
    ordered = be.order_nodes(boundary)
    vis = be.visualize(grid, ordered)
    ok = isinstance(boundary, list) and isinstance(ordered, list) and vis is not None
    results["module_tests"]["boundary_extractor"] = {"status": "PASS" if ok else "FAIL", "details": {"boundary_count": len(boundary), "ordered_count": len(ordered), "vis_shape": tuple(vis.shape)}}
except Exception as exc:
    record_error("boundary_extractor", exc)
    results["module_tests"]["boundary_extractor"] = {"status": "FAIL", "details": "exception"}

try:
    planner = PathPlanner()
    planner.set_grid(grid, 20)
    traversable = [n for row in grid for n in row if n["status"] in ("SAFE","PARTIAL")]
    start = traversable[0]
    target = traversable[-1]
    path = planner.astar(start["row"], start["col"], target["row"], target["col"])
    ok = isinstance(path, list) and len(path) > 0 and all("lat" in p and "lon" in p for p in path)
    results["module_tests"]["path_planner"] = {"status": "PASS" if ok else "FAIL", "details": {"path_len": len(path)}}
except Exception as exc:
    record_error("path_planner", exc)
    results["module_tests"]["path_planner"] = {"status": "FAIL", "details": "exception"}

try:
    hc = HeadingCalculator()
    d = hc.haversine_distance(29.9, 78.1, 29.9001, 78.1001)
    b = hc.calculate_bearing(29.9, 78.1, 29.9001, 78.1001)
    e = hc.heading_error(20, b)
    cmd = hc.get_motion_command(e, d)
    ok = d > 0 and 0 <= b <= 360 and cmd in ("FORWARD","LEFT","RIGHT","STOP")
    results["module_tests"]["heading_calculator"] = {"status": "PASS" if ok else "FAIL", "details": {"distance": d, "bearing": b, "error": e, "cmd": cmd}}
except Exception as exc:
    record_error("heading_calculator", exc)
    results["module_tests"]["heading_calculator"] = {"status": "FAIL", "details": "exception"}

try:
    v = VisualLock(min_good_matches=8)
    img = np.full((220,220,3), 50, dtype=np.uint8)
    cv2.circle(img, (110,110), 40, (0,0,255), -1)
    cv2.line(img, (30,30), (190,190), (255,255,255), 3)
    v.set_target_images([img, img.copy()])
    frame = np.full((480,640,3), 80, dtype=np.uint8)
    frame[180:400, 220:440] = cv2.resize(img, (220,220))
    detected, confidence, details = v.detect(frame)
    ok = isinstance(detected, bool) and isinstance(confidence, float) and "offset_px" in details
    results["module_tests"]["visual_lock"] = {"status": "PASS" if ok else "FAIL", "details": {"detected": detected, "confidence": confidence, "offset": details.get("offset_px")}}
except Exception as exc:
    record_error("visual_lock", exc)
    results["module_tests"]["visual_lock"] = {"status": "FAIL", "details": "exception"}

try:
    tc = ThresholdChecker()
    red = tc.check(800, 200)
    yellow = tc.check(500, 200)
    green = tc.check(100, 100)
    enriched = tc.enrich_payload({"metal":500,"gas":100})
    ok = red=="RED" and yellow=="YELLOW" and green=="GREEN" and enriched.get("status") == "YELLOW"
    results["module_tests"]["threshold_checker"] = {"status": "PASS" if ok else "FAIL", "details": {"red": red, "yellow": yellow, "green": green}}
except Exception as exc:
    record_error("threshold_checker", exc)
    results["module_tests"]["threshold_checker"] = {"status": "FAIL", "details": "exception"}

try:
    bp = BackendPoster(url="http://127.0.0.1:3000/api/hazards", retries=1)
    invalid = bp.post({"lat":1,"lon":2,"status":"BAD"})
    valid = bp.post({"lat":1.0,"lon":2.0,"status":"RED"})
    # backend likely absent; valid should generally be False but method should not crash.
    ok = (invalid is False) and isinstance(valid, bool)
    results["module_tests"]["backend_poster"] = {"status": "PASS" if ok else "FAIL", "details": {"invalid_result": invalid, "valid_result": valid}}
except Exception as exc:
    record_error("backend_poster", exc)
    results["module_tests"]["backend_poster"] = {"status": "FAIL", "details": "exception"}

# ---------- Connection tests ----------
try:
    sys_obj = TrinetraSystem()
    ok_load, msg_load = sys_obj.load_drone_snapshot()
    c1 = ok_load and sys_obj.grid is not None and sys_obj.terrain_data is not None
    results["connection_tests"]["terrain_to_grid_pipeline"] = {"status": "VALID" if c1 else "BROKEN", "details": msg_load}

    bnodes = sys_obj.boundary_extractor.extract(sys_obj.grid)
    ord_nodes = sys_obj.boundary_extractor.order_nodes(bnodes)
    p = PathPlanner(); p.set_grid(sys_obj.grid, sys_obj.grid_size); p.build_path(ord_nodes)
    c2 = len(p.waypoints) > 0 if ord_nodes else True
    results["connection_tests"]["grid_to_boundary_to_planner"] = {"status": "VALID" if c2 else "BROKEN", "details": {"boundary": len(bnodes), "ordered": len(ord_nodes), "wps": len(p.waypoints)}}

    st = sys_obj.rover_api.get_state(); se = sys_obj.rover_api.get_sensor(); frame = sys_obj.rover_api.get_camera_frame()
    c3 = all(k in st for k in ["lat","lon","heading"]) and all(k in se for k in ["metal","gas","obstacle"]) and frame is not None
    results["connection_tests"]["navigation_to_rover_api"] = {"status": "VALID" if c3 else "BROKEN", "details": {"state": st, "sensor": se}}

    # UI->main action proxies (without launching Tk loop)
    checks = {
        "set_mode": hasattr(sys_obj, "set_mode"),
        "select_target_from_map": hasattr(sys_obj, "select_target_from_map"),
        "set_target_images": hasattr(sys_obj, "set_target_images"),
        "start_mission": hasattr(sys_obj, "start_mission"),
        "stop_mission": hasattr(sys_obj, "stop_mission"),
    }
    c4 = all(checks.values())
    results["connection_tests"]["ui_to_main_contract"] = {"status": "VALID" if c4 else "BROKEN", "details": checks}
except Exception as exc:
    record_error("connection_tests", exc)

# ---------- Surveillance mode flow simulation ----------
try:
    s = TrinetraSystem()
    s.set_mode("surveillance")
    ok, msg = s.load_drone_snapshot()
    results["surveillance_flow"].append({"step": 1, "name": "load_drone_snapshot", "ok": ok, "details": msg})

    boundary = s.boundary_extractor.extract(s.grid)
    ordered = s.boundary_extractor.order_nodes(boundary)
    results["surveillance_flow"].append({"step": 2, "name": "boundary_extract_order", "ok": len(ordered) > 0, "details": {"boundary": len(boundary), "ordered": len(ordered)}})

    ok_start, msg_start = s.start_mission()
    results["surveillance_flow"].append({"step": 3, "name": "start_mission", "ok": ok_start, "details": msg_start})

    time.sleep(3.5)
    nav_state = s._nav.state.value if s._nav else None
    alive = s._mission_thread.is_alive() if s._mission_thread else False
    results["surveillance_flow"].append(
        {
            "step": 4,
            "name": "runtime_status_after_3_5s",
            "ok": nav_state in ["PLANNING", "NAVIGATING", "AVOIDING_OBSTACLE", "COMPLETE", "VISION_LOCK"],
            "details": {"state": nav_state, "thread_alive": alive},
        }
    )

    stopped = s.stop_mission()
    results["surveillance_flow"].append({"step": 5, "name": "stop_mission", "ok": stopped[0], "details": stopped[1]})
except Exception as exc:
    record_error("surveillance_flow", exc)
    results["surveillance_flow"].append({"step": "EXCEPTION", "ok": False, "details": str(exc)})

# ---------- Target mode flow simulation ----------
try:
    t = TrinetraSystem()
    t.set_mode("target")
    ok, msg = t.load_drone_snapshot()
    results["target_flow"].append({"step": 1, "name": "load_drone_snapshot", "ok": ok, "details": msg})

    # Simulate UI target selection programmatically.
    traversable = [n for row in t.grid for n in row if n["status"] in ("SAFE","PARTIAL")]
    target = traversable[-1]
    t.selected_target = target
    results["target_flow"].append({"step": 2, "name": "target_selection", "ok": True, "details": {"row": target["row"], "col": target["col"]}})

    # Simulate multi-image upload.
    import tempfile, os
    img = np.full((140,140,3), 60, dtype=np.uint8)
    cv2.circle(img, (70,70), 35, (0,0,255), -1)
    cv2.line(img, (10,10), (130,130), (255,255,255), 3)
    tmp_paths = []
    for i in range(2):
        fd, pth = tempfile.mkstemp(suffix=f"_{i}.png")
        os.close(fd)
        cv2.imwrite(pth, img)
        tmp_paths.append(pth)
    ok_img, msg_img = t.set_target_images(tmp_paths)
    results["target_flow"].append({"step": 3, "name": "upload_target_images", "ok": ok_img, "details": msg_img})

    ok_start, msg_start = t.start_mission()
    results["target_flow"].append({"step": 4, "name": "start_mission", "ok": ok_start, "details": msg_start})

    # Let loop run and observe state transitions.
    seen = []
    t0 = time.time()
    while time.time() - t0 < 7.0:
        if t._nav is not None:
            st = t._nav.state.value
            if st not in seen:
                seen.append(st)
        time.sleep(0.25)

    nav = t._nav
    results["target_flow"].append({
        "step": 5,
        "name": "runtime_state_observation",
        "ok": nav is not None,
        "details": {
            "seen_states": seen,
            "thread_alive": t._mission_thread.is_alive() if t._mission_thread else False,
            "last_error": nav.last_error if nav else None,
            "waypoints": len(nav.planner.waypoints) if nav else None,
        }
    })

    # Optional behavior check: obstacle-triggered replanning may not happen in each short run.
    saw_replan = ("AVOIDING_OBSTACLE" in seen)
    results["target_flow"].append({
        "step": 6,
        "name": "obstacle_replan_signal",
        "ok": True,
        "details": {"seen_replanning": saw_replan, "optional_check": True}
    })
    if not saw_replan:
        results["notes"].append("Target flow did not encounter obstacle within observation window (optional).")

    # Optional behavior check: vision lock depends on proximity and may not trigger in short window.
    saw_vision = ("VISION_LOCK" in seen or "COMPLETE" in seen)
    results["target_flow"].append({
        "step": 7,
        "name": "vision_lock_activation",
        "ok": True,
        "details": {"seen_vision_or_complete": saw_vision, "optional_check": True}
    })
    if not saw_vision:
        results["notes"].append("Target flow did not reach vision-lock/completion within observation window (optional).")

    stop_ok, stop_msg = t.stop_mission()
    results["target_flow"].append({"step": 8, "name": "stop_mission", "ok": stop_ok, "details": stop_msg})

    for pth in tmp_paths:
        try:
            os.remove(pth)
        except Exception:
            pass
except Exception as exc:
    record_error("target_flow", exc)
    results["target_flow"].append({"step": "EXCEPTION", "ok": False, "details": str(exc)})

# ---------- Critical breakpoint detection ----------
for scope in ["surveillance_flow", "target_flow"]:
    for step in results[scope]:
        if isinstance(step, dict) and step.get("ok") is False:
            results["critical_breakpoints"].append({"scope": scope, "step": step.get("step"), "name": step.get("name"), "details": step.get("details")})

print(json.dumps(results, indent=2))
