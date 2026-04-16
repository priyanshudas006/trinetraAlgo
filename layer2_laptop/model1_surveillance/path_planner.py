"""Path planner supporting surveillance and target modes."""

from __future__ import annotations

import heapq
import json
import math
from typing import Dict, List, Optional, Tuple
try:
    from ..utils.debug import debug_log
except ImportError:
    try:
        from layer2_laptop.utils.debug import debug_log
    except ImportError:
        from utils.debug import debug_log

Grid = List[List[dict]]


class PathPlanner:
    def __init__(self) -> None:
        self.waypoints: List[dict] = []
        self.grid: Optional[Grid] = None
        self.grid_size: int = 20
        self._grid_version: int = 0
        self._cache: Dict[Tuple[int, int, int, int, int], List[dict]] = {}

    def set_grid(self, grid: Grid, grid_size: int = 20) -> None:
        self.grid = grid
        self.grid_size = grid_size
        self._grid_version += 1
        self._cache.clear()

    def build_path(self, boundary_nodes_with_latlon: List[dict]) -> List[dict]:
        self.waypoints = [self._as_waypoint(i, node) for i, node in enumerate(boundary_nodes_with_latlon)]
        self._log_selected_path()
        return self.waypoints

    def astar(self, start_row: int, start_col: int, target_row: int, target_col: int) -> List[dict]:
        if self.grid is None:
            return []

        cache_key = (self._grid_version, start_row, start_col, target_row, target_col)
        if cache_key in self._cache:
            self.waypoints = [wp.copy() for wp in self._cache[cache_key]]
            return self.waypoints

        if not self._is_traversable(start_row, start_col) or not self._is_traversable(target_row, target_col):
            return []

        def h_cost(r: int, c: int) -> float:
            return math.hypot(r - target_row, c - target_col)

        open_heap: List[Tuple[float, int, int]] = [(0.0, start_row, start_col)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {(start_row, start_col): 0.0}
        visited: set[Tuple[int, int]] = set()

        directions = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

        while open_heap:
            _, r, c = heapq.heappop(open_heap)
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if r == target_row and c == target_col:
                return self._build_from_chain(came_from, (start_row, start_col), (target_row, target_col), cache_key)

            for dr, dc, move_cost in directions:
                nr, nc = r + dr, c + dc
                if not self._is_traversable(nr, nc):
                    continue
                if (nr, nc) in visited:
                    continue

                node_penalty = float(self.grid[nr][nc].get("heuristic", 0.0))
                tentative = g_cost[(r, c)] + move_cost + node_penalty
                if tentative < g_cost.get((nr, nc), float("inf")):
                    g_cost[(nr, nc)] = tentative
                    came_from[(nr, nc)] = (r, c)
                    f_score = tentative + h_cost(nr, nc)
                    heapq.heappush(open_heap, (f_score, nr, nc))

        return []

    def replan(self, current_row: int, current_col: int, target_row: int, target_col: int) -> List[dict]:
        return self.astar(current_row, current_col, target_row, target_col)

    def nearest_node(self, lat: float, lon: float, traversable_only: bool = True) -> Optional[dict]:
        if self.grid is None:
            return None
        best = None
        best_d = float("inf")
        for row in self.grid:
            for node in row:
                if traversable_only and node["status"] not in ("SAFE", "PARTIAL"):
                    continue
                if node.get("lat") is None or node.get("lon") is None:
                    continue
                d = (node["lat"] - lat) ** 2 + (node["lon"] - lon) ** 2
                if d < best_d:
                    best_d = d
                    best = node
        return best

    def mark_blocked(self, row: int, col: int) -> None:
        if self.grid is None:
            return
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.grid[row][col]["status"] = "BLOCKED"
            self.grid[row][col]["heuristic"] = max(float(self.grid[row][col]["heuristic"]), 10.0)
            self._grid_version += 1
            self._cache.clear()

    def get_next_waypoint(self, current_index: int) -> tuple[Optional[dict], int]:
        for i in range(current_index, len(self.waypoints)):
            if not self.waypoints[i]["visited"]:
                return self.waypoints[i], i
        return None, -1

    def mark_visited(self, index: int, scan_result: Optional[str] = None) -> None:
        if 0 <= index < len(self.waypoints):
            self.waypoints[index]["visited"] = True
            self.waypoints[index]["scan_result"] = scan_result

    def export_path(self, filepath: str = "path_output.json") -> None:
        with open(filepath, "w", encoding="utf-8") as file_handle:
            json.dump(self.waypoints, file_handle, indent=2)

    def calculate_total_distance(self) -> float:
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += self._haversine(
                self.waypoints[i]["lat"],
                self.waypoints[i]["lon"],
                self.waypoints[i + 1]["lat"],
                self.waypoints[i + 1]["lon"],
            )
        return round(total, 2)

    def _build_from_chain(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        target: Tuple[int, int],
        cache_key: Tuple[int, int, int, int, int],
    ) -> List[dict]:
        chain = [target]
        node = target
        while node in came_from:
            node = came_from[node]
            chain.append(node)
        chain.reverse()
        if chain[0] != start:
            return []

        waypoints = [self._as_waypoint(i, self.grid[r][c]) for i, (r, c) in enumerate(chain)]
        self.waypoints = waypoints
        self._cache[cache_key] = [wp.copy() for wp in waypoints]
        self._log_selected_path()
        return waypoints

    def _log_selected_path(self) -> None:
        debug_log("PATH", f"Selected path nodes: {len(self.waypoints)}")
        for i, wp in enumerate(self.waypoints):
            debug_log(
                "PATH_NODE",
                (
                    f"{i} -> row={wp['row']} col={wp['col']} "
                    f"lat={float(wp['lat']):.6f} lon={float(wp['lon']):.6f} "
                    f"heuristic={float(wp['heuristic']):.3f} status={wp['status']}"
                ),
            )

    def _as_waypoint(self, idx: int, node: dict) -> dict:
        return {
            "id": idx,
            "row": node["row"],
            "col": node["col"],
            "lat": node["lat"],
            "lon": node["lon"],
            "heuristic": node["heuristic"],
            "status": node["status"],
            "visited": False,
            "scan_result": None,
        }

    def _is_traversable(self, row: int, col: int) -> bool:
        if self.grid is None:
            return False
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        return self.grid[row][col]["status"] in ("SAFE", "PARTIAL")

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371000.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return radius * c
