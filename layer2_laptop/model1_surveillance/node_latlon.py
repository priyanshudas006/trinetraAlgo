"""Pixel-to-GPS mapper for grid nodes."""

from __future__ import annotations

import math
from typing import Dict


class NodeLatLon:
    """Assigns latitude/longitude to grid nodes from drone pose and camera FOV."""

    def __init__(
        self,
        drone_lat: float,
        drone_lon: float,
        altitude: float,
        fov_h: float = 90.0,
        fov_v: float = 90.0,
        grid_size: int = 20,
        image_w: int = 1000,
        image_h: int = 1000,
        yaw_deg: float = 0.0,
    ) -> None:
        self.drone_lat = drone_lat
        self.drone_lon = drone_lon
        self.altitude = max(altitude, 1.0)
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.grid_size = grid_size
        self.image_w = image_w
        self.image_h = image_h
        self.yaw_rad = math.radians(yaw_deg)

        footprint_w = 2.0 * self.altitude * math.tan(math.radians(self.fov_h / 2.0))
        footprint_h = 2.0 * self.altitude * math.tan(math.radians(self.fov_v / 2.0))
        self.m_per_px_x = footprint_w / float(self.image_w)
        self.m_per_px_y = footprint_h / float(self.image_h)

    def calculate(self, node: Dict[str, float]) -> None:
        center_px = node.get("center_px")
        if center_px is None:
            cell_w = self.image_w // self.grid_size
            cell_h = self.image_h // self.grid_size
            center_px = (node["col"] * cell_w + (cell_w // 2), node["row"] * cell_h + (cell_h // 2))

        dx_px = float(center_px[0]) - (self.image_w / 2.0)
        dy_px = float(center_px[1]) - (self.image_h / 2.0)

        east_m = dx_px * self.m_per_px_x
        north_m = -dy_px * self.m_per_px_y

        # Rotate offset by drone yaw to align image frame with world frame.
        world_east = east_m * math.cos(self.yaw_rad) - north_m * math.sin(self.yaw_rad)
        world_north = east_m * math.sin(self.yaw_rad) + north_m * math.cos(self.yaw_rad)

        node["lat"] = self.drone_lat + (world_north / 111111.0)
        node["lon"] = self.drone_lon + (world_east / (111111.0 * math.cos(math.radians(self.drone_lat))))