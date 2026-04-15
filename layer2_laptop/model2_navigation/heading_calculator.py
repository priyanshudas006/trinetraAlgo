# layer2_laptop/model2_navigation/heading_calculator.py

import math

class HeadingCalculator:
    def __init__(self):
        self.EARTH_RADIUS = 6371000

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians,
                                     [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return self.EARTH_RADIUS * c

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        lat1, lat2 = map(math.radians, [lat1, lat2])
        dlon = math.radians(lon2 - lon1)

        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1)*math.sin(lat2) -
             math.sin(lat1)*math.cos(lat2)*math.cos(dlon))

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def heading_error(self, current_heading, target_bearing):
        error = target_bearing - current_heading
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
        return error

    def get_motion_command(self, error, distance,
                           turn_threshold=10, stop_distance=1.0):

        if distance < stop_distance:
            return "STOP"

        if error > turn_threshold:
            return "RIGHT"
        elif error < -turn_threshold:
            return "LEFT"
        else:
            return "FORWARD"