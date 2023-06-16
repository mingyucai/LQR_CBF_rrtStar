import numpy as np
def read_waypoints(path_to_waypoints_file):
    waypoints = np.load(path_to_waypoints_file, allow_pickle=True)
    return waypoints