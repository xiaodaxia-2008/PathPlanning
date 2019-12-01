"""
By Zhen Xiao, Nov 29, 2019
"""

import numpy as np
import heapq
from typing import Union, List, Tuple
from IPython import embed


def Dist(node1, node2):
    return np.linalg.norm(np.array(node1) - np.array(node2))


def AStartPath(field, start: Union[List[int], Tuple[int], np.ndarray], end: Union[List[int], Tuple[int], np.ndarray]) -> Union[None, np.ndarray]:
    """
    field: a 2d NxM grid, from (0, 0) to (N-1, M-1)
    start: start positon, eg. (1, 1)
    end: start positon, eg. (3, 4)
    """
    start = tuple(start)
    end = tuple(end)
    # distance from start to current plus current to end, current node, father node, list/tuple can be compared
    start_node = (Dist(start, end), 0.0, start, None)
    open_set = [start_node]  # waiting to be visited
    # key(node) : father_node(through which is shortest dist to start), dist_to_start_node
    visited_set = {}
    visited_history = []
    heapq.heapify(open_set)
    find_path = False
    while open_set and not find_path:
        _, dist_to_start, node, father_node = heapq.heappop(open_set)
        visited_set[node] = (father_node, dist_to_start)
        visited_history.append(field.GridToWorld(*node))
        if end == node:
            find_path = True
            break
        neighbours = field._NeighbourValidNoCollisionCells_(node)
        for child_node in neighbours:
            child_node = tuple(child_node)
            dist_child_node_to_start = Dist(child_node, node) + dist_to_start
            if child_node in visited_set:
                if visited_set[child_node][1] > dist_child_node_to_start:
                    visited_set[child_node] = (node, dist_child_node_to_start)
            else:
                # there are 2 different heuristic methods
                # use distance_current_node_to_end_node
                node_wait_visit = (Dist(child_node, end),
                                   dist_child_node_to_start, child_node, node)

                # use distance_current_node_to_end_node_plus_current_node_to_start_node
                # node_wait_visit = (Dist(child_node, end),
                #                    dist_child_node_to_start + dist_child_node_to_start, child_node, node)

                # if node_wait_visit is already in open_set and the new heuristic distance is longer, we shouldn't add it, 
                # but find a node in heapq is time consuming, so we don't remove it
                # the heapq will contain same nodes but different distances 
                heapq.heappush(open_set, node_wait_visit)

    if find_path:
        path = []
        node = end
        while node is not None:
            path.append(field.GridToWorld(*node))
            node = visited_set[node][0]
        return np.array(path), np.array(visited_history)
    else:
        return None, None


if __name__ == "__main__":
    from DistanceField import DistanceField, plt
    field = DistanceField(x_size=1.0, y_size=1.0, x_origin=0.0,
                          y_origin=0.0, resolution=0.1, max_distance=0.05)
    field.AddObstacleRectangle(np.array([0.5, 0.5]), np.array(
        [0.6, 0.6]), update_nearby_grd=False)
    path = np.array([[0.1, 0.1], [0.8, 0.8]])
    start = field.WorldToGrid(*path[0])
    end = field.WorldToGrid(*path[1])

    path_rst, history = AStartPath(field, start, end)
    if path_rst is not None:
        field.Display(path=path_rst, show_grid=True, history_data=history)

    # plt.ion()
    # line = plt.plot([0.0], [0.0])[0]
    # for i in range(len(history)+1):
    #     line.set_data(history[:i].T)
    #     plt.pause(0.1)
