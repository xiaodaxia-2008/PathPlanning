"""
By Zhen Xiao, Nov 27, 2019
"""
import logging
import math
import random
import time
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy import interpolate

from AStar import AStartPath

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Cell:
    def __init__(self, gradient=np.zeros((2,))):
        self.m_gradient: np.ndarray = gradient.copy()
        self.m_obstacle: bool = False

    def __repr__(self):
        return "<Cell at 0x{:x}, gradient {}, obstacle {}>".format(id(self), list(self.m_gradient), self.m_obstacle)


class Rectangle:
    def __init__(self, point0: np.ndarray, point1: np.ndarray):
        """Grid Rectangle 
        
        Arguments:
            point0 {np.ndarray[int, [2,]]} -- rectangle left down corner in grid coordinate
            point1 {np.ndarray[int, [2,]]} -- rectangle right up corner in grid coordinate
        """
        self.m_center = (0.5 * (point0 + point1)).astype(int)
        if np.all(point0 < self.m_center):
            self.m_corner0 = point0  # left-down corner
            self.m_corner1 = point1  # right-up corner
        else:
            self.m_corner0 = point1  # left-down corner
            self.m_corner1 = point0  # right-up corner
        self.m_extent = np.ceil(np.abs(point1 - self.m_center)).astype(int)
        self.m_volume: float = np.prod(self.m_extent) * \
            np.power(2, self.m_extent.shape[0])

    def __repr__(self):
        return "<Rectangle at 0x{:x}, center {}, extent {}>".format(id(self), self.m_center, self.m_extent)


class Circle:
    def __init__(self, center: np.ndarray, radius: int):
        """Grid Circle
        
        Arguments:
            center {np.ndarray[int, [2,]]} -- radius center in grid coordinate
            radius {int} -- circle radius in grid coordinate
        """
        self.m_center = center
        self.m_radius = radius
        self.m_volume: float = np.pi * self.m_radius * self.m_radius
        # if sphere
        if len(center) == 3:
            self.m_volume *= 4/3*self.m_radius

    def __repr__(self):
        return "<Circle at 0x{:x}, center {}, radius {}>".format(id(self), self.m_center, self.m_radius)


class DistanceField:
    """
    NOTE: 
    1) functions end with _ , return the grid coordinate, otherwise return world coordinate
    2) functions start with _ , require the grid coordinate, otherwise require world coordinate
    """

    def __init__(self, x_size: float = 1.0, y_size: float = 1.0, x_origin: float = 0.0,
                 y_origin: float = 0.0, resolution: float = 0.01, max_distance: float = 0.1):
        self.m_x_origin: float = x_origin
        self.m_y_origin: float = y_origin
        self.m_x_size: float = x_size
        self.m_y_size: float = y_size
        self.m_resolution: float = resolution
        self.m_resolution_half: float = 0.5 * self.m_resolution
        self.m_max_distance: float = max_distance
        self.m_max_distance_cells: int = math.ceil(
            self.m_max_distance / self.m_resolution)
        self.m_x_cell_num: int = math.ceil(self.m_x_size / self.m_resolution)
        self.m_y_cell_num: int = math.ceil(self.m_y_size / self.m_resolution)
        self.m_grid: List[List[Cell]] = [[Cell() for _ in range(self.m_y_cell_num)]
                                         for _ in range(self.m_x_cell_num)]
        self.m_obstacles = []
        self.m_clearance: int = 1  # distance in grid to nereast obstacle cell
        self.m_neighbour_dxy = None

    def GetCell(self, x: float, y: float) -> Cell:
        x_idx, y_idx = self.WorldToGrid(x, y)
        return self.m_grid[x_idx][y_idx]

    def GetNearbyObstacleGradient(self, x: float, y: float) -> np.ndarray:
        collision = self.IsCellInCollision(x, y)
        coeff = 1.0 if collision else 0.01
        return self.GetCell(x, y).m_gradient * coeff

    def GetObstacleCenterGradient(self, x: float, y: float) -> np.ndarray:
        grd = np.zeros((2))
        tc = self.WorldToGrid(x, y)
        for obs in self.m_obstacles:
            grd += self._Compute2PointsGradient(obs.m_center,
                                                tc) * obs.m_volume
        collision = self.IsCellInCollision(x, y)
        coeff = 1.0 if collision else 0.01
        return grd * coeff

    def RandomNoCollisionCellInRange(self, center: np.ndarray, radius: float) -> Union[None, np.ndarray]:
        gcenter = self.WorldToGrid(*center)
        gradius = math.ceil(radius / self.m_resolution)
        ncs = self._NeighbourCircleCells(gcenter, gradius)
        idx_set = set(range(ncs.shape[0]))
        while True:
            idx = random.sample(idx_set, 1)[0]
            idx_set.remove(idx)
            if not self._IsCellInCollision(*ncs[idx]):
                return ncs[idx]
            if not idx_set:
                return None

    def RandomNoCollisionCell(self) -> np.ndarray:
        return self.GridToWorld(*self._RandomNoCollisionCell_())

    def _RandomNoCollisionCell_(self, x_range: List[int] = None, y_range: List[int] = None) -> np.ndarray:
        if x_range is None:
            x_range = [0, self.m_x_cell_num-1]
        if y_range is None:
            y_range = [0, self.m_y_cell_num-1]
        collision = True
        while collision:
            gx = np.random.randint(*x_range)
            gy = np.random.randint(*y_range)
            collision = self._IsCellInCollision(gx, gy)
        return np.array([gx, gy])

    def GetGridBound(self):
        # x0, x1, y0, y1
        return self.m_x_origin - self.m_resolution_half, self.m_x_origin + self.m_x_size + self.m_resolution_half, \
            self.m_y_origin - self.m_resolution_half, self.m_y_origin + \
            self.m_resolution_half + self.m_y_size

    def IsCellInCollision(self, x: float, y: float):
        return self._IsCellInCollision(*self.WorldToGrid(x, y))

    def _IsCellInCollision(self, x_idx: int, y_idx: int):
        # return self.GetCell(x_idx, y_idx).m_obstacle
        tg = np.array([x_idx, y_idx])
        for obs in self.m_obstacles:
            if isinstance(obs, Rectangle):
                if np.all(np.abs(tg - obs.m_center) <= obs.m_extent + self.m_clearance):
                    return True
            elif isinstance(obs, Circle):
                if np.linalg.norm(tg - obs.m_center) <= self.m_clearance + obs.m_radius:
                    return True
        return False

    def _IsCellValid(self, x_idx: int, y_idx: int):
        return 0 <= x_idx < self.m_x_cell_num and 0 <= y_idx < self.m_y_cell_num

    def WorldToGrid(self, x: float, y: float) -> np.ndarray:
        x_idx, y_idx = math.ceil((x - self.m_x_origin + self.m_resolution_half) / self.m_resolution), math.ceil(
            (y - self.m_y_origin + self.m_resolution_half) / self.m_resolution)
        if self._IsCellValid(x_idx, y_idx):
            return np.array([x_idx, y_idx])
        else:
            logger.error(
                f"Out of grid range: {(x_idx, y_idx)}, current range {(self.m_x_cell_num, self.m_y_cell_num)}")
            return None

    def GridToWorld(self, x_idx: int, y_idx: int):
        if self._IsCellValid(x_idx, y_idx):
            return np.array([x_idx * self.m_resolution + self.m_x_origin, y_idx * self.m_resolution + self.m_y_origin])
        else:
            logger.error(
                f"Out of grid range: {(x_idx, y_idx)}, current range {(self.m_x_cell_num, self.m_y_cell_num)}")
            return None

    def NeighbourCells_(self, point: np.ndarray) -> np.ndarray:
        # world in, grid out
        return self._NeighbourCells_(self.WorldToGrid(*point))

    def _NeighbourValidNoCollisionCells_(self, point: np.ndarray) -> np.ndarray:
        ncs = self._NeighbourCells_(point)
        ncs_valid = []
        for c in ncs:
            if self._IsCellValid(*c) and not self._IsCellInCollision(*c):
                ncs_valid.append(c)
        return np.array(ncs_valid)

    def NeighbourCells(self, point: np.ndarray) -> np.ndarray:
        # world in, world out
        return np.array([self.GridToWorld(*c) for c in self._NeighbourCells_(self.WorldToGrid(*point))])

    def _NeighbourCells_(self, grid_idx: np.ndarray) -> np.ndarray:
        # grid in, grid out
        if self.m_neighbour_dxy is None:
            self.m_neighbour_dxy = np.empty((8, 2), dtype=int)
            i = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    self.m_neighbour_dxy[i] = [dx, dy]
                    i += 1
        return self.m_neighbour_dxy + grid_idx

    def _NeighbourCircleCells(self, gcenter: np.ndarray, gradius: int) -> np.ndarray:
        max_dist_sq = gradius * gradius
        gresult = []
        for dx in range(-gradius, gradius+1):
            for dy in range(-gradius, gradius+1):
                dist = dx * dx + dy * dy
                if dist > max_dist_sq or dist == 0:
                    continue
                gresult.append(gcenter + np.array([dx, dy]))
        return np.array(gresult)

    def _CellBoundPoints(self, x_idx: int, y_idx: int) -> np.ndarray:
        x, y = self.GridToWorld(x_idx, y_idx)
        cell_x0 = x - self.m_resolution_half
        cell_x1 = x + self.m_resolution_half
        cell_y0 = y - self.m_resolution_half
        cell_y1 = y + self.m_resolution_half
        return np.array([[cell_x0, cell_x1, cell_x1, cell_x0], [cell_y0, cell_y0, cell_y1, cell_y1]])

    def _Compute2PointsGradient(self, oc: np.ndarray, tc: np.ndarray) -> np.ndarray:
        grd = tc - oc
        grd_norm = np.linalg.norm(grd)
        dist_sq = np.square(grd_norm)
        max_dist_cells_sq = self.m_max_distance_cells * self.m_max_distance_cells
        if grd_norm > 0 and dist_sq < max_dist_cells_sq:
            return (1 - dist_sq / max_dist_cells_sq) / grd_norm * grd
        else:
            return np.zeros(oc.shape[0])

    def Compute2PointsGradient(self, obs: np.ndarray, tar: np.ndarray) -> np.ndarray:
        oc = self.WorldToGrid(*obs)
        tc = self.WorldToGrid(*tar)
        return self._Compute2PointsGradient(oc, tc)

    def AddObstacleRectangle(self, point0: np.ndarray, point1: np.ndarray, update_nearby_grd=False):
        self.m_obstacles.append(Rectangle(self.WorldToGrid(
            *point0), self.WorldToGrid(*point1)))
        obs = self.m_obstacles[-1]
        self.m_max_distance_cells = max(
            self.m_max_distance_cells, math.ceil(max(obs.m_extent) * 2))
        self.m_max_distance = self.m_max_distance_cells * self.m_resolution
        x1, y1 = point1
        coords = []
        x = point0[0]
        while True:
            y = point0[1]
            while True:
                coords.append([x, y])
                y += self.m_resolution
                if y > y1 + self.m_resolution:
                    break
            x += self.m_resolution
            if x > x1 + self.m_resolution:
                break
        return self.AddObstacles(coords, update_nearby_grd)

    def AddObstacles(self, coords_in_world: np.ndarray, update_nearby_grd: bool = False):
        for x, y in coords_in_world:
            og = self.WorldToGrid(x, y)
            self._AddObstacle(og, update_nearby_grd)
        logger.info(f"Finish adding {len(coords_in_world)} obstacles")

    def _AddObstacle(self, grid_idx: np.ndarray, update_nearby_grd: bool = False):
        ocell = self.m_grid[grid_idx[0]][grid_idx[1]]
        if ocell.m_obstacle:
            return
        ocell.m_obstacle = True
        if not update_nearby_grd:
            return
        for dx in range(-self.m_max_distance_cells, self.m_max_distance_cells + 1):
            for dy in range(-self.m_max_distance_cells, self.m_max_distance_cells + 1):
                dxy = np.array([dx, dy])
                tg = grid_idx + dxy
                if not self._IsCellValid(*tg):
                    continue
                self.m_grid[tg[0]][tg[1]
                                   ].m_gradient += self._Compute2PointsGradient(grid_idx, tg)

    def Display(self, path=None, show_gradient=False, show_obstacle=True, show_grid=False,
                show_obstacle_verbose=False, title="Field", curvature=None, increments=None, history_data=None):
        import mpl_toolkits.axisartist as axisartist
        fig = plt.figure(figsize=(5, 5))
        ax = axisartist.Subplot(fig, 111)
        fig.add_axes(ax)
        ax.axis[:].set_visible(False)
        ax.axis["x"] = ax.new_floating_axis(0, 0)
        ax.axis["x"].set_axisline_style("->", size=1.0)
        ax.axis["x"].set_label("x")
        ax.axis["x"].set_axis_direction("bottom")
        ax.axis["y"] = ax.new_floating_axis(1, 0)
        ax.axis["y"].set_axisline_style("-|>", size=1.0)
        ax.axis["y"].set_label("y")
        ax.axis["y"].set_axis_direction("left")
        plt.plot(0, 0, "o", ms=8)
        plt.title(title)
        logger.debug(
            f"DistanceField {self.m_x_cell_num} {self.m_y_cell_num}, origin: {self.GridToWorld(0, 0)}")
        logger.debug(f"")
        plt.plot(*self.GridToWorld(0, 0), "o", ms=4)
        x0, x1, y0, y1 = self.GetGridBound()
        plt.plot([x0, x1], [y0, y0], "r", lw=2)
        plt.plot([x0, x1], [y1, y1], "r", lw=2)
        plt.plot([x0, x0], [y0, y1], "r", lw=2)
        plt.plot([x1, x1], [y0, y1], "r", lw=2)
        if show_obstacle:
            for obs in self.m_obstacles:
                if not isinstance(obs, Rectangle):
                    continue
                corner0 = self._CellBoundPoints(*obs.m_corner0)[:, 0]
                corner2 = self._CellBoundPoints(*obs.m_corner1)[:, 2]
                corner1 = np.array([corner2[0], corner0[1]])
                corner3 = np.array([corner0[0], corner2[1]])
                plt.fill(*np.c_[corner0, corner1, corner2, corner3], "r")

        if show_grid or show_gradient or show_obstacle_verbose:
            for x_idx in range(self.m_x_cell_num):
                if show_grid:
                    plt.plot([x_idx * self.m_resolution + self.m_resolution_half + self.m_x_origin] * 2,
                             [self.m_y_origin - self.m_resolution_half, self.m_y_origin + self.m_y_size + self.m_resolution_half], "r", lw=1)
                for y_idx in range(self.m_y_cell_num):
                    if show_grid:
                        plt.plot([self.m_x_origin - self.m_resolution_half, self.m_x_origin + self.m_x_size + self.m_resolution_half], [
                            y_idx * self.m_resolution + self.m_resolution_half + self.m_y_origin] * 2, "r", lw=1)
                    cell = self.m_grid[x_idx][y_idx]
                    cell_bound_points = self._CellBoundPoints(x_idx, y_idx)
                    if show_obstacle_verbose and cell.m_obstacle:
                        plt.fill(*cell_bound_points, "r")
                    elif show_gradient:
                        gradient_norm = np.linalg.norm(cell.m_gradient)
                        gradient_norm /= 3
                        plt.fill(*cell_bound_points,
                                 color=[min(1, max(0, 1 - gradient_norm))]*3)
                        cell_coord = self.GridToWorld(x_idx, y_idx)
                        gnorm = np.linalg.norm(cell.m_gradient)
                        if gnorm > 0:
                            plt.arrow(*cell_coord, *(self.m_resolution_half * cell.m_gradient / gnorm),
                                      length_includes_head=True, color=(0, 1, 0))
        if path is not None:
            path: np.ndarray = np.array(path)
            plt.plot(*path.T, "g*-", ms=3, lw=1)
            plt.plot(*path[[0, -1]].T, "bo--", ms=4)
            # plt.plot(path[0, 0], path[0, 1], "bo", ms=4)
            # plt.plot(path[-1, 0], path[-1, 1], "b^", ms=4)
            if curvature is not None:
                plt.quiver(path[1:-1, 0], path[1:-1, 1],
                           curvature[:, 0], curvature[:, 1], color=(0, 0, 0))
            if increments is not None:
                plt.quiver(path[1:-1, 0], path[1:-1, 1],
                           increments[:, 0], increments[:, 1], color=(0, 0, 0))
        plt.axis("scaled")
        plt.axis("off")
        if history_data is not None:
            plt.ion()
            line = plt.plot(history_data[0].T, "y^--")[0]
            for i in range(history_data.shape[0]+1):
                line.set_data(history_data[:i].T)
                plt.pause(0.01)
            input("any key to continue...")
        else:
            plt.show()


class Planner:
    def __init__(self, learn_rate: float = 0.001, max_distance: float = 0.01):
        self.m_field = DistanceField(x_size=1.0, y_size=1.0, x_origin=0.0,
                                     y_origin=0.0, resolution=0.01, max_distance=max_distance)
        self.m_learn_rate = learn_rate
        self.m_smooth_weight = 0.8
        self.m_obstacle_weight = 0.2
        self.m_curature_weight = 0.1
        self.m_max_iterations = 5000
        self.m_iterations_after_collision_free = 10

    def AddObstacleRectangle(self, point0: np.ndarray, point1: np.ndarray, update_nearby_grd: bool = False):
        self.m_field.AddObstacleRectangle(point0, point1, update_nearby_grd)

    def PlanWithGradientDescend(self, point0: np.ndarray, point1: np.ndarray, num_free_points: int = 10, show_process: bool = False, process_steps: int = 5, clearance: float = 0.001):
        self.m_field.m_clearance = math.ceil(
            clearance / self.m_field.m_resolution)
        path = np.array([(1 - s) * point0 + s * point1 for s in np.linspace(0.0,
                                                                            1.0, num_free_points + 2, endpoint=True)])
        iteration = 0
        collision_free = False
        try:
            it_collision_free = self.m_max_iterations
            while (iteration < self.m_max_iterations and not collision_free) or iteration - it_collision_free < self.m_iterations_after_collision_free:
                collision_free = True
                o_cost, o_increments = self.ComputeObstacleCost(path)
                s_cost, s_increments = self.ComputeSmoothCost(path)
                c_cost, c_increments = self.ComuteCurvatureCost(path)
                c_increments *= self.m_learn_rate * \
                    np.linalg.norm(o_increments, axis=1).reshape(-1, 1)
                increments = (o_increments + s_increments +
                              c_increments) * self.m_learn_rate
                path[1:-1] += increments
                collision_free = self.CheckPointsCollisionFree(path)
                if collision_free:
                    path = self.UniformPath(path)
                if show_process and iteration % process_steps == 0:
                    self.m_field.Display(
                        path=path, title="{}th plan, uniform {}".format(iteration, collision_free), curvature=c_increments, show_obstacle=False, increments=increments)
                collision_free = self.CheckPointsCollisionFree(path)
                if collision_free and it_collision_free == self.m_max_iterations:
                    it_collision_free = iteration
                    self.m_learn_rate *= 0.5
                logger.info(
                    f"Planning, iteration {iteration}, collision free: {collision_free}, obstacle cost: {o_cost :6.5f}, smooth cost: {s_cost :6.5f}, curvature cost: {c_cost :6.5f}")
                iteration += 1
            logger.info(
                f"Planning finished with {iteration} iterations, collision free: {collision_free}")
        except Exception as e:
            logger.error("Got exception: {}".format(e))
            collision_free = False
        return collision_free, path

    def PlanWithAStar(self, point0: List[float], point1: List[float], clearance=0.001):
        self.m_field.m_clearance = math.ceil(
            clearance / self.m_field.m_resolution)
        field = self.m_field
        path, history = AStartPath(field, field.WorldToGrid(*point0),
                                   field.WorldToGrid(*point1))
        if path is not None:
            path = self.SmoothFilter(path)
            return True, path, history
        else:
            return False, None, None

    def PlanWithRRT(self, point0, point1, step_size=0.2, show_process=False, process_steps=1, clearance=0.001):
        # TODO: need to be improved
        self.m_field.m_clearance = math.ceil(
            clearance / self.m_field.m_resolution)
        path = [point0, point1]
        iteration = 0
        collision_free = False
        path_copy = path.copy()
        while not collision_free:
            collision_free = True
            i = 0
            while i < len(path) - 1:
                seg_no_colli = self.CheckLinearPathCollisionFree(
                    path[i], path[i+1])
                collision_free &= seg_no_colli
                if not seg_no_colli:
                    point = self.m_field.RandomNoCollisionCellInRange(
                        path[i], step_size)
                    path_copy.insert(i+1, point)
                    idx = i+1 if i+1 < len(path_copy) else len(path_copy) - 1
                    # short cut on path
                    for j in range(i+1):
                        if self.CheckLinearPathCollisionFree(path_copy[j], path_copy[idx]):
                            for k in range(j+1, idx):
                                path_copy.pop(k)
                            break
                    break
                i += 1
            path = path_copy.copy()
            if show_process and iteration % process_steps == 0:
                self.m_field.Display(path=path, title="RRT it {}, collision free {}".format(
                    iteration, collision_free))
            if collision_free:
                break
            iteration += 1
        logger.info(
            f"Finish planning with RRT with {iteration} iterations, collision free {collision_free}")
        return collision_free, np.array(path)

    def CheckLinearPathCollisionFree(self, start, end, num=50):
        for s in np.linspace(0, 1, num=num, endpoint=True):
            p = (1-s) * start + s * end
            if self.m_field.IsCellInCollision(*p):
                return False
        return True

    def CheckPointsCollisionFree(self, points):
        half_idx = points.shape[0] // 2
        for point in points[half_idx:]:
            if self.m_field.IsCellInCollision(*point):
                return False
        for point in points[:half_idx]:
            if self.m_field.IsCellInCollision(*point):
                return False
        return True

    def ComputeObstacleCost(self, waypoints):
        center_weight = 0.5
        nearby_weight = 0.5
        num_free_points = waypoints.shape[0] - 2
        increments = np.zeros((num_free_points, waypoints.shape[1]))
        cost = 0.0
        for i in range(num_free_points):
            point = waypoints[i+1]
            grd = self.m_field.GetObstacleCenterGradient(
                *point) * center_weight + self.m_field.GetNearbyObstacleGradient(*point) * nearby_weight
            cost += np.linalg.norm(grd)
            increments[i] = grd
        return cost * self.m_obstacle_weight, increments * self.m_obstacle_weight

    def ComuteCurvatureCost(self, waypoints):
        vel = (waypoints[1:] - waypoints[:-1])
        cur = vel[1:] - vel[:-1]
        # R = np.array([[0.0, -1.0], [1.0, 0.0]])
        # cur = (R @ vel[:-1].T).T
        return np.linalg.norm(cur), -cur * self.m_curature_weight

    def ComputeSmoothCost(self, waypoints, step_length=None):
        if step_length is None:
            l_avg = np.linalg.norm(
                waypoints[-1] - waypoints[0]) / (waypoints.shape[0] - 1)
        else:
            l_avg = step_length
        cost = 0
        increments = np.zeros((waypoints.shape[0] - 2, waypoints.shape[1]))
        point_pre = waypoints[0]
        point_cur = waypoints[1]
        for i in range(1, waypoints.shape[0] - 1):
            point_next = waypoints[i+1]
            l_cur = np.linalg.norm(point_cur - point_pre)
            l_next = np.linalg.norm(point_next - point_cur)
            cost += np.square((l_cur - l_avg) / l_avg)
            increments[i-1] = 2*(l_cur - l_avg) * (point_cur - point_pre) / l_cur - \
                2 * (l_next - l_avg) * \
                (point_next - point_cur) / l_next
            point_pre, point_cur = point_cur, point_next
        return cost * self.m_smooth_weight, increments * self.m_smooth_weight

    @staticmethod
    def UniformPath(waypoints):
        waypoints = __class__.SmoothFilter(waypoints)
        # use cublic spline
        ss_waypoints = np.r_[[0.0], np.add.accumulate(
            np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1))]
        path = interpolate.CubicSpline(
            ss_waypoints, waypoints, bc_type="clamped")
        waypoints = np.array([path(s) for s in np.linspace(
            0.0, ss_waypoints[-1], waypoints.shape[0], endpoint=True)])
        logger.info("Finish uniforming path!")
        return waypoints

    @staticmethod
    def SmoothFilter(path: np.ndarray) -> np.ndarray:
        # from scipy import signal
        # average filter with size 1
        path_smooth = path.copy()
        kernal_size = 1
        kernal = np.tile([1/(2 * kernal_size + 1)], kernal_size * 2 + 1)
        for i in range(1, path.shape[0] - 1):
            neighbour = np.zeros((kernal_size*2+1, path.shape[1]))
            neighbour[kernal_size] = path[i]
            if i - kernal_size >= 0:
                neighbour[:kernal_size] = path[i-kernal_size:i]
            else:
                neighbour[:kernal_size] = path[i-1]
            if i + kernal_size + 1 < path.shape[0]:
                neighbour[kernal_size+1:] = path[i+1:kernal_size+i+1]
            else:
                neighbour[kernal_size+1:] = path[i+1]
            path_smooth[i] = neighbour.T @ kernal
        return path_smooth


def Test_DistanceField():
    # Test_DistanceField
    field = DistanceField(x_size=1.0, y_size=1.0, x_origin=0.0,
                          y_origin=0.0, resolution=0.01, max_distance=0.05)
    field.AddObstacles([[0.4, 0.4], [0.39, 0.4], [0.38, 0.4], [0.37, 0.4]])
    field.AddObstacleRectangle([0.5, 0.5], [0.7, 0.6])
    for cgrid in field.NeighbourCells_(np.array([0.4, 0.4])):
        field._AddObstacle(cgrid)
    path = np.array([[0.3, 0.3],
                     [0.35, 0.23],
                     [0.7, 0.8]])
    # print(field.NeighbourCells(np.array([0.5, 0.5])))
    field.Display(path=path, show_obstacle=True, show_obstacle_verbose=True)


def Test_Planner(display_result=False, update_nearby_grd=False, random_seed=None):
    planner = Planner(learn_rate=0.001, max_distance=0.1)

    # Adding obstacles
    # field = planner.m_field
    # field.AddObstacles([[0.3, 0.4], [0.29, 0.4], [0.28, 0.4], [0.27, 0.4]])
    # field.AddObstacleRectangle([0.5, 0.5], [0.7, 0.6])
    # for cgrid in field.NeighbourCells_(np.array([0.3, 0.4])):
    #     field._AddObstacle(cgrid)
    # planner.AddObstacleRectangle([0.45, 0.5], [0.65, 0.6])

    # Adding random obstacles
    for _ in range(2):
        if random_seed is None:
            i = np.random.randint(2, 1000)
        else:
            i = random_seed
        # if _:
        #     np.random.seed(320)
        # else:
        #     np.random.seed(702)
        points = np.random.uniform(low=0.15, high=0.85, size=(2, 2))
        while min(points[1] - points[0]) < 0.05 or max(points[1] - points[0]) > 0.5:
            i += 1
            np.random.seed(i)
            points = np.random.uniform(low=0.15, high=0.85, size=(2, 2))
        planner.AddObstacleRectangle(*points, update_nearby_grd)
        logger.info("Random seed {}".format(i))

    pos_start = np.array([0.1, 0.1])
    pos_target = np.array([0.9, 0.9])
    history_data = None
    # success, path = planner.PlanWithGradientDescend(pos_start, pos_target,
    #                                                        num_free_points=40, show_process=True, process_steps=100, clearance=0.02)
    # success, path = planner.PlanWithRRT(
    #     pos_start, pos_target, show_process=True, process_steps=1)
    success, path, history_data = planner.PlanWithAStar(pos_start, pos_target,
                                                        clearance=planner.m_field.m_resolution*1)
    if display_result:
        planner.m_field.Display(path=path, title="Final opt result, no collision {}".format(success),
                                show_obstacle_verbose=True, history_data=history_data)
    return success


if __name__ == "__main__":
    # Test_DistanceField()
    Test_Planner(display_result=True, update_nearby_grd=False)
