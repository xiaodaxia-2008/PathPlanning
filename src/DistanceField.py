"""
By Zhen Xiao, Nov 27, 2019
"""
import os
import logging
import math
import random
import time
import datetime
from typing import List, Union

from IPython import embed

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
CWD_DIR = os.path.dirname(__file__)


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
        if np.all(point0 <= point1):
            self.m_corner0 = point0  # left-down corner
            self.m_corner1 = point1  # right-up corner
        else:
            self.m_corner0 = point1  # left-down corner
            self.m_corner1 = point0  # right-up corner
        self.m_volume: float = np.prod(self.m_corner1 - self.m_corner0 + np.ones(2))

    def __repr__(self):
        return "<Rectangle at 0x{:x}, corner0 {}, corner1 {}, volume {}>".format(id(self), self.m_corner0, self.m_corner1, self.m_volume)


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
                 y_origin: float = 0.0, resolution: float = 0.05, max_distance: float = 0.1, clearance: float = 0.01):
        self.m_x_origin: float = x_origin
        self.m_y_origin: float = y_origin
        self.m_x_size: float = x_size
        self.m_y_size: float = y_size
        self.m_resolution: float = resolution
        self.m_resolution_half: float = 0.5 * self.m_resolution
        self.m_max_distance: float = max_distance
        self.m_max_distance_cells: int = math.ceil(
            self.m_max_distance / self.m_resolution)
        self.m_x_cell_num: int = math.floor(
            self.m_x_size / self.m_resolution + 0.0001) + 1
        self.m_y_cell_num: int = math.floor(
            self.m_y_size / self.m_resolution + 0.0001) + 1
        self.m_grid: List[List[Cell]] = [[Cell() for _ in range(self.m_y_cell_num)]
                                         for _ in range(self.m_x_cell_num)]
        self.m_obstacles = []
        # distance in grid to nereast obstacle cell
        self.m_clearance: int = math.ceil(clearance / self.m_resolution)
        self.m_neighbour_dxy = None

    def GetCell(self, x: float, y: float) -> Cell:
        x_idx, y_idx = self.WorldToGrid(x, y)
        return self.m_grid[x_idx][y_idx]

    def GetNearbyObstacleGradient(self, x: float, y: float) -> np.ndarray:
        collision = self.IsCellInCollision(x, y)
        coeff = 1.0 if collision else 0.2
        return self.GetCell(x, y).m_gradient * coeff

    def GetObstacleCenterGradient(self, x: float, y: float) -> np.ndarray:
        grd = np.zeros((2))
        pos = np.array([x, y])
        for obs in self.m_obstacles:
            if isinstance(obs, Rectangle):
                center = 0.5*(self.GridToWorld(*obs.m_corner0) + self.GridToWorld(*obs.m_corner1))
            else:
                center = obs.m_center
            grd += self.Compute2PointsGradient(center, pos) * obs.m_volume
        collision = self.IsCellInCollision(x, y)
        coeff = 1.0 if collision else 0.2
        return grd * coeff

    def RandomNoCollisionCellInRange(self, center: np.ndarray, radius: float) -> Union[None, np.ndarray]:
        gcenter = self.WorldToGrid(*center)
        gradius = math.ceil(radius / self.m_resolution)
        ncs = self._NeighbourCircleCells_(gcenter, gradius)
        idx_set = set(range(ncs.shape[0]))
        while True:
            idx = random.sample(idx_set, 1)[0]
            idx_set.remove(idx)
            if not self._IsCellInCollision(*ncs[idx]) and self._IsCellValid(*ncs[idx]):
                return self.GridToWorld(*ncs[idx])
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
        return self.m_x_origin - self.m_resolution_half, self.m_x_origin + self.m_x_cell_num * self.m_resolution - self.m_resolution_half, \
            self.m_y_origin - self.m_resolution_half, self.m_y_origin + \
            self.m_resolution * self.m_y_cell_num - self.m_resolution_half

    def IsCellInCollision(self, x: float, y: float):
        return self._IsCellInCollision(*self.WorldToGrid(x, y))

    def _IsCellInCollision(self, x_idx: int, y_idx: int):
        # return self.GetCell(x_idx, y_idx).m_obstacle
        tg = np.array([x_idx, y_idx])
        for obs in self.m_obstacles:
            if isinstance(obs, Rectangle):
                if np.all(obs.m_corner0 - self.m_clearance <= tg) and np.all(tg <= obs.m_corner1 + self.m_clearance):
                    return True
            elif isinstance(obs, Circle):
                if np.linalg.norm(tg - obs.m_center) <= self.m_clearance + obs.m_radius:
                    return True
        return False

    def _IsCellValid(self, x_idx: int, y_idx: int):
        return 0 <= x_idx < self.m_x_cell_num and 0 <= y_idx < self.m_y_cell_num

    def WorldToGrid(self, x: float, y: float) -> np.ndarray:
        x_idx, y_idx = math.floor((x - self.m_x_origin + self.m_resolution_half) / self.m_resolution), math.floor(
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

    def _NeighbourCircleCells_(self, gcenter: np.ndarray, gradius: int) -> np.ndarray:
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

    def _Compute2PointsGradient(self, gpoint0: np.ndarray, gpoint1: np.ndarray) -> np.ndarray:
        return self.Compute2PointsGradient(self.GridToWorld(*gpoint0), self.GridToWorld(*gpoint1))

    def Compute2PointsGradient(self, point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
        grd = point1 - point0
        grd_norm = np.linalg.norm(grd)
        dist_sq = np.square(grd_norm)
        max_dist_sq = self.m_max_distance * self.m_max_distance
        if grd_norm > 0 and dist_sq < max_dist_sq:
            return (1 - dist_sq / max_dist_sq) / grd_norm * grd
        else:
            return np.zeros(point0.shape[0])

    def AddObstacleRectangle(self, point0: np.ndarray, point1: np.ndarray, update_nearby_grd=False):
        obs = Rectangle(self.WorldToGrid(*point0), self.WorldToGrid(*point1))
        self.m_obstacles.append(obs)
        self.m_max_distance_cells = max(
            self.m_max_distance_cells, math.ceil(max(obs.m_corner1 - obs.m_corner0) + self.m_clearance))
        self.m_max_distance = self.m_max_distance_cells * self.m_resolution
        for x in np.arange(point0[0], point1[0], self.m_resolution):
            for y in np.arange(point0[1], point1[1], self.m_resolution):
                self._AddObstacle(self.WorldToGrid(x, y), update_nearby_grd)

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
                show_obstacle_verbose=False, title="Field", curvature=None, increments=None, history_data=None, history_data_increment=True):
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
                    plt.plot([x_idx * self.m_resolution - self.m_resolution_half + self.m_x_origin] * 2,
                             [y0, y1], "r", lw=1)
                for y_idx in range(self.m_y_cell_num):
                    if show_grid:
                        plt.plot([x0, x1], [y_idx * self.m_resolution -
                                            self.m_resolution_half + self.m_y_origin] * 2, "r", lw=1)
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
        plt.axis("scaled")
        plt.axis("off")
        if path is not None:
            path: np.ndarray = np.array(path)
            plt.plot(*path[[0, -1]].T, "bo--", ms=4)
            plt.plot(*path.T, "g*-", ms=3, lw=2)
            if curvature is not None:
                plt.quiver(path[1:-1, 0], path[1:-1, 1],
                           curvature[:, 0], curvature[:, 1], color=(0, 0, 0))
            if increments is not None:
                plt.quiver(path[1:-1, 0], path[1:-1, 1],
                           increments[:, 0], increments[:, 1], color=(0, 0, 0))
        os.makedirs(f"{CWD_DIR}/result_pics/", exist_ok=True)
        plt.savefig(f"{CWD_DIR}/result_pics/PlanningResult.png")
        if history_data is not None:
            plt.ion()
            line = plt.plot(0, 0, "y^--", ms=4, lw=1)[0]
            files = []
            os.makedirs(f"{CWD_DIR}/imgs", exist_ok=True)
            step = math.ceil(len(history_data) / 50)
            # step = 1
            his_len = len(history_data)
            if history_data_increment:
                his_len += 1
            for i in range(his_len):
                if history_data_increment:
                    line.set_data(history_data[:i].T)
                else:
                    line.set_data(history_data[i].T)
                plt.pause(0.01)
                if i % step == 0:
                    files.append(f"{CWD_DIR}/imgs/{i}.png")
                    plt.savefig(files[-1])
            plt.close()
            # input("any key to continue...")
            from .GenerateGif import CreateGif
            import shutil
            
            CreateGif(files, f"{CWD_DIR}/result_pics/PlanResult{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.gif", duration=0.1)
            shutil.rmtree(f"{CWD_DIR}/imgs")
        else:
            plt.show()


def Test_DistanceField():
    # Test_DistanceField
    field = DistanceField(x_size=1.2, y_size=1.2, x_origin=0.0,
                          y_origin=0.0, resolution=0.2, max_distance=0.05)
    # field.AddObstacles([[0.4, 0.4], [0.39, 0.4], [0.38, 0.4], [0.37, 0.4]])
    field.AddObstacleRectangle([0.5, 0.5], [0.7, 0.6])
    # for cgrid in field.NeighbourCells_(np.array([0.4, 0.4])):
    #     field._AddObstacle(cgrid)
    path = np.array([[0.3, 0.3],
                     [0.35, 0.23],
                     [0.7, 0.8]])
    # print(field.NeighbourCells(np.array([0.5, 0.5])))
    field.Display(path=path, show_obstacle=True,
                  show_obstacle_verbose=True, show_grid=True)
    embed()


if __name__ == "__main__":
    Test_DistanceField()
