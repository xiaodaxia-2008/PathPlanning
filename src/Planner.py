"""
By Zhen Xiao, Nov 27, 2019
"""
from .AStar import AStartPath
from .DistanceField import DistanceField, logging, np, embed
from typing import List, Union
import math
from scipy import interpolate

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, learn_rate: float = 0.001, resolution=0.1, max_distance: float = 0.01):
        self.m_field = DistanceField(x_size=1.0, y_size=1.0, x_origin=0.0,
                                     y_origin=0.0, resolution=resolution, max_distance=max_distance)
        self.m_learn_rate = learn_rate
        self.m_smooth_weight = 0.4
        self.m_obstacle_weight = 0.5
        self.m_curature_weight = 0.1
        self.m_max_iterations = 2000
        self.m_iterations_after_collision_free = 10

    def AddObstacleRectangle(self, point0: np.ndarray, point1: np.ndarray, update_nearby_grd: bool = False):
        self.m_field.AddObstacleRectangle(point0, point1, update_nearby_grd)

    def PlanWithGradientDescend(self, point0: np.ndarray, point1: np.ndarray, num_free_points: int = 10, show_process: bool = False, process_steps: int = 5, clearance: float = 0.01):
        if self.m_field.IsCellInCollision(*point0):
            logger.error("Start point {} is in collision!".format(point0))
            return False, np.array([point0, point1])
        elif self.m_field.IsCellInCollision(*point1):
            logger.error("End point {} is in collision!".format(point1))
            return False, np.array([point0, point1])
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
                iteration += 1
                if show_process and iteration % process_steps == 0:
                    self.m_field.Display(
                        path=path, title="{}th plan, uniform {}".format(iteration, collision_free), curvature=c_increments, show_obstacle=True, increments=increments)
                collision_free = self.CheckPointsCollisionFree(path)
                if collision_free and it_collision_free == self.m_max_iterations:
                    it_collision_free = iteration
                    self.m_learn_rate *= 0.5
                logger.info(
                    "Planning, iteration {}, collision free: {}, obstacle cost: {:6.5f}, smooth cost: {:6.5f}, curvature cost: {:6.5f}".format(iteration, collision_free, o_cost, s_cost, c_cost))
            logger.info(
                "Planning finished with {} iterations, collision free: {}".format(iteration, collision_free))
        except Exception as e:
            logger.error("Got exception: {}".format(e))
            collision_free = False
        return collision_free, path

    def PlanWithAStar(self, point0: List[float], point1: List[float], clearance=0.001, speed_prior: bool = False):
        if self.m_field.IsCellInCollision(*point0):
            logger.error("Start point {} is in collision!".format(point0))
            return False, np.array([point0, point1])
        elif self.m_field.IsCellInCollision(*point1):
            logger.error("End point {} is in collision!".format(point1))
            return False, np.array([point0, point1])
        self.m_field.m_clearance = math.ceil(
            clearance / self.m_field.m_resolution)
        field = self.m_field
        path, history = AStartPath(field, field.WorldToGrid(*point0),
                                   field.WorldToGrid(*point1), speed_prior=speed_prior)
        if path is not None:
            path = self.SmoothFilter(path)
            return True, path, history
        else:
            return False, None, None

    def PlanWithRRT(self, point0: List[float], point1: List[float], step_size=0.2, max_iterations=1000, show_process=False, process_steps=1, clearance=0.001):
        if self.m_field.IsCellInCollision(*point0):
            logger.error("Start point {} is in collision!".format(point0))
            return False, np.array([point0, point1])
        elif self.m_field.IsCellInCollision(*point1):
            logger.error("End point {} is in collision!".format(point1))
            return False, np.array([point0, point1])
        # TODO: need to be improved
        field = self.m_field
        field.m_clearance = math.ceil(
            clearance / self.m_field.m_resolution)
        path = [point0]
        iteration = 1
        find_path = False
        while not find_path and iteration <= max_iterations:
            cur_point = path[-1]
            if np.linalg.norm(point1 - cur_point) <= step_size:
                if self.CheckLinearPathCollisionFree(cur_point, point1):
                    path.append(point1)
                    find_path = True
                    break
            gen_next_point = False
            while not gen_next_point:
                next_point = field.RandomNoCollisionCellInRange(
                    cur_point, step_size)
                if next_point is None:
                    logger.error("Failed to get RandomNoCollisionCellInRange")
                    return False, None
                gen_next_point = self.CheckLinearPathCollisionFree(
                    cur_point, next_point)
            path.append(next_point)

            if show_process and iteration % process_steps == 0:
                field.Display(path=path, title="RRT it {}".format(
                    iteration))
            iteration += 1
        logger.info(
            "Finish planning with RRT with {} iterations, find path {}".format(iteration, find_path))
        if show_process:
            field.Display(path=path, title="Final RRT path without shortcut")
        # Short cut path
        history = path
        path, history_short_cut = self.ShortCut(path)
        if show_process:
            self.m_field.Display(path=path, title="RRT Short cut",
                                 history_data=history_short_cut, history_data_increment=False)
        return find_path, np.array(path), np.array(history)

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

    def ShortCut(self, path: np.ndarray) -> np.ndarray:
        import random
        history = []
        iteration = 0
        idx_removed = set()
        idx_visited = set()
        idx_right = 0
        idx_max = len(path) - 1
        max_iterations = idx_max
        max_points_removed = len(path) - 2
        while iteration < max_iterations and len(idx_removed) < max_points_removed:
            iteration += 1
            idx_left = random.randint(idx_right, idx_max)
            if idx_left + 2 <= idx_max:
                idx_right = random.randint(idx_left + 2, idx_max)
            else:
                continue
            logger.debug(
                "Iteration {}, idx left {}, idx right {}, idx max {}, len idx visited {}, path len {}".format(
                    iteration, idx_left, idx_right, idx_max, len(idx_visited), len(path)))
            if len(idx_visited) == round((len(path) * len(path) - 3 * (len(path) - 1) + 2) * 0.5):
                break
            if (idx_left, idx_right) not in idx_visited:
                idx_visited.add((idx_left, idx_right))
            else:
                continue
            if self.CheckLinearPathCollisionFree(path[idx_left], path[idx_right],
                                                 num=np.linalg.norm(path[idx_right] - path[idx_left]) // self.m_field.m_resolution_half):
                for i in range(idx_left+1, idx_right):
                    idx_removed.add(i)
            if idx_right > idx_max - 2:
                path = np.array([path[i] for i in range(
                    idx_max+1) if i not in idx_removed])
                idx_right = 0
                idx_removed = set()
                idx_visited = set()
                idx_max = len(path) - 1
                max_points_removed = len(path) - 2
                history.append(path)
        path = np.array([path[i]
                         for i in range(idx_max+1) if i not in idx_removed])
        return path, history


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
    Test_Planner(display_result=True, update_nearby_grd=False)
