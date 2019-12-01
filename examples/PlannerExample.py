"""
By Zhen Xiao, Nov 29, 2019
"""
from src.Planner import Planner, logging, np
from IPython import embed
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR)
             

def Planner_Example(display_result=False, update_nearby_grd=False, random_seed=None, a_star_speed_prior=False):
    planner = Planner(learn_rate=0.01, resolution=0.05, max_distance=0.3)

    # Adding obstacles
    field = planner.m_field
    field.m_clearance = 1
    # field.AddObstacles([[0.3, 0.4], [0.29, 0.4], [0.28, 0.4], [0.27, 0.4]])
    # field.AddObstacleRectangle([0.5, 0.5], [0.7, 0.6])
    # for cgrid in field.NeighbourCells_(np.array([0.3, 0.4])):
    #     field._AddObstacle(cgrid)
    # planner.AddObstacleRectangle([0.45, 0.5], [0.65, 0.6])
    # field.AddObstacleRectangle(np.array([0.5, 0.5]), np.array(
    #     [0.6, 0.6]), update_nearby_grd=False)

    # Adding random obstacles
    num_obs = 1
    if random_seed is not None:
        num_obs = len(random_seed)
    for _ in range(num_obs):
        i = random_seed[_] if random_seed is not None else np.random.randint(
            0, 10000)
        np.random.seed(i)
        points = np.random.uniform(low=0.18, high=0.88, size=(2, 2))
        while min(points[1] - points[0]) < 0.05 or max(points[1] - points[0]) > 0.5:
            i += 1
            np.random.seed(i)
            points = np.random.uniform(low=0.18, high=0.88, size=(2, 2))
        planner.AddObstacleRectangle(*points, update_nearby_grd)
        logger.info("Random seed {}".format(i))

    pos_start = np.array([0.1, 0.1])
    pos_target = np.array([0.94, 0.94])
    history_data = None
    success, path = planner.PlanWithGradientDescend(pos_start, pos_target,
                                                           num_free_points=30, show_process=True, process_steps=1000, clearance=0.01)
    # success, path = planner.PlanWithRRT(
    #     pos_start, pos_target, show_process=True, process_steps=1)
    # success, path, history_data = planner.PlanWithAStar(
    #     pos_start, pos_target, clearance=0.001, speed_prior=a_star_speed_prior)
    if display_result:
        planner.m_field.Display(path=path, title="Gradient Descending Planning Result",
                                show_obstacle_verbose=True, history_data=history_data, show_grid=True)
    return success


if __name__ == "__main__":
    Planner_Example(display_result=True, update_nearby_grd=False)
