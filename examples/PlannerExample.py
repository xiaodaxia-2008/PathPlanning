"""
By Zhen Xiao, Nov 29, 2019
"""
from src.Planner import Planner, logging, np
logger = logging.getLogger(__name__)


def Planner_Example(display_result=False, update_nearby_grd=False, random_seed=None):
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
        planner.m_field.Display(path=path, title="A* Planning".format(success),
                                show_obstacle_verbose=True, history_data=history_data)
    return success


if __name__ == "__main__":
    Planner_Example(display_result=True, update_nearby_grd=False)
