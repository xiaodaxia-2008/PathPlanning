"""
By Zhen Xiao, Nov 29, 2019
"""
from examples.PlannerExample import Planner_Example, np

if __name__ == "__main__":
    random_seed = np.random.randint(0, 10000, (2,))
    random_seed = [8881, 7511]
    Planner_Example(method="AStar", display_result=True, update_nearby_grd=False,
                    random_seed=random_seed, a_star_speed_prior=False)
    print(random_seed)
