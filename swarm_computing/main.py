import numpy as np

from particle import Particle
from particle_swarm_optimisation import PSO
from optproblems.cec2005 import F1


def main():
    swarm_size = 1000
    informant_size = 10
    particle_dimension = 10
    global_best = np.random.rand(particle_dimension)
    ret_vel = 1
    ret_personal_best = 1
    ret_informants_best = 1
    ret_global_best = 2
    particle_jump_size = 0.5
    particles = [
        Particle(
            np.random.uniform(-100, 100, particle_dimension),
            np.random.uniform(-0.1, 0.1, particle_dimension),
        )
        for i in range(swarm_size)
    ]

    pso = PSO(
        swarm_size,
        informant_size,
        particles,
        global_best,
        ret_vel,
        ret_personal_best,
        ret_informants_best,
        ret_global_best,
        particle_jump_size
    )
    problem = F1(particle_dimension)
    problem_boundary = [-100, 100]
    problem_optimum = -450
    result = pso.swarm_optimisation(problem, problem_boundary, problem_optimum)

    return result


if __name__ == '__main__':
    main()
