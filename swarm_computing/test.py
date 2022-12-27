import numpy as np
from optproblems.cec2005 import F1

from particle import Particle
from particle_swarm_optimisation import PSO

swarm_size = 1000
informant_size = 10
particle_dimension = 10
global_best = np.random.rand(particle_dimension)
ret_vel = 1
ret_personal_best = 1
ret_informants_best = 1
ret_global_best = 2
particle_jump_size = 0.5
problem = F1(particle_dimension)

p = Particle(
    np.random.uniform(-100, 100, particle_dimension),
    np.random.rand(particle_dimension),
)

s_p = particles = [
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


def test_evaluate_fitness():
    fitness = p.evaluate_fitness(problem)
    return fitness


def test_evaluate_informant_fitness():
    best_informant = p.evaluate_informant_fitness(problem)
    return best_informant


def test_swarm_optimization():
    result = pso.swarm_optimisation(problem, [-100, 100], -450)
    return result

result = test_swarm_optimization()
print(result[0], result[1])
