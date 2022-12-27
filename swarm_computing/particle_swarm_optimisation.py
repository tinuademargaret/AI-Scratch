"""
Implementation of particle swarm optimisation based on
Kennedy and Eberhart s standard PSO paper https://fumacrom.com/1fjk9
and the pseudocode written in the book Essentials of Metaheuristics
"""
import math
import random
import numpy as np


class PSO:

    def __init__(self,
                 swarm_size,
                 informant_size,
                 swarm_particles,
                 global_best,
                 ret_vel,
                 ret_personal_best,
                 ret_informants_best,
                 ret_global_best,
                 particle_jump_size
                 ):
        self.swarm_size = swarm_size
        self.informant_size = informant_size
        self.swarm_particles = swarm_particles
        self.global_best = global_best
        self.ret_vel = ret_vel
        self.ret_personal_best = ret_personal_best
        self.ret_informants_best = ret_informants_best
        self.ret_global_best = ret_global_best
        self.particle_jump_size = particle_jump_size
        self.global_best_fitness_score = math.inf

    def update_swarm_fitness(self, problem):
        """
        Update the fitness of all the particles in a swarm
        and the global fitness
        :param problem: benchmark function
        """

        for i, particle in enumerate(self.swarm_particles):
            particle.update_fitness(problem)
            particle.update_informant_fitness(problem)
            if particle.fitness < self.global_best_fitness_score:
                self.global_best_fitness_score = particle.min_error
                self.global_best = particle.personal_best

    def assign_informants(self):
        """
        Randomly assigns informants from the swarm for all particles
        :return: 
        """

        for particle in self.swarm_particles:
            informants = np.random.choice(self.swarm_particles, size=self.informant_size)
            particle.informants = informants

    def swarm_optimisation(self, problem, problem_boundary, problem_optimum):
        """
        The PSO algorithm. Iterates to find the best solution. Terminates
        when the global best fitness score is close to the global optimum
        or when the maximum number of iterations is reached.
        :param problem: benchmark function
        :param problem_boundary: boundary of the benchmark function
        :param problem_optimum: Global optimum value of the test problem
        :return: global best fitness score, and number of evaluations.
        """
        self.assign_informants()
        max_evaluations = 500
        cost_history = []
        
        num_evaluations = 0
        while self.global_best_fitness_score - problem_optimum > 0.00001 and num_evaluations < max_evaluations:
            
            self.update_swarm_fitness(problem)
            
            for i, particle in enumerate(self.swarm_particles):
                a = (0.4/max_evaluations**2) * (num_evaluations - max_evaluations)**2 + 0.4
                b = random.uniform(0, self.ret_personal_best)
                c = random.uniform(0, self.ret_informants_best)
                d = random.uniform(0, self.ret_global_best)
                x_g = self.global_best

                particle.update_velocity(a, b, c, d, x_g)

                particle.update_position(self.particle_jump_size, problem_boundary)
                
            num_evaluations += 1
            
        return self.global_best_fitness_score, num_evaluations, cost_history
