import math
import numpy as np

from optproblems import Individual


class Particle:
    """
    Defines a particle in the swarm
    """

    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.informants = None
        self.personal_best = None
        self.informants_best = None
        self.fitness = math.inf

    def evaluate_fitness(self, problem):
        """
        Evaluates the fitness of a particle
        :param problem: benchmark function
        :return: Fitness score
        """
        solution = Individual(self.position)
        problem.evaluate(solution)
        return solution.objective_values

    def evaluate_informant_fitness(self, problem):
        """
        Return the best fitness score from the
        particle's informants
        :param problem: benchmark function
        :return: informant's best fitness score
        """
        fitness = math.inf
        best_informant = None

        for informant in self.informants:
            solution = Individual(informant.position)
            problem.evaluate(solution)
            score = solution.objective_values
            if score < fitness:
                fitness = score
                best_informant = informant.position

        return best_informant

    def update_informant_fitness(self, problem):
        """
        Updates the particle's `informant_best`
        attribute
        :param problem: benchmark function
        """
        new_best_informant = self.evaluate_informant_fitness(problem)
        self.informants_best = new_best_informant

    def update_fitness(self, problem):
        """
        Updates the fitness of a particle if
        the new fitness score is better than the existing
        fitness score
        :param problem: benchmark
        """
        score = self.evaluate_fitness(problem)
        if score < self.fitness:
            self.fitness = score
            self.personal_best = self.position

    def update_velocity(self, a, b, c, d, x_g):
        """
        Updates the velocity of a particle
        :param a: Inertia weight
        :param b: personal best acceleration coefficient
        :param c: Informant's best acceleration coefficient
        :param d: global best acceleration coefficient
        :param x_g: positoin of the global best particle
        :return:
        """
        v = self.velocity
        x = self.position
        x_p = self.personal_best
        x_i = self.informants_best

        for i in range(len(v)):
            self.velocity[i] = a * v[i] + b * (x_p[i] - x[i]) + c * (x_i[i] - x[i]) + d * (x_g[i] - x[i])

    def update_position(self, particle_jump_size, problem_boundary):
        """
        Updates the position of the particle.
        :param problem_boundary: boundary of the given benchmark
        function
        :param particle_jump_size: jump size of particle
        :return:
        """
        x = np.copy(self.position)
        v = self.velocity

        x_prime = x + particle_jump_size * v

        new_x_prime = self.reposition_particle(x_prime, problem_boundary)

        if (new_x_prime - x_prime).any() > 0:
            self.recompute_velocity(new_x_prime, particle_jump_size)

        self.position = new_x_prime
        return self.position, self.velocity

    def reposition_particle(self, x_prime, problem_boundary):
        """
        Uses variable wise exponential confined approach
        to re-adjust a particle that goes out of bound
        :param x_prime: new particle position
        :param problem_boundary: boundary of test problem
        :return: re-adjusted position of particle
        """
        r = np.random.uniform()
        x = self.position

        for i in range(len(x_prime)):
            l_b_i = problem_boundary[0]
            u_b_i = problem_boundary[1]
            if x_prime[i] < l_b_i:
                x_prime[i] = x[i] - np.log(1 + r * (np.exp(x[i] - l_b_i) - 1))
            elif x_prime[i] > u_b_i:
                x_prime[i] = x[i] + np.log(1 + r * (np.exp(u_b_i - x[i]) - 1))

        return x_prime

    def recompute_velocity(self, new_x_prime, particle_jump_size):
        """
        Re computes the velocity of a particle if it goes out
        of bound
        :param new_x_prime: re-adjusted particle position
        :param particle_jump_size: particle jump size
        :return:
        """
        self.velocity = (new_x_prime - self.position) / particle_jump_size
