from collections import namedtuple

Coordinates = namedtuple('Coordinates', 'x y z')


class LorenzModel:

    def __init__(self, sigma, beta, rho, time_step):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.time_step = time_step

    def run_model(self, num_time_steps):
        position = Coordinates(2., 5., 4.)
        yield position
        for _ in range(num_time_steps):
            position = self.__get_next_position(position)
            yield position

    def __get_next_position(self, current: Coordinates) -> Coordinates:
        nextX = current.x + self.time_step * (self.sigma *
                                              (current.y - current.x))
        nextY = current.y + self.time_step * (
            current.x * (self.rho - current.z) - current.y)
        nextZ = current.z + self.time_step * (
            current.x * current.y - self.beta * current.z)
        return Coordinates(nextX, nextY, nextZ)
