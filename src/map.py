import numpy as np
import matplotlib.pylab as plt

WALL_OFFSET = 5


class Map:

    def __init__(self):
        _nodes = []
        _devices = []
        self._fig, self._ax = plt.subplots()

        plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
        plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
        plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
        plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

        self.add_node(np.array([-5, 5]))
        self.add_node(np.array([5, -5]))
        self.add_node(np.array([-5, -5]))
        self.add_node(np.array([5, 5]))
        self.add_node(np.array([0, 0]))

        self.add_device(np.array([0, 2.5]))

    def add_node(self, position: np.array):
        '''
        Adds a new node to the map

                Parameters:
                        position (numpy.array) the position of the node
        '''
        self._ax.add_artist(plt.Circle(position, 0.25, color='blue'))

    def add_device(self, position: np.array):
        '''
        Adds a new device to the map

                Parameters:
                        position (numpy.array) the position of the node
        '''
        self._ax.add_artist(plt.Circle(position, 0.25, color='red'))


if __name__ == '__main__':
    Map()

    plt.show()
