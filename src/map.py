import numpy as np




class Cell():


    def __init__(self, center, cell_length ) -> None:
        self._center = center
        self._cell_length = cell_length
        self._probability = 0
        self._covariance = 0


    @property
    def corners(self):
        offset = self._cell_length/2
        corner_vectors = np.array([[0,offset],[offset,0],[0,-offset],[-offset,0]])
        return self._center + corner_vectors

    @property
    def center(self):
        return self._center

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, prob):
        self._probability = prob

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        self._covariance = cov

    def isNeighbor(self,cell):
        return np.linalg.norm(cell.center - self._center) <= np.sqrt(2*self._cell_length)

    def __str__(self) -> str:
        return f"Cell: Center:{self._center} Prob: {self._probability} Cov: {self._covariance}"




class Map():


    def __init__(self,starting_point = None, ending_point=None, cell_size = 1) -> None:
        self._cells = []
        self._dimensions = (starting_point,ending_point)
        self._cell_size = cell_size

        if starting_point is None:
            starting_point = (0.5,0.5)

        if ending_point is not None:
            for i in np.arange(start=starting_point[0],stop=ending_point[0], step=self._cell_size):
                for j in np.arange(start=ending_point[1],stop = starting_point[1], step =-self._cell_size):
                    center = np.array([i,j])
                    self._cells.append(Cell(center,self._cell_size))
    


    def add_new_cells(self, new_cells):
        for cell in new_cells:
            self._cells.append(cell)

    @property
    def get_dimensions(self):
        return self._dimensions

    @property
    def get_cells(self):
        return self._cells
    @property
    def get_cell_size(self):
        return self._cell_size

