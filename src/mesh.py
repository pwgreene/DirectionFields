import numpy as np

class Mesh(object):

    def __init__(self, vertices, faces, edges=None):
        self.vertices = vertices
        self.faces = faces
        if edges is None:
            self.edges = np.array((0, 2))
            for i, j, k in faces:
                self.edges = np.vstack((self.edges, np.array([i, j])))
                self.edges = np.vstack((self.edges, np.array([j, k])))
                self.edges = np.vstack((self.edges, np.array([k, i])))
        else:
            self.edges = edges