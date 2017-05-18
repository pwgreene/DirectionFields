from helpers import *

"""
Collection of the algorithms listed in pseudocode on last page of the paper
"""

# algorithm 2
def vertex_angles(K, reference_edges=None):
    """
    computes polar coordinates of outgoing halfedges around each vertex
    :param K: triangle mesh
    :param l: edge lengths
    :param adj: adjacency matrix (csr_matrix)
    :return: vector of angles (numpy array)
    """
    theta = np.zeros((len(K.vertices), len(K.vertices)))
    vertex_neighbors = neighboring_vertices(K, reference_edges)
    print vertex_neighbors
    for i in range(len(K.vertices)):
        theta_i = 0 # cumulative angle
        for p in range(len(vertex_neighbors[i])-1): # vertices adjacent to i
            j_p = vertex_neighbors[i][p]
            theta[i, j_p] = theta_i
            theta_i += tip_angle(K, (i, j_p, vertex_neighbors[i][p+1]))
        for j_p in vertex_neighbors[i]:
            theta[i, j_p] = 2*math.pi*theta[i, j_p] / theta_i
    return theta

# algorithm 3
def edge_data(K, adj, theta, X, v):
    """
    initializes basic edge data
    :param K: triangle mesh
    :param adj: adjacency matrix (csr_matrix) of K
    :param theta: vector of angles (output from vertex_angles())
    :param X: unit vectors describing desired pattern orientation
    :param v: values for line frequency
    :return: omega and s (encodes relative orientation)
    """
    rows, cols = adj.nonzero()
    nv = len(K.vertices)
    s = np.zeros((nv, nv))
    omega = np.zeros((nv, nv))
    for i, j in zip(rows, cols):
        if i < j: # only go through edges once (canonically oriented)
            rho = -theta[i, j] + theta[j, i] + math.pi
            s[i, j] = np.real(np.sign(np.exp(1j*rho)*X[i] * X[j]))  # TODO: figure out inner product calculation
            phi_i = np.angle(X[i])
            phi_j = np.angle(s[i, j]*X[j])
            l = np.linalg.norm(K.vertices[i] - K.vertices[j]) # length from i to j
            omega[i, j] = l/2.0*(v[i]*np.cos(phi_i - theta[i, j]) + v[j]*np.cos(phi_j - theta[j, i]))
    return scipy.sparse.csr_matrix(omega), scipy.sparse.csr_matrix(s)

# algorithm 4
def energy_matrix(K, adj, omega, s):
    """
    builds matrix defining the energy
    :param K: the triangle mesh
    :param omega: (VxV numpy array) edge data (output from vertex_angles())
    :param s: (VxV numpy array) edge data (output from vertex_angles())
    :return: A, the energy matrix (2V x 2V numpy array)
    """
    nv = K.vertices.shape[0]
    A = np.zeros((2*nv, 2*nv))
    rows, cols = adj.nonzero()
    for v_i, v_j in zip(rows, cols):
        if v_i < v_j:
            beta_i, beta_j = opposite_angles(K, (v_i, v_j), adj)
            w = .5*(1./np.tan(beta_i) + 1./np.tan(beta_j))

            i = 2*v_i
            j = 2*v_j

            A[i, i] += w
            A[i+1, i+1] += w
            A[j, j] += w
            A[j+1, j+1] += w

            a = w * np.cos(omega[v_i, v_j])
            b = w * np.sin(omega[v_i, v_j])

            if s[v_i, v_j] >= 0:
                A[i+0,j+0], A[i+0,j+1] = -a, -b
                A[i+1,j+0], A[i+1,j+1] = b, -a

                A[j+0,i+0], A[j+0,i+1] = -a, b
                A[j+1,i+0], A[j+1,i+1] = -b, -a
            else:
                A[i+0,j+0], A[i+0,j+1] = -a, b
                A[i+1,j+0], A[i+1,j+1] = b, a

                A[j+0,i+0], A[j+0,i+1] = -a, b
                A[j+1,i+0], A[j+1,i+1] = b, a
    return A


# algorithm 5
def mass_matrix(K):
    """
    builds mass matrix associated with vertices
    :param K: the triangle mesh
    :return: B, the mass matrix (2V x 2V numpy array)
    """
    nv = K.vertices.shape[0]
    B = np.zeros((2*nv, 2*nv))
    for f in K.faces:
        area = triangle_area(K, f)
        for i in f:
            B[i, i] += area/3.
    return B

# algorithm 6
def principal_eigenvector(A, B):
    """
    computes an eigenvector corresponding to the smallest eigenvalue via the inverse power method
    :param A: energy matrix (2V x 2V numpy array)
    :param B: mass matrix (2V x 2V numpy array)
    :return: x, the eigenvector with smallest eigenvalue
    """
    L = cholesky_factor(A)
    x = uniform_rand(A.shape[0])
    N = 20 # number of iterations of the inverse power method
    for i in range(N):
        x = back_solve(L, np.dot(B, x))
        x /= math.sqrt(np.dot(np.dot(x, B), x))
    return x

# algorithm 7
def texture_coordinates(K, psi, omega, s):
    """
    computes final texture coordinates
    :param K: the triangle mesh
    :param psi: an eigenvector (np array - output of principal_eigenvector)
    :param omega: edge data (np array)
    :param s: edge data (np array
    :return: tuple of the following three np arrays:
    alpha (collection of coordinates, one for each triangle corner, and two more for the midpoint m and
    duplicate vertex l of each branch triangle)
    n (indices for each triangle indicating whether one should apply nonlinear interpolation)
    S (indices for each triangle indicating whether one should draw the barycentric subdivision)
    """
    nt, nv = len(K.faces), len(K.vertices)
    alpha = np.zeros((nt, 5))
    S = np.zeros((nt, ))
    n = np.zeros((nt, ))
    v = np.zeros((nv, nv))
    for f in range(nt):
        i, j, k = K.faces[f]
        # is each edge canonical?
        c_ij = 1 if i < j else -1
        c_jk = 1 if j < k else -1
        c_ki = 1 if k < i else -1
        z_i, z_j, z_k = psi[i], psi[j], psi[k]  #local copies of edge data
        v[i, j], v[j, k], v[k, i] = c_ij*omega[i, j], c_jk*omega[j, k], c_ki*omega[k, i]
        S[f] = s[i, j] * s[j, k] * s[k, i]  # compute branch index

        if S[f] < 0: # branch triangle?
            v[k, i] = -v[k, i]  # want transport to tau(i1), not i1
        if s[i, j] < 0:  # make values at j consistent with i
            z_j = np.conj(z_j)
            v[i, j] *=  c_ij
            v[j, k] *= -c_jk
        if S[f]*s[k, i] < 0: # make values at k consistent with i
            z_k = np.conj(z_k)
            v[k, i] *= -c_ki
            v[j, k] *=  c_jk
        # compute angles at triangle corners
        alpha[f, 0] = np.angle(z_i)
        alpha[f, 1] = alpha[f, 0] + v[i, j] - np.angle(np.exp(1j*z_i/z_j))
        alpha[f, 2] = alpha[f, 1] + v[j, k] - np.angle(np.exp(1j*z_j/z_k))
        alpha[f, 3] = alpha[f, 2] + v[k, i] - np.angle(np.exp(1j*z_k/z_i))  # branch coordinate, l
        alpha[f, 4] = alpha[f, 0] + (alpha[f, 3] - alpha[f, 0])/2.0  # midpoint coordinate, m
        n[f] = 1./(2*math.pi) * (alpha[f, 3] - alpha[f, 0])
        # adjust zeros
        alpha[f, 1] -= 2*math.pi*n[f]/3.
        alpha[f, 2] -= 4*math.pi*n[f]/3.
    return alpha, n, S

if __name__ == "__main__":
    # testing accuracy of above functions
    mesh = trimesh.load_mesh('../moomoo.off')
    # mesh.show()
    adj = adjacency_matrix(mesh)
    vertex_angles(mesh)
    # A = np.array([[1, 0], [0, 1]])
    # B = np.array([[4, 4], [3, 4]])
    # print principal_eigenvector(A, B)
