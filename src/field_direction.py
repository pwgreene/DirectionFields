from helpers import *
from section_integrals import *
import warnings

DEBUG = True


def setup(K, n, s):
    # warnings.filterwarnings("error")
    nv, nt = len(K.vertices), len(K.faces)
    adj = adjacency_matrix(K)

    # Compute scaling factor for angles around each vertex. scale[i] is scaling factor for v_i

    # Pick arbitrary reference edge ij. X_i = j (the index of vertex j)
    X = np.zeros((nv, ))
    for i in range(nv):
        # choose edge ij on triangle ijk
        j = adj.getrow(i).nonzero()[1][0]
        X[i] = j

    vertex_neighbors = neighboring_vertices(K, reference_edges=X)

    # Compute transport at edges
    r, scale = compute_edge_transport(K, X, vertex_neighbors, n)

    # Compute holonomy per face (omega_ijk)
    omega = np.zeros((nt, ))
    for f in range(nt):
        i, j, k = K.faces[f]
        omega[f] = np.angle(r[i, j]*r[j, k]*r[k, i])
        assert -math.pi < omega[f] < math.pi

    # Assemble M.
    M = mass(K)

    # Assemble A.
    A = energy(K, M, omega, r, s)

    return M, A, X, scale, vertex_neighbors, r, omega



def compute_edge_transport(K, X, v_neighbors, n):
    """
    computes scale and parallel transport between all edges
    :param K: triangle mesh
    :param X: basis vector per vertex
    :param v_neighbors: per-vertex list of neighbors (ordered counterclockwise)
    :param n: vector field degree
    :return: r, where r[i, j] is the transport between vertex i and j
    s, the scaling factors, where s[i] is the scaling factor for vertex i
    """
    # first compute scaling factor
    nv = len(K.vertices)
    s = np.zeros((nv, ))
    theta = scipy.sparse.lil_matrix((nv, nv))  # stores angle between each edge and the reference vector
    for i in range(nv):
        # print i, v_neighbors[i]
        theta_total = 0
        for p in range(len(v_neighbors[i])-1):
            j, k = v_neighbors[i][p], v_neighbors[i][p+1]
            theta[i, j] = theta_total
            # print theta[i, j]
            theta_total += tip_angle(K, (i, j, k))
        s[i] = 2*math.pi/theta_total

    # compute r, the parallel transport factors
    r = scipy.sparse.lil_matrix((nv, nv), dtype=np.complex)
    for i, j in K.edges:
        # need to rotate e_ij when looking at vertex j
        rho_ij = n*(s[j]*(theta[j, i]) - s[i]*(theta[i, j]) + math.pi)
        r[i, j] = np.exp(1j*rho_ij)

    return r, s


def build_mass_matrix(K, r, omega):
    """
    Building mass matrix as defined in the paper
    :param K: triangle mesh
    :param r: transport at edges (sparse |V|x|V| np array of complex vals)
    :param omega: curvatures on the faces
    :return: mass matrix (|V|x|V| np array)
    """
    nv = len(K.vertices)
    M = np.zeros((nv, nv), dtype=np.complex)
    for ijk in range(len(K.faces)):
        f = mesh.faces[ijk]
        area = triangle_area(K, f)
        for i, j, k in [[f[0], f[1], f[2]], [f[1], f[2], f[0]], [f[2], f[0], f[1]]]:
            M[i, i] += area/6.

            if abs(omega[ijk]) > math.pi:
            # if True:
                M_jk = np.conj(r[j, k])*area
                M_jk *= ((6*np.exp(1j*omega[ijk]) - 6 - 6*1j*omega[ijk] + 3*omega[ijk]**2 + 1j*omega[ijk]**3)/(3*omega[ijk]**4))
            else:
                M_jk = mass_nondirect(omega[ijk])

            M[j, k] += M_jk

    return M


def build_energy_matrix(K, r, omega, M, s):
    """
    Building energy matrix as described in the paper
    :param K: triangle mesh
    :param r: transport at edges (sparse |V|x|V| np array of complex vals)
    :param omega: curvatures on the faces
    :param M: the mass matrix (|V|x|V| np array)
    :param s:
    :return: the energy matrix
    """
    nv = len(K.vertices)
    A = np.zeros((nv, nv), dtype=np.complex)
    D = np.zeros((nv, nv), dtype=np.complex)

    # define some helper functions as outlined in the paper
    def f1(s):
        return 1/s**4 * (3 + 1j*s + s**4/24 - 1j*s**5/60 + (-3 + 2j*s + s**2/2)*np.exp(1j*s))

    def f2(s):
        return 1/s**4 * (4 + 1j*s - 1j*s**3/6 - s**4/12 + 1j*s**5/30 + (-4 + 3j*s + s**2)*np.exp(1j*s))

    # compute Delta
    for ijk in range(len(K.faces)):
        f = K.faces[ijk]
        area = triangle_area(K, f)
        for i, j, k in [[f[0], f[1], f[2]], [f[1], f[2], f[0]], [f[2], f[0], f[1]]]:
            p_ij = K.vertices[j] - K.vertices[i]
            p_jk = K.vertices[k] - K.vertices[j]
            p_ki = K.vertices[i] - K.vertices[k]

            D[i, i] += 1.0/(4*area) * (np.dot(p_jk, p_jk) +
                                       omega[ijk]**2 * ((np.dot(p_ij, p_ij) + np.dot(p_ij, -p_ki) + np.dot(p_ki, p_ki))/90))

            if abs(omega[ijk]) > math.pi:
            # if True:
                D_jk = np.conj(r[j, k])/area * ((np.dot(p_ij, p_ij) + np.dot(p_ki, p_ki)) * f1(omega[ijk]) +
                                                    np.dot(p_ij, -p_ki) * f2(omega[ijk]))
            else:
                g_ii = np.dot(p_ij, p_ij)
                g_ij = np.dot(p_ij, -p_ki)
                g_jj = np.dot(p_ki, p_ki)
                D_jk = dirichlet_nondirect(s, g_ii, g_ij, g_jj)

            D[j, k] += D_jk

    # now compute A, the energy matrix
    for ijk in range(len(K.faces)):
        f = K.faces[ijk]
        area = triangle_area(K, f)
        for i, j, k in [[f[0], f[1], f[2]], [f[1], f[2], f[0]], [f[2], f[0], f[1]]]:
            A[i, i] += D[i, i] - s*omega[ijk]*M[i, i]/area

            epsilon = check_orientation(f, (j, k))
            A[j, k] += D[j, k] - s*(omega[ijk]/area*M[j, k] - epsilon*1j*np.conj(r[j, k])/2.)

    return A


def mass_nondirect(s):
    if s > 0:
        t = s/math.pi * 2 - 1
        return np.conj(m12(t))
    else:
        t = -s/math.pi * 2 - 1
        return m12(t)


def dirichlet_nondirect(s, gii, gij, gjj):
    if s > 0:
        t = s*2 / math.pi - 1
        return np.conj((gii + gjj)*s11(t) + gij*s12(t))
    else:
        t = -s*2 / math.pi - 1
        return (gii + gjj)*s11(t) + gij*s12(t)


def mass(K):
    """
    Mass per vertex is barycentric area (or dual area)
    :param K: triangle mesh
    :return: M, the mass matrix (np array)
    """
    M = scipy.sparse.lil_matrix((len(K.vertices), len(K.vertices)), dtype=np.complex)
    dual_areas = compute_dual_areas(K)
    for i in range(len(K.vertices)):
        M[i, i] = dual_areas[i]/2.
    return M


def energy(K, M, omega, r, s):
    """
    Compute the energy matrix
    :param K: triangle mesh
    :param r:
    :return:
    """
    A = scipy.sparse.lil_matrix((len(K.vertices), len(K.vertices)), dtype=np.complex)
    for ijk in range(len(K.faces)):
        f = K.faces[ijk]
        for i, j, k in [[f[0], f[1], f[2]], [f[1], f[2], f[0]], [f[2], f[0], f[1]]]:
            ik = K.vertices[k]-K.vertices[i]
            jk = K.vertices[k]-K.vertices[j]
            theta = np.arccos(np.dot(-jk, ik)/(np.linalg.norm(jk)*np.linalg.norm(ik)))
            cotan = np.cos(theta)/np.sin(theta)
            w = .5*cotan
            area = triangle_area(K, f)

            A[i, i] += w - s*(omega[ijk]/area)*M[i, i]
            A[i, j] -= w*r[i, j]# + s*(omega[ijk])*1./12*r[i, j]

            A[j, j] += w - s*(omega[ijk]/area)*M[j, j]
            A[j, i] -= w*r[j, i]# + s*(omega[ijk])*1./12*r[j, i]
    return A

def compute_principal_directions(K, v_neighbors, scale):
    """
    compute per-vertex principal directions based on curvature
    :param K: triangle mesh
    :return: |V| x 1 np array
    """
    nv = len(K.vertices)
    principal_directions = np.zeros((nv, ), dtype=np.complex)
    theta = scipy.sparse.lil_matrix((nv, nv))  # stores angle between each edge and the reference vector
    for i in range(nv):
        # print i, v_neighbors[i]
        theta_total = 0
        for p in range(len(v_neighbors[i])-1):
            j, k = v_neighbors[i][p], v_neighbors[i][p+1]
            e_ij = K.vertices[j] - K.vertices[i]
            l = np.linalg.norm(e_ij)
            # compute dihedral angle between faces adjacent to edge (i, j)
            if p == 0:
                n_1 = np.cross(K.vertices[v_neighbors[i][-2]] - K.vertices[i], e_ij)
            else:
                n_1 = np.cross(K.vertices[v_neighbors[i][p-1]] - K.vertices[i], e_ij)
            n_1 /= np.linalg.norm(n_1)
            n_2 = np.cross(e_ij, K.vertices[k] - K.vertices[i])
            n_2 /= np.linalg.norm(n_2)
            alpha = np.arctan2(np.dot(e_ij/l, np.cross(n_2, n_1)), np.dot(n_2, n_1))
            r = complex(np.cos(2*theta_total), np.sin(2*theta_total))
            principal_directions[i] += l * alpha * r

            theta_total += tip_angle(K, (i, j, k))*scale[i]
        principal_directions[i] /= -4.0

    return principal_directions

@timeme
def smoothest_field(M, A, s, nv, power_iteration=False):
    """
    computes the smooth field
    :param M: mass matrix
    :param A: energy matrix
    :param nv: number of vertices in the mesh
    :return: u, the vector of rotations (complex np array)
    """

    # print M
    # print A
    if not power_iteration:
        val, u = scipy.sparse.linalg.eigsh(A - s*M, k=1, M=M, which="SM")
        u = u[:, 0]
        # print "eigenval", val
    else:
        u = uniform_rand(nv, complex=True)
        n_iterations = 20
        L = cholesky_factor(A, sparse=True)  # at least make sure A is pos definite
        for i in range(n_iterations):
            print "iteration", i+1, "...", "residual:", residual_generalized(A, M, u)
            x = scipy.linalg.solve(L*L.T, M*u)
            u = x/np.sqrt(np.dot(x, M*x))

    print "residual:", residual_generalized(A, M, u)
    # for i in range(20):
    #     print np.angle(u[i]), u[i]
    return u

@timeme
def curvature_aligned_field(M, A, q, nv):
    """
    computes a curvature aligned field given M and A
    :param M: mass matrix
    :param A: energy matrix
    :param nv: number of vertices in the mesh
    :return: u, the vector of rotations (complex np array)
    """
    #lambda_t = scipy.sparse.csc_matrix(shape=(nv, 1))
    u = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), M*q)
    return u/np.linalg.norm(u)  # normalize u


def direction_field(K, s, n, field_type="SM"):
    """
    compute the n-direction field on a mesh
    :param K: triangle mesh
    :param s: in range (-1, 1)
    :param n: n-direction (natural number, typically 1, 2, or 4)
    :return: u, a collection of vectors on the vertices
    X, the initial arbitrary vectors at each vertex
    scale, scaling factors per vertex
    singularities, a list of indices of faces where singularities happen (estimated)
    """
    nv = len(K.vertices)
    M, A, X, scale, neighbors, r, omega = setup(K, n, s)
    print "Computed M and A"
    if field_type == "SM":
        print "Computing smoothest field"
        u = smoothest_field(scipy.sparse.csc_matrix(M), scipy.sparse.csc_matrix(A), s, nv)
    elif field_type == "CA":
        print "Computing curvature aligned field"
        q = compute_principal_directions(K, neighbors, scale)
        u = curvature_aligned_field(M, A, q, nv)
    singularities_indices = compute_singularities(K, u, r, omega)
    # for f_i in range(len(singularities_indices)):
    #     if abs(singularities_indices[f_i]) == 1:
    #         print K.faces[f_i], f_i, singularities_indices[f_i]
    #         singularities.append(f_i)
    return u, X, scale, singularities_indices


# mostly for testing
def compute_holonomy(K, n):
    """
    Tester function that performs holonomy calculation per face
    :param K: triangle mesh
    :param n: vector field degree
    :return: omega (holonomy per face) scale (per vertex)
    """
    nv, nt = len(K.vertices), len(K.faces)
    X = np.zeros((nv, ))
    for i in range(nv):
        # choose edge ij on triangle ijk
        j = adj.getrow(i).nonzero()[1][0]
        X[i] = j

    vertex_neighbors = neighboring_vertices(K, reference_edges=X)

    r, scale = compute_edge_transport(K, X, vertex_neighbors, n)
    # print r
    # Compute holonomies
    omega = np.zeros((nt, ))
    for f in range(nt):
        i, j, k = K.faces[f]
        # print i, j, k, np.angle(r[i, j]), np.angle(r[j, k]), np.angle(r[k, i])
        omega[f] = np.angle(r[i, j]*r[j, k]*r[k, i])
        assert -math.pi < omega[f] < math.pi

    return omega, scale

def compute_singularities(K, u, r, omega):
    """
    compute index of singularities per-face for the direction field (psi in paper - section 6.1.3)
    :param K: triangle mesh
    :param u: np array of complex numbers
    :param r: np array of edge transport coefficients
    :return: -1, 0, +1 for every face
    """
    nv, nt = K.vertices.shape[0], K.faces.shape[0]
    singularities_indices = []
    for f in range(nt):
        i, j, k = K.faces[f]
        # print i, j, k
        # print u[i], r[i,j], u[j]
        w_ij = np.angle(u[j]/(r[i, j]*u[i]))
        w_jk = np.angle(u[k]/(r[j, k]*u[j]))
        w_ki = np.angle(u[i]/(r[k, i]*u[k]))
        index = 1./(2*math.pi) * (w_ij + w_jk + w_ki + omega[f])
        if index == -1 or index == 1:
            singularities_indices.append(f)
    return singularities_indices

def test_principal_direction(K, neighbors, scale):
    """
    Just a tester function for principal direction computation. displays the mesh with plotly
    Input is same as compute_principal_direction() function
    :return: None
    """
    directions = compute_principal_directions(K, neighbors, scale)
    textured_mesh(mesh, directions, 'principal_direction.html')
    return

def test_edge_transport():
    """
    Just a tester function for edge transport calculations
    :return: None
    """
    n = 1.0
    K = trimesh.Trimesh()
    K.faces, K.vertices = create_tri_grid(3)
    K.vertices[5][2] = 0.5
    # K.vertices[6][2] = 0.1
    nv = len(K.vertices)
    # Pick arbitrary reference edge ij. X_i = j (the index of vertex j)
    X = np.zeros((nv, ))
    adj = adjacency_matrix(K)
    for i in range(nv):
        # choose edge ij on triangle ijk
        j = adj.getrow(i).nonzero()[1][0]
        X[i] = j

    neighbors = neighboring_vertices(K, reference_edges=X)
    r, scale = compute_edge_transport(K, X, neighbors, n)

    x, y, z = K.vertices[:, 0], K.vertices[:, 1], K.vertices[:, 2]
    vector_field = np.zeros((nv, 3))
    #testing flat plane
    for face in [[9, 5, 6]]:#K.faces:
        print face
        i, j, k = face
        rotation = r[i, j] * r[j, k] * r[k, i]
        print r[i, j], r[j, k], r[k, i]
        print i, scale[i]
        print j, scale[j]
        print k, scale[k]
        print np.angle(rotation)
        for sample_vec in face:
            vector_field[sample_vec] = np.array([1, 1, 0])
            # how to visualize the vector field
            vector_field[sample_vec] = rotate_and_project_into_triangle(K, sample_vec, X[sample_vec], scale[sample_vec],
                                                                        np.angle(rotation), neighbors[sample_vec])
            # ref_edge = K.vertices[int(X[sample_vec])] - K.vertices[sample_vec]
            # ref_edge /= np.linalg.norm(ref_edge)
            # vector_field[sample_vec] = np.dot(rotation_matrix(np.array([0, 0, 1]), np.angle(rotation)), ref_edge)
    u, v, w = vector_field[:, 0], vector_field[:, 1], vector_field[:, 2]
    plot_vector_field(K, x,y,z, u,v,w, length=.2, pivot='tail')


def write_to_file(K, u, n, ref_edges, write_angle=True, singularities=None, filename="output.txt"):
    """
    Write the direction field into filename with each line having 3 elements: v_index ref_edge_index direction
    First line is the number of vertices
    :param K: triangle mesh
    :param u: direction field per vertex
    :param n: direction field degree
    :param write_angle: if true, will write theta, otherwise will write the complex number
    :return: None
    """
    nv = len(K.vertices)
    with open(filename, 'w') as out:
        out.write("%s %s %s\n" % (nv, n, int(singularities is not None)))
        for i in range(nv):
            if write_angle:
                # usually go with this -- no parse complex numbers in C++ ?
                line = "%s %s" % (i, int(ref_edges[i]))
                for k in range(n):
                    # get complex roots of u
                    phi = np.angle(u[i])
                    root = np.absolute(u[i])**1./n * np.exp(1j*(phi/n + 2*k*math.pi/n))
                    line += " %s" % np.angle(root)
            else:
                u_i = u[i]
                line = "%s %s %s" % (i, int(ref_edges[i]), u_i)

            out.write(line + "\n")
        if singularities:
            out.write("%s\n" % len(singularities))
            for f_i in singularities:  # write index of face with a singularity
                out.write("%s\n" % f_i)
    print "data written to", filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("must input a filename")
    filename = sys.argv[1]
    if filename.split(".")[1] != "off":
        raise Exception("must input a file of type .off")
    which_file = filename.split(".")[0]
    mesh = trimesh.Trimesh()  # trimesh.load_mesh('../data/'+which_file+'.off')
    mesh.vertices, mesh.faces = parse_off('../data/'+which_file+'.off')
    adj = adjacency_matrix(mesh)
    nt, nv = len(mesh.faces), len(mesh.vertices)
    n = 1
    s = .1
    write = True
    display = False
    # test_edge_transport()
    u, X, scale = direction_field(mesh, s, n, field_type="CA")
    # print u
    if write:
        write_to_file(mesh, u, n, X, write_angle=True, filename='../data/'+which_file+".txt")
    if display:
        field_vectors = np.zeros((nv, 3))

        vertex_n, face_n = compute_normals(mesh)
        theta = np.zeros((nv, ))
        for i in range(nv):
            theta[i] = np.angle(u[i])# * scale[i]  # scaling??
            X_i = mesh.vertices[int(X[i])] - mesh.vertices[i]

            # from c++ code:
            # x = X_i - np.dot(X_i, vertex_n[i])*vertex_n[i]
            # x /= np.linalg.norm(x)
            # jx = np.cross(face_n[i], x)
            # phi = theta[i] + (2.*math.pi)*n + math.pi/2.
            #
            # field_vectors[i] = np.absolute(u[i]) * (np.cos(phi)*x + np.sin(phi)*jx)
            # field_vectors[i] /= np.linalg.norm(field_vectors[i])

            # from paper
            # rotation = rotation_matrix(vertex_n[i], theta[i])
            # X_i = mesh.vertices[int(X[i])] - mesh.vertices[i]
            # X_i /= np.linalg.norm(X_i)
            # field_vectors[i] = np.dot(rotation, X_i)

        # u = uniform_rand(nv, complex=True)
        # # TODO: right way to reconstruct field from reference edges?
        neighbors = neighboring_vertices(mesh, reference_edges=X)
        for i in range(nv):
            theta = np.angle(u[i])
            theta = theta if theta > 0 else 2*math.pi + theta  # needs to be in range [0, 2pi)
            field_vectors[i] = rotate_and_project_into_triangle(mesh, i, X[i], scale[i], theta, neighbors[i])
        # print theta
        # vals = np.zeros((nv))
        # for i in range(nv):
        #     n_neighbors = 0
        #     for j in adj.getrow(i).nonzero()[1]:
        #         n_neighbors += 1
        #         vals[i] += abs(np.arccos(np.dot(field_vectors[i], field_vectors[j])))
        #     vals[i] /= n_neighbors
        # textured_mesh(mesh, vals, "vals.html")

        # only display part of the vector bundle
        x_v, y_v, z_v, x_d, y_d, z_d = [], [], [], [], [], []
        for i in range(nv):
            if mesh.vertices[i, 2] > 0:
                x_v.append(mesh.vertices[i, 0])
                y_v.append(mesh.vertices[i, 1])
                z_v.append(mesh.vertices[i, 2])
                x_d.append(field_vectors[i, 0])
                y_d.append(field_vectors[i, 1])
                z_d.append(field_vectors[i, 2])

        # display all
        # x_v, y_v, z_v = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
        # x_d, y_d, z_d = field_vectors[:, 0]*2, field_vectors[:, 1]*2, field_vectors[:, 2]*2

        plot_vector_field(mesh, x_v, y_v, z_v, x_d, y_d, z_d, length=.02)

    # test holonomy computation
    # holonomy_per_face, scale = compute_holonomy(mesh, 1)
    # curvature_per_face = np.zeros((nt, ))
    # curvature_per_vertex = np.zeros((len(mesh.vertices),))
    # adjacent_faces = [[] for x in range(nv)]
    # for i in range(len(mesh.faces)):
    #     # gaussian curvature, K_ijk = omega_ijk/(area_ijk*n)
    #     curvature_per_face[i] = holonomy_per_face[i]/(n*triangle_area(mesh, mesh.faces[i]))
    #     for j in range(3):
    #         adjacent_faces[mesh.faces[i][j]].append(i)
    # for i in range(len(mesh.vertices)):
    #     for j in adjacent_faces[i]:
    #         curvature_per_vertex[i] += curvature_per_face[j]
    #     curvature_per_vertex[i] /= len(adjacent_faces[i])

    # print holonomy_per_face
    # print curvature_per_face
    # print holonomy_per_face[0], mesh.faces[0]
    # print mesh.vertices[0], mesh.vertices[16], mesh.vertices[861]
    # textured_mesh(mesh, curvature_per_vertex, 'gauss_from_holonomy.html')
    #
    # textured_mesh(mesh, compute_scale(mesh, adj), 'scale.html')
