import trimesh
import numpy as np
import random
import math
from copy import copy
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

"""
Compilation of standard routines for the stripe pattern implementation
"""

def tip_angle(K, ijk):
    """
    :param K: the mesh
    :param i: (tup) first, second, third vertex index for triangle t in K
    :return: the angle at vertex i in K
    """
    i, j, k = ijk
    e1 = K.vertices[i] - K.vertices[j]
    e2 = K.vertices[i] - K.vertices[k]
    theta = np.arccos(np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)))
    return theta

def tip(ij, adj):
    """
    :param K: triangle mesh
    :param ij: edge (tuple of vertex indices)
    :param adj: adjacency matrix of K
    :return: the vertex at the tip of ij (oriented clockwise)
    """
    i, j = ij
    i_adj = adj[i]
    j_adj = adj[j]
    # elements equal to two in edge_adj are the vertices where i, j share a vertex (triangle ijk and ijl)
    edge_adj = i_adj + j_adj
    return [k for k in edge_adj.nonzero()[1] if edge_adj[0, k] == 2.0]


def opposite_angles(K, ij, adj):
    """
    :param K: the mesh
    :param ij: (tup) edge in triangle
    :param adj: adjacency matrix (csr_matrix) for mesh K
    :return: (tup) the two angles opposite edge ij (one angle on either adjacent face)
    """
    i, j = ij
    i_adj = adj[i]
    j_adj = adj[j]
    theta = []
    # elements equal to two in edge_adj are the vertices where i, j share a vertex (triangle ijk and ijl)
    edge_adj = i_adj + j_adj
    for k in edge_adj.nonzero()[1]:
        if edge_adj[0, k] == 2.0:
            e_ki = K.vertices[i] - K.vertices[k]
            e_kj = K.vertices[j] - K.vertices[k]
            theta.append(np.arccos(np.dot(e_ki, e_kj)/(np.linalg.norm(e_ki)*np.linalg.norm(e_kj))))
    assert len(theta) == 2 # must have exactly two elements equal to 2.0
    return tuple(theta)



def triangle_area(K, ijk):
    """
    :param K: the mesh
    :param ijk: 3d vector of vertex indices that make up triangle in K
    :return: area of triangle formed by vertices i, j, and k
    """
    i, j, k = ijk
    theta = tip_angle(K, ijk)
    e1 = K.vertices[i] - K.vertices[j]
    e2 = K.vertices[i] - K.vertices[k]
    return .5 * np.linalg.norm(e1) * np.linalg.norm(e2) * np.sin(theta)

def cholesky_factor(A, sparse=False):
    """
    :param A: 2d numpy array (matrix)
    :return: cholesky factor of A
    """
    if sparse:
        return scipy.linalg.cholesky(A.todense(), lower=True)
    else:
        return np.linalg.cholesky(A)

def uniform_rand(n, complex=False):
    """
    :param n: length of returned array
    :return: a vector of n numbers in the interval [-1, 1], picked uniformly at random
    """
    if complex:
        return np.array([np.exp(1j*random.uniform(0, 2*math.pi)) for i in range(n)], dtype=np.complex).transpose()
    else:
        return np.array([random.uniform(-1, 1) for i in range(n)]).transpose()

def size(A):
    """
    :param A: 2d numpy array (matrix)
    :return: size of *square* matrix A. raises error if A is not square
    """
    assert A.shape[0] == A.shape[1]
    return A.shape[0]

def back_solve(L, b):
    """
    :param L: cholesky factor of A
    :param b: (np array) RHS vector
    :return: if L is Cholesky factor of A, solves Ax = b
    """
    x = scipy.linalg.solve_triangular(L, b, lower=True)
    return x

def adjacency_matrix(K):
    """
    :param K: triangle mesh
    :return: a |V| X |V| (sparse) adjacency matrix for K
    """
    nv = K.vertices.shape[0]
    adj = scipy.sparse.lil_matrix((nv, nv))
    for t in K.faces:
        adj[t[0], t[1]] = 1
        adj[t[1], t[2]] = 1
        adj[t[2], t[0]] = 1
    return adj

def faces_adjacent(K):
    """
    :param K: the triangle mesh
    :return: faces_adj, a list of lists where faces_adj[i] is a list of an index of faces adjacent to vertex i
    """
    faces_adj = [[] for i in K.vertices]
    for i in range(len(K.faces)):
        for v in K.faces[i]:
            faces_adj[v].append(i)
    return faces_adj

def neighboring_vertices(K, reference_edges=None):
    """
    :param K: triangle mesh
    :adj: adjacency matrix for K
    :return: neighbors, a list of lists where neighbors[i] is a list of neighbors of vertex i ordered so that
    two adjacent vertices in the list are adjacent in K (ordered counterclockwise)
    """
    neighbors = [[] for i in K.vertices]
    faces_adj = faces_adjacent(K)
    for i in range(len(neighbors)):
        edges = []
        for j in faces_adj[i]:
            f = copy(list(K.faces[j]))
            if f[1] == i:
                f = [f[2]] + [f[0]]
            else:
                f.remove(i)
            edges.append(f)
        if reference_edges is not None:
            for u, v in edges:
                if u == reference_edges[i]:
                    sorted_neighbors = [u, v] # j_0 is provided reference edge
                    break
            edges.remove([u, v])
        else:
            sorted_neighbors = edges[0] # choose j_0 arbitrarily
            edges.remove(edges[0])
        while edges:  # make a path by following edges' end vertices until you are back at the start
            edges_len = len(edges)  # checks to see if edges doesn't decrease on each loop
            for e_i in range(len(edges)):
                if edges[e_i][0] == sorted_neighbors[-1]:
                    # print sorted_neighbors
                    sorted_neighbors.append(edges[e_i][1])
                    if len(edges) > 1:
                        edges = edges[:e_i] + edges[e_i+1:]
                    else:
                        edges = []
                    break
            if len(edges) == edges_len:  # when we don't have a closed mesh
                for e in range(len(edges)):
                    edges[e] = edges[e][::-1]
                sorted_neighbors = sorted_neighbors[::-1]
        # assert sorted_neighbors[-1] == sorted_neighbors[0] # must be equal (assuming closed surface)
        neighbors[i] = sorted_neighbors[::-1]
    return neighbors

def compute_dual_areas(K):
    """
    compute per-vertex dual areas
    :param K: triangle mesh
    :return: |V|x1 np array
    """
    areas = np.zeros((len(K.vertices)))
    for i,j,k in K.faces:
        area = triangle_area(K, (i, j, k))
        areas[i] += area/3.
        areas[j] += area/3.
        areas[k] += area/3.
    return areas

def compute_normals(K):
    """
    calculates per-vertex and per-face normals
    :param K: triangle mesh
    :return: tuple of arrays (per_vertex_normals, per_face_normals), length |V| and |T|, respectively
    """
    nt, nv = len(K.faces), len(K.vertices)
    face_n = np.zeros((nt, 3))
    vertex_n = np.zeros((nv, 3))
    adjacent_faces = [[] for n in range(nv)]
    for i in range(len(K.faces)):
        # compute triangle normals
        edge1 = K.vertices[K.faces[i][1]] - K.vertices[K.faces[i][0]]
        edge2 = K.vertices[K.faces[i][2]] - K.vertices[K.faces[i][1]]
        face_n[i] = np.cross(edge1, edge2)
        face_n[i] /= np.linalg.norm(face_n[i])
        # add face each vertex's list of adjacent faces, to be used later
        for j in range(3):
            adjacent_faces[K.faces[i][j]].append(face_n[i])

    # compute per vertex normals (average of normals of faces), and (u, v) coordinate system
    for i in range(len(K.vertices)):
        total = np.zeros((1, 3))
        for j in range(len(adjacent_faces[i])):
            total += adjacent_faces[i][j]
        vertex_n[i] = total/len(adjacent_faces[i])
        vertex_n[i] /= np.linalg.norm(vertex_n[i])

    return vertex_n, face_n

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. from http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def check_orientation(f, e):
    """
    check whether the orientation of the edge e agrees with the orientation of the face f
    :param f: face, indices of the vertices of the face
    :param e: edge, indices of the edges of the face
    :return: +1 if agree, -1 o.w.
    """
    i, j = e
    # assert i in f and j in f
    # for x in range(len(f)):
    #     if f[x] == i:
    #         i_index = x
    #     elif f[x] == j:
    #         j_index = x
    # return 1 if i_index < j_index else -1
    return 1 if i < j else -1

def project_into_plane(u, n):
    """
    projects v into plane defined by n
    :param v: vector
    :param n: normal (unit) vector
    :return: a new projected vector
    """
    # n should be unit
    return u - np.dot(u, n)*n

def create_tri_grid(n):
    """
    creates an nxn triangluated mesh of the unit grid
    :param n: (int) number of rows/cols
    :return: T (np array) of triangles
    X (np array) of vertices
    """
    T = np.zeros((2*n*n,3), dtype=int)
    X = np.zeros(((n+1)*(n+1),3))

    # GENERATE MESH HERE
    nv = 0
    nt = 0
    for i in range(n+1):
        for j in range(n+1):
            X[nv] = np.array([i/float(n), j/float(n), 0])
            if i > 0 and j < n:
                T[nt] = np.array([nv, nv-n-1, nv-n])
                nt += 1
            if i > 0 and j > 0:
                T[nt] = np.array([nv, nv-1, nv-n-1])
                nt += 1
            nv += 1

    return T, X

def parse_obj(filename):
    vecv = []
    vecf = []
    with open(filename) as f:
        for line in f.readlines():
            items = line.split()
            if not items:
                continue
            if items[0] == 'v':
                vecv.append(items[1:])
            elif items[0] == 'f':
                if '/' in items[1]:
                    #raise NotImplementedError("don't have capabilities for vn or vt")
                    vecf.append([f.split('/')[0] for f in items[1:]])
                else:
                    vecf.append(items[1:])
    # make sure to zero index!
    return np.array([vecv], dtype=np.float64)[0], np.array([vecf], dtype=np.int64)[0]-1

def parse_off(filename):
    with open(filename) as f:
        line = f.readline()
        while not line or line[0] == "#":
            line = f.readline()
        assert "OFF" in line
        line = f.readline()
        nv, nt, ne = [int(i) for i in line.split()]
        vertices = np.zeros((nv, 3), dtype=float)
        faces = np.zeros((nt, 3), dtype=int)
        for i in range(nv):
            line = f.readline()
            vertices[i] = np.array(line.split(), dtype=float)
        for j in range(nt):
            line = f.readline()
            faces[j] = np.array(line.split()[1:], dtype=int)

    return vertices, faces

def obj_to_off(in_file, out_file):
    vertices, faces = parse_obj(in_file)
    with open(out_file, 'w') as out:
        out.write("OFF\n")
        out.write("%d %d 0\n" % (len(vertices), len(faces)))
        for v in vertices:
            out.write("%f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces:
            # zero indexing!
            out.write("3   %d %d %d\n" % (f[0], f[1], f[2]))

def rotate_and_project_into_triangle(K, v, X_v, scale_v, theta, neighbors_v):
    """
    rotates the given vector X_v by theta and appropriately projects it into the triangle in that direction
    :param K: triangle mesh
    :param v: index of vector v
    :param X_v: index of reference vertex (so that edge (v, X_v) is the reference edge for v)
    :param scale_v: float - scale factor for vertex v
    :param theta: float (should NOT be scaled)
    :param neighbors_v: list of neighbors of vertex v
    :return: new projected vertex
    """
    theta_total = 0
    face, offset = None, None
    assert X_v == neighbors_v[0]  # edge (v, X_v) should be the reference edge for v
    for i in range(len(neighbors_v)-1):
        j, k = neighbors_v[i], neighbors_v[i+1]
        theta_jk = tip_angle(K, (v, j, k)) * scale_v
        if theta_total <= theta < theta_total + theta_jk:
            face = (v, j, k)
            offset = theta - theta_total  # how much the vector is rotated into the face
            break
        theta_total += theta_jk

    assert face is not None  # problem if we didn't find the face to match angle

    i, j, k = face
    ij, ik = K.vertices[j] - K.vertices[i], K.vertices[k] - K.vertices[i]
    face_n = np.cross(ij, ik)
    face_n /= np.linalg.norm(face_n)

    new_vec = np.dot(rotation_matrix(face_n, offset), ij)
    return new_vec / np.linalg.norm(new_vec)  # normalize


def plot_vector_field(mesh, x,y,z, u,v,w, length=2, pivot='middle'):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, u, v, w, color="red", length=length, pivot=pivot)
    tri = Poly3DCollection(mesh.triangles)
    tri.set_edgecolor('k')
    tri.set_facecolor((0, 0, 1, .5))
    ax.add_collection3d(tri)

    plt.show()


def rayleigh_quotient_generalized(A, B, x):

    return ((x.getH() * (A*x)) / (x.getH() * (B*x))).item(0)


def residual_generalized(A, B, x):
    x = np.matrix(x)
    if x.shape[0] == 1:
        x = x.transpose()
    lam = rayleigh_quotient_generalized(A, B, x)
    return np.linalg.norm(A*x - lam*(B*x)) / np.linalg.norm(x)


def map_z2color(zval, colormap, vmin, vmax):
    t=(zval-vmin)/float((vmax-vmin)) #normalize val
    print t
    R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+\
           ','+'{:d}'.format(int(B*255+0.5))+')'

def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))

def plotly_trianglecolors(x, y, z, colors, simplices, colormap=cm.cubehelix, plot_edges=False):
    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)

    facecolor = [map_z2color(val, colormap, np.min(colors), np.max(colors)) for val in colors]

    I,J,K=tri_indices(simplices)
    triangles=Mesh3d(x=x,y=y,z=z,
                     # intensity=ncolors,
                     facecolor=facecolor,
                     i=I,j=J,k=K,name='',
                     showscale=True,
                     colorbar=ColorBar(tickmode='array',
                                       tickvals=[np.min(z), np.max(z)],
                                       ticktext=['{:.3f}'.format(np.min(colors)),
                                                 '{:.3f}'.format(np.max(colors))])
                     )

    if plot_edges is False:  # the triangle sides are not plotted
        return Data([triangles])
    else:
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
        lines=Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',line=Line(color='rgb(50,50,50)', width=1.5))
        return Data([triangles, lines])

def plotly_trisurf(x, y, z, colors, simplices, colormap=cm.RdBu, plot_edges=False):

    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)

    ncolors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))

    I,J,K=tri_indices(simplices)
    triangles=Mesh3d(x=x,y=y,z=z,
                     intensity=ncolors,
                     colorscale='Viridis',
                     i=I,j=J,k=K,name='',
                     showscale=True,
                     colorbar=ColorBar(tickmode='array',
                                       tickvals=[np.min(z), np.max(z)],
                                       ticktext=['{:.3f}'.format(np.min(colors)),
                                                 '{:.3f}'.format(np.max(colors))]))

    if plot_edges is False: # the triangle sides are not plotted
        return Data([triangles])
    else:
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
        lines=Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',line=Line(color='rgb(50,50,50)', width=1.5))
        return Data([triangles, lines])

def textured_mesh(mesh, per_vertex_signal, filename):
    x = mesh.vertices.transpose()[0]; y = mesh.vertices.transpose()[1]; z = mesh.vertices.transpose()[2];
    data1 = plotly_trisurf(x, y, z, per_vertex_signal, mesh.faces, colormap=cm.RdBu,plot_edges=True)
    axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
    layout = Layout(width=800, height=800,scene=Scene(xaxis=XAxis(axis),yaxis=YAxis(axis),zaxis=ZAxis(axis),aspectratio=dict(x=1,y=1,z=1)))
    fig1 = Figure(data=data1, layout=layout)
    plot(fig1, filename=filename)

def textured_mesh_byface(mesh, per_face_signal, filename):
    x = mesh.vertices.transpose()[0]; y = mesh.vertices.transpose()[1]; z = mesh.vertices.transpose()[2];
    data1 = plotly_trianglecolors(x, y, z, per_face_signal, mesh.faces,plot_edges=True)
    axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
    layout = Layout(width=800, height=800,scene=Scene(xaxis=XAxis(axis),yaxis=YAxis(axis),zaxis=ZAxis(axis),aspectratio=dict(x=1,y=1,z=1)))
    fig1 = Figure(data=data1, layout=layout)
    plot(fig1, filename=filename)

def timeme(method):
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))

        print "time to compute field:", endTime - startTime,'ms'
        return result

    return wrapper

if __name__ == "__main__":
    # mesh = trimesh.load_mesh('../data/bunny.off')
    # nt, nv = len(mesh.faces), len(mesh.vertices)
    #testing calls for the above functions
    # print uniform_rand(10)
    # A = np.array([[1,0], [0,4]])
    # B = np.array([[1, 2, 3], [4, 5, 6]])
    # assert size(A) == 2
    # try:
    #     size(B)
    # except AssertionError:
    #     print "B is not square"
    # L = cholesky_factor(A)
    # print L
    # print back_solve(L, np.array([1,1]).transpose())


    # adjacent_faces = [[] for i in range(nv)] # keeps a list of adjacent face normals for every vertex
    # # mesh.show()
    # adj = adjacency_matrix(mesh)
    # # print opposite_angles(mesh, (0,16), adj)


    # arbitrary reference vectors
    # X = np.zeros((nv, ))
    # for i in range(nv):
    #     # choose edge ij on triangle ijk
    #     j = adj.getrow(i).nonzero()[1][0]
    #     X[i] = j

    # neighbors = neighboring_vertices(mesh)
    # print neighbors[0]
    # neighbors_reference = neighboring_vertices(mesh, X)
    # print neighbors_reference[0]
    # vertex_n, face_n = compute_normals(mesh)
    #
    # textured_mesh(mesh, np.zeros(nv), 'gauss.html')
    # vertex_n *= 2
    # X = np.zeros((nv, 3))
    # for i in range(nv):
    #     # choose edge ij on triangle ijk
    #     j = adj.getrow(i).nonzero()[1][0]
    #     X[i] = mesh.vertices[j] - mesh.vertices[i]

    # Testing vertex normals
    # plot_vector_field(mesh, mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
    #                   vertex_n[:, 0], vertex_n[:, 1], vertex_n[:, 2])
    # Testing arbitrary vector X[i] (edge ij on triangle ijk)
    # plot_vector_field(mesh, mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
    #                   X[:, 0], X[:, 1], X[:, 2])

    # area = 0
    # for f in mesh.faces:
    #     area += triangle_area(mesh, f)
    # assert area == mesh.area
    #
    # print parse_obj("../data/bunny_1k.obj")
    obj_to_off("../data/blub.obj", "../data/blub.off")
    # parse_off("../data/moomoo.off")
    # a = np.matrix([[1, 1], [1, 1]])
    # b = np.matrix([[1, 1], [1, 1]])
    # x = np.matrix([[1], [1]])
    # print a.shape, b.shape, x.shape
    # print residual_generalized(a, b, x)











