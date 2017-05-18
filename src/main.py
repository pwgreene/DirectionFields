from helpers import *
from stripe_pattern import *
from field_direction import direction_field, write_to_file
from mesh import Mesh
import sys
import getopt
import os

"""
the main stripe pattern algorithm.
K: the triangle mesh
X:
v:
"""
def compute_stripe_pattern(K, X, v):
    adj = adjacency_matrix(K)
    theta = vertex_angles(K, adj)
    omega, s = edge_data(K, adj, theta, X, v)
    A = energy_matrix(K, adj, omega, s)
    B = mass_matrix(K)
    psi = principal_eigenvector(A, B)
    alpha, n, S = texture_coordinates(K, psi, omega, s)
    return alpha, n, S

def compute_field_direction(filename, n=1, s=1., field_type=None):
    if filename.split(".")[1] != "off":
        raise Exception("must input a file of type .off")
    which_file = filename.split(".")[0]
    mesh = Mesh(*parse_off('../data/'+which_file+'.off'))
    # mesh = trimesh.Trimesh()
    # mesh.vertices, mesh.faces = parse_off('../data/'+which_file+'.off')

    u, X, scale, singularities = direction_field(mesh, s, n, field_type=field_type)
    if len(singularities) == 0:
        write_to_file(mesh, u, n, X, write_angle=True, filename='../data/'+which_file+".txt")
    else:
        write_to_file(mesh, u, n, X, write_angle=True, singularities=None, filename='../data/'+which_file+".txt")

def main(argv):
    args = ["npr"]
    try:
        opts, args = getopt.getopt(argv, "f:t:s:n:", args)
    except getopt.GetoptError:
        print 'main.py -f <inputfile> -t "SM" (smoothest) or "CA" (curvature aligned) -s s (float) -n (int) [--npr]'
        sys.exit(2)
    s = 0
    n = 1
    npr = False
    filename = None
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -f <inputfile> -t "SM" (smoothest) or "CA" ' \
                  '(curvature aligned) -s s (float in [0, 1]) -n (int) [--npr]'
            sys.exit()
        elif opt == "-f":
            filename = arg
        elif opt == "-t":
            field_type = arg
        elif opt == "-s":
            s = float(arg)
        elif opt == "-n":
            n = int(arg)
        elif opt == "--npr":
            npr = True
        else:
            print "unexpected argument", opt
            print 'main.py -f <inputfile> -t "SM" (smoothest) or "CA" ' \
                  '(curvature aligned) -s s (float in [0, 1]) -n (int) [--npr]'
            sys.exit()

    if filename is None:
        print "Please input a filename. use -h for help"
        sys.exit()

    compute_field_direction(filename, n, s, field_type=field_type)
    obj_name = filename.split(".")[0]
    # call C++ executable (make sure its built first in /viewer/build directory -- in root)
    if npr:
        os.system("cat ../data/"+obj_name+".off ../data/"+obj_name+".txt | ../viewer/build/NPRViewer")
    else:
        os.system("cat ../data/"+obj_name+".off ../data/"+obj_name+".txt | ../viewer/build/fieldViewer")


if __name__ == "__main__":
    # initialize K, X, v
    # K = trimesh.load_mesh('../moomoo.off')
    # v = np.ones((len(K.vertices),))
    # v *= .5
    # X = np.ones((len(K.vertices),), dtype=np.complex)
    # print stripe_pattern(K, X, v)

    main(sys.argv[1:])





