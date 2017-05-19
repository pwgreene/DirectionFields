This is an Python implementation of the paper "Globally Optimal Direction Fields" Knoppel et. al. (2013).

The direction field is computed using Numpy and Scipy libraries. The visualizer is a C++ executable that utilizes OpenGL. (note that in order to run the visualizer on non-Mac machines, you may need to recompile the viewer on your machine - see viewer/README.txt)

To compute and visualize a direction field on mesh, you must run main.py and input a .off file. For example:

```
$ cd src
$ main.py moomoo.off
```

Additionally, you can add arguments for the parameters s and n (default 0 and 1 respectively):

```$ main.py moomoo.off -s 0 -n 4```

You can visualize the direction field in NPR style by adding the --npr option:

```$ main.py cup.off -s 0 -n 4 --npr```
