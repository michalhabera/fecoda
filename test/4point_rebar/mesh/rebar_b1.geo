SetFactory("OpenCASCADE");

// Mesh.RecombineAll = 1; // Apply recombination algorithm to all surfaces, ignoring per-surface spec
// Mesh.Recombine3DAll = 1; // Apply recombination3D algorithm to all volumes, ignoring per-volume spec
// // Mesh.SubdivisionAlgorithm = 2; // Mesh subdivision algorithm (0=none, 1=all quadrangles, 2=all hexahedra)

// Radius for rebars
bar_tension_r = 0.008;
roffset = 0.04;

ydim = 0.3 + bar_tension_r + roffset;
xdim = 0.25;

rlx = roffset + bar_tension_r;
rly = roffset + bar_tension_r;

rrx = xdim - roffset - bar_tension_r;
rry = rly;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {xdim, ydim, 0, 1.0};
Point(3) = {xdim, 0.0, 0, 1.0};
Point(4) = {0.0, ydim, 0, 1.0};

Circle(1) = {rlx, rly, 0, bar_tension_r, 0, 2*Pi};
Circle(2) = {rrx, rry, 0, bar_tension_r, 0, 2*Pi};

Line(3) = {4, 2};
Line(4) = {2, 3};
Line(5) = {3, 1};
Line(6) = {1, 4};

Curve Loop(3) = {6, 3, 4, 5};
Curve Loop(1) = {1};
Curve Loop(2) = {2};

Plane Surface(1) = {1, 2, 3};
Plane Surface(2) = {2};
Plane Surface(3) = {1};

Extrude {0, 0, 3.5} {
  Surface{1}; Surface{2}; Surface{3}; Layers{layers};
}

// rebar tension right
Physical Volume(1) = {2};

// rebar tension left
Physical Volume(2) = {3};

// concrete
Physical Volume(10) = {1};
//+
Characteristic Length {7, 5, 8, 6} = 0.2;
