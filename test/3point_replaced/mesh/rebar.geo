SetFactory("OpenCASCADE");

// Mesh.RecombineAll = 1; // Apply recombination algorithm to all surfaces, ignoring per-surface spec
// Mesh.Recombine3DAll = 1; // Apply recombination3D algorithm to all volumes, ignoring per-volume spec
// Mesh.SubdivisionAlgorithm = 2; // Mesh subdivision algorithm (0=none, 1=all quadrangles, 2=all hexahedra)

// Radius for rebars
bar_tension_r = 0.012;
roffset = 0.05;

ydim = 0.3;
xdim = 0.2;

rlx = roffset + bar_tension_r;
rly = roffset + bar_tension_r;

rrx = xdim - roffset - bar_tension_r;
rry = rly;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {xdim, ydim, 0, 1.0};
Point(3) = {xdim, 0.0, 0, 1.0};
Point(4) = {0.0, ydim, 0, 1.0};

// rebars bottom
Circle(1) = {rlx, rly, 0, bar_tension_r, 0, 2*Pi};
Circle(2) = {rrx, rry, 0, bar_tension_r, 0, 2*Pi};
// Circle(7) = {xdim / 2, rly, 0, bar_tension_r, 0, 2*Pi};

// rebars top
Circle(8) = {rlx, ydim - rly, 0, bar_tension_r, 0, 2*Pi};
Circle(9) = {rrx, ydim - rry, 0, bar_tension_r, 0, 2*Pi};
// Circle(10) = {xdim / 2, ydim - rly, 0, bar_tension_r, 0, 2*Pi};

Line(3) = {4, 2};
Line(4) = {2, 3};
Line(5) = {3, 1};
Line(6) = {1, 4};

Curve Loop(3) = {6, 3, 4, 5};

// rebars bottom
Curve Loop(1) = {1};
Curve Loop(2) = {2};
// Curve Loop(7) = {7};

// // rebars top
Curve Loop(8) = {8};
Curve Loop(9) = {9};
// Curve Loop(10) = {10};

Plane Surface(1) = {3, 1, 2, 8, 9};

// rebars bottom
Plane Surface(2) = {2};
Plane Surface(3) = {1};
// Plane Surface(4) = {7};

// // rebars top
Plane Surface(5) = {8};
Plane Surface(6) = {9};
// Plane Surface(7) = {10};

Extrude {0, 0, length} {
  Surface{1}; Surface{2}; Surface{3}; Surface{5}; Surface{6}; Layers{layers};
}

Physical Volume(2) = {2};
Physical Volume(3) = {3};
Physical Volume(4) = {4};
Physical Volume(5) = {5};

// Physical Volume(6) = {6};
// Physical Volume(7) = {7};

// concrete
Physical Volume(10) = {1};
//+
// Characteristic Length {7, 5, 8, 6} = 0.2;
