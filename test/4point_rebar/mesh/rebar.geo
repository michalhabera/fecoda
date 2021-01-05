SetFactory("OpenCASCADE");

// Radius for rebars
bar_tension_r = 0.01;
bar_compression_r = 0.006;

Point(1) = {0, 0, 0, 1.0};

Point(2) = {0.3, 0.3, 0, 1.0};

Point(3) = {0.3, 0.0, 0, 1.0};

Point(4) = {0.0, 0.3, 0, 1.0};

Circle(1) = {0.03, 0.03, 0, bar_tension_r, 0, 2*Pi};
Circle(2) = {0.27, 0.03, 0, bar_tension_r, 0, 2*Pi};
Circle(7) = {0.15, 0.03, 0, bar_tension_r, 0, 2*Pi};

Circle(8) = {0.03, 0.27, 0, bar_compression_r, 0, 2*Pi};
Circle(9) = {0.27, 0.27, 0, bar_compression_r, 0, 2*Pi};

Line(3) = {4, 2};
Line(4) = {2, 3};
Line(5) = {3, 1};
Line(6) = {1, 4};

Curve Loop(3) = {6, 3, 4, 5};
Curve Loop(1) = {1};
Curve Loop(7) = {7};
Curve Loop(2) = {2};

Curve Loop(8) = {8};
Curve Loop(9) = {9};

Plane Surface(1) = {1, 2, 3, 7, 8, 9};
Plane Surface(2) = {2};
Plane Surface(3) = {1};
Plane Surface(7) = {7};

Plane Surface(8) = {8};
Plane Surface(9) = {9};

Extrude {0, 0, 2} {
  Surface{1}; Surface{2}; Surface{3}; Surface{7}; Surface{8}; Surface{9}; Layers{30};
}

// rebar tension right
Physical Volume(1) = {2};

// rebar tension left
Physical Volume(2) = {3};

// rebar tension center
Physical Volume(3) = {4};

// rebar compression left
Physical Volume(4) = {5};

// rebar compression right
Physical Volume(5) = {6};

// concrete
Physical Volume(10) = {1};

// bottom
Physical Surface(1) = {15};
