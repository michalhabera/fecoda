SetFactory("OpenCASCADE");
Cylinder(1) = {0, 0, 0, 0, 0, H / 1000, D / 2 / 1000, 2*Pi};
Physical Surface(1) = {3};
Physical Surface(2) = {2};

Physical Volume(3) = {1};
