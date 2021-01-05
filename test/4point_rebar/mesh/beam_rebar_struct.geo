//+
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 0.3, 0.3, 0};
//+
Circle(5) = {0.2, 0.2, 0, 0.01, 0, 2*Pi};
Circle(6) = {0.1, 0.2, 0, 0.01, 0, 2*Pi};

//+
Curve Loop(2) = {5};
//+
Plane Surface(2) = {2};

//+
Curve Loop(3) = {6};
//+
Plane Surface(3) = {3};

//+
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
//+
Curve Loop(1) = {5};
//+
Surface(2) = {1};
//+


// Extrude {0, 0, 2} {
//  Surface{1}; Surface{2}; Layers{20};
// }

// rebar
// Physical Volume(1) = {2};

// concrete
// Physical Volume(2) = {1};

// Physical Surface(1) = {3};
// Physical Surface(2) = {6};
