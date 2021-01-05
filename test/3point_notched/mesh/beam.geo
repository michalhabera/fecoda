SetFactory("Built-in");
d = 0.102;
h = 0.102;
l = 0.838;
s = 0.838-0.05;

nw = scale;
nh = 0.5*h;

Point(1) = {0, 0, 0};
Point(2) = {d, 0, 0};
Point(3) = {d, 0, h};
Point(4) = {0, 0, h};

Point(5) = {0, l, 0};
Point(6) = {d, l, 0};
Point(7) = {d, l, h};
Point(8) = {0, l, h};

Point(9) = {0, (l-nw)/2, 0};
Point(10) = {0, (l+nw)/2, 0};
Point(11) = {d, (l-nw)/2, 0};
Point(12) = {d, (l+nw)/2, 0};

Point(13) = {0, (l-nw)/2, nh};
Point(14) = {0, (l+nw)/2, nh};
Point(15) = {d, (l-nw)/2, nh};
Point(16) = {d, (l+nw)/2, nh};

Point(19) = {0, (l-s)/2, 0};
Point(20) = {d, (l-s)/2, 0};

Point(21) = {0, (l-s)/2+s, 0};
Point(22) = {d, (l-s)/2+s, 0};

Point(23) = {0, l/2, h};
Point(24) = {d, l/2, h};


Field[1] = Box;
//+
Field[1].XMax = d;
Field[1].XMin = 0;
Field[1].ZMax = d;
Field[1].ZMin = d/4;
Field[1].Thickness = 0.1;
Background Field = 1;

Field[1].VIn = scale;
Field[1].VOut = 0.1;

Field[1].YMax = l/2+0.02;
Field[1].YMin = l/2-0.02;
//+
Line(1) = {5, 21};
//+
Line(2) = {21, 10};
//+
Line(3) = {10, 14};
//+
Line(4) = {14, 13};
//+
Line(5) = {13, 9};
//+
Line(6) = {9, 19};
//+
Line(7) = {19, 1};
//+
Line(8) = {1, 4};
//+
Line(9) = {4, 23};
//+
Line(10) = {23, 8};
//+
Line(11) = {8, 5};
//+
Line(12) = {6, 22};
//+
Line(13) = {22, 12};
//+
Line(14) = {12, 16};
//+
Line(15) = {16, 15};
//+
Line(16) = {15, 11};
//+
Line(17) = {11, 20};
//+
Line(18) = {20, 2};
//+
Line(19) = {2, 3};
//+
Line(20) = {3, 24};
//+
Line(21) = {24, 7};
//+
Line(22) = {7, 6};
//+
Line(23) = {6, 5};
//+
Line(24) = {21, 22};
//+
Line(25) = {8, 7};
//+
Line(26) = {24, 23};
//+
Line(27) = {16, 14};
//+
Line(28) = {13, 15};
//+
Line(29) = {19, 20};
//+
Line(30) = {2, 1};
//+
Line(31) = {4, 3};
//+
Curve Loop(1) = {1, 24, -12, 23};
//+
Plane Surface(1) = {1};
//+
Line(32) = {10, 12};
//+
Line(33) = {11, 9};
//+
Curve Loop(2) = {24, 13, -32, -2};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3, -27, -14, -32};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {4, 28, -15, 27};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {5, -33, -16, -28};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {6, 29, -17, 33};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {7, -30, -18, -29};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {30, 8, 31, -19};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {9, -26, -20, -31};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {10, 25, -21, 26};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {11, -23, -22, -25};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18};
//+
Plane Surface(13) = {13};
//+
Surface Loop(1) = {2, 1, 12, 11, 13, 8, 7, 6, 5, 4, 3, 9, 10};
//+
Volume(1) = {1};
//+
Physical Volume("1") = {1};
//+
Physical Curve("2") = {24};
//+
Physical Curve("3") = {29};
//+
Physical Curve("4") = {26};
