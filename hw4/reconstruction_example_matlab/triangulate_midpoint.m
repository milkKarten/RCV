
function outpoint = triangulate_midpoint(pl,pr, Rlr,tlr);
% outpoint = triangulate_midpoint(rayleft,rayright,R,t);
% Triangulate two rays for 3d reconstruction 
plt = pl;
prt = pr;
% take cross product, to find smallest segment between rays

q = cross(plt,Rlr*prt);
q = q./norm(q); % normalize it


% Find the scalars a,b,c from this equation
% a (plt  + c (q) = b ( Rlr prt ) + Tlr
% Solve 3 equations, 3 unknows, exact solution
A = [plt  (-Rlr*prt) q];


solveit = inv(A)*tlr;
a = solveit(1);b = solveit(2);c = solveit(3);


% 3D point is a*plt + c*0.5*q

outpoint = a*plt + c*0.5*q





