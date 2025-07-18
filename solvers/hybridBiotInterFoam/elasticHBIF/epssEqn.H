//Update Solid Speed from Displacement Field
Us = fvc::ddt(D);
Us.correctBoundaryConditions();
phis = fvc::interpolate(Us) & mesh.Sf();  
Dsup = D*epss;

//Solving Solid Continuity Equation

Info<< "\nCalculating epss" << endl;
fvScalarMatrix epssEqn
(
	fvm::ddt(epss)
      + fvm::div(phis,epss)
);

epssEqn.solve();
epss.correctBoundaryConditions();

// I don't know why this is here... Doesn't let you set solid and fluid domains!
epss.min(0.99);
epss.max(0.01);

// epss.min(0.999);
// epss.max(0.0);
