/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    biofilmGrowthFoam

Description
    Solves the steady or transient transport equation for a scalar, and can include a reaction rate dependent constant

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating scalar transport\n" << endl;

    #include "CourantNo.H"

    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {

            // first solve convection equation with Monod kinetics consumption
            C.max(0.); 
            fvScalarMatrix ConvectionEqn //(C=Concentration=rho*wf/mw)
            (
                    fvm::ddt(epsf,C) 
                    + fvm::div(phi,C) 
                    - fvm::laplacian((Dion*epsf), C)
                    + fvm::Sp(mu*B/(Ks + C)/Y, C) // biomass consumption, Monod kinetics, implicit
                    // - mu*B*C/(Ks + C)/Y // biomass consumption, Monod kinetics
                    //+ fvm::div(phic,C) //solid movement correction

                    // // simplified version to test
                    // fvm::ddt(C) 
                    // + fvm::div(phi,C) 
                    // - fvm::laplacian(Dion,C)
            );
            ConvectionEqn.relax();
            ConvectionEqn.solve();
            // Update boundary conditions
            C.correctBoundaryConditions();

            // now solve for the biofilm growth
            fvScalarMatrix growthEquation //(C=Concentration=rho*wf/mw)
            (
                    fvm::ddt(B) 
                    - fvm::Sp(mu*B/(Ks + C)/Y, C) // biomass consumption, Monod kinetics, implicit
                    + fvm::Sp(kd, B) // biomass death rate
            );
            growthEquation.relax();
            growthEquation.solve();

            // now solve for the biofilm death
            fvScalarMatrix deathEquation //(C=Concentration=rho*wf/mw)
            (
                    fvm::ddt(Bd) 
                    - fvm::Sp(kd, B) // biomass death rate
            );
            deathEquation.relax();
            deathEquation.solve();

            // Dion=Df*pow((1-epss),(n-1));
            // epsf=1-epss;
        }

        runTime.write();
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
