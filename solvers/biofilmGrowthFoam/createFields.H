Info<< "Reading field C\n" << endl;

volScalarField C
(
    IOobject
    (
        "C",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field U\n" << endl;

volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

Info<< "Reading solid fraction field 'epss'\n" << endl;
volScalarField epss
(
    IOobject
    (
        "epss",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading local biofilm concentration B\n" << endl;
volScalarField B
(
    IOobject
    (
        "B",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading local dead biofilm concentration Bd\n" << endl;
volScalarField Bd
(
    IOobject
    (
        "Bd",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading/Calculating fluid fraction field 'epsf'\n" << endl;
volScalarField epsf
(
    IOobject
    (
        "epsf",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE   
    ),
    (1-epss)
);


Info<< "Reading diffusivity 'Df'\n" << endl;
dimensionedScalar Df
(
    "Df",
    transportProperties.lookup("Df")
);

Info<< "Reading diffusivity tourtuosity coefficient n\n" << endl;
dimensionedScalar n
(
    "n",
    transportProperties.lookup("n")
);

Info<< "Reading biofilm growth rate mu\n" << endl;
dimensionedScalar mu
(
    "mu",
    transportProperties.lookup("mu")
);

Info<< "Reading biofilm death rate kd\n" << endl;
dimensionedScalar mu
(
    "kd",
    transportProperties.lookup("kd")
);

Info<< "Reading Monod half-saturation constant Ks\n" << endl;
dimensionedScalar Ks
(
    "Ks",
    transportProperties.lookup("Ks")
);

Info<< "Reading biomass yield rate Y\n" << endl;
dimensionedScalar Y
(
    "Y",
    transportProperties.lookup("Y")
);

Info<< "Calculating Corrected Solid-Phase Ion Diffusivity 'Dion = Df*(1-epss)^(n-1)'\n" << endl;
volScalarField Dion
(
    IOobject
    (
        "Dion",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    Df*pow((1-epss),(n-1))
);

// this calculates the surface fluxes fvc::flux(U)
#include "createPhi.H"

#include "createFvOptions.H"