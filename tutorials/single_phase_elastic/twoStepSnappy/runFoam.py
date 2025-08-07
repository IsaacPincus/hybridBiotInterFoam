from foamlib import FoamCase
from fluidfoam import MeshVisu, readmesh, readvector
from fluidfoam.readof import getVolumes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

path = './pillarsSnappySolids'

caseVelocity = FoamCase(path) # Loads the OpenFOAM case
#caseVelocity[0]["U"].internal_field = [0, 0, 0] # Set the initial velocity field to zero
caseVelocity.clean() # clean up case, runs 'clean' in directory

# now get all the boundary cells, i.e. those next to a wall

caseVelocity.run() # The run file only sets up the mesh
caseVelocity.run("elasticHBPF") # Run the case itself
for time in caseVelocity: # Iterate over the time directories
    print(time.time) # Print the time