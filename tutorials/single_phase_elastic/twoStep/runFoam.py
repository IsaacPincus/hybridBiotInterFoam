#%%
from foamlib import FoamCase
from fluidfoam import MeshVisu, readmesh, readvector
from fluidfoam.readof import getVolumes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from PIL import Image
import skimage
import skfmm
import os
import subprocess
import shutil

def create_mapping_arrays(Nx, Ny):
    """
    Create lookup arrays for fast pixel-cell conversion.
    
    Args:
        Nx (int): number of cells in x-direction
        Ny (int): number of cells in y-direction
        
    Returns:
        tuple: (pixel_to_cell_map, cell_to_pixel_map)
    """
    # Create 2D array: pixel_to_cell_map[j, i] = cell_id
    pixel_to_cell_map = np.zeros((Ny, Nx), dtype=int)
    
    # Create arrays: cell_to_pixel_map[cell_id] = (i, j)
    cell_to_pixel_i = np.zeros(Nx * Ny, dtype=int)
    cell_to_pixel_j = np.zeros(Nx * Ny, dtype=int)
    
    for j in range(Ny):
        for i in range(Nx):
            cell_id = i + j * Nx
            pixel_to_cell_map[j, i] = cell_id
            cell_to_pixel_i[cell_id] = i
            cell_to_pixel_j[cell_id] = j
    
    return pixel_to_cell_map, (cell_to_pixel_i, cell_to_pixel_j)

def image_to_openfoam_mask(image_path, Nx=None, Ny=None, method='threshold', 
                          threshold=128, invert=False):
    """
    Convert image to OpenFOAM boolean field.
    
    Returns:
        tuple: (mask_2d, openfoam_field, Nx, Ny)
    """
    # Read image
    image = Image.open(image_path).convert('L')
    
    # Resize if specified
    if Nx and Ny:
        image = image.resize((Nx, Ny))
    else:
        Nx, Ny = image.size
    
    image_array = np.array(image)
    
    # Create mask based on method
    if method == 'threshold':
        thresh_val = threshold
    elif method == 'mean':
        thresh_val = np.mean(image_array)
    elif method == 'otsu':
        from skimage.filters import threshold_otsu
        thresh_val = threshold_otsu(image_array)
    
    mask_2d = image_array > thresh_val
    
    if invert:
        mask_2d = ~mask_2d
    
    # Convert to OpenFOAM field (as integers: 1=True, 0=False)
    openfoam_field = np.zeros(Nx * Ny, dtype=int)
    
    for j in range(Ny):
        for i in range(Nx):
            cell_id = i + j * Nx
            openfoam_field[cell_id] = int(mask_2d[j, i])
    
    print(f"Image size: {Nx} x {Ny}")
    print(f"Threshold: {thresh_val:.1f}")
    print(f"True cells: {np.sum(openfoam_field)} / {len(openfoam_field)}")
    
    return mask_2d, openfoam_field, Nx, Ny

def create_blockmesh_dict(Nx, Ny, L, W, thickness=0.1, output_dir="system"):
    """
    Create blockMeshDict for a 2D rectangular mesh.
    
    Args:
        Nx (int): Number of cells in x-direction (length)
        Ny (int): Number of cells in y-direction (width)  
        L (float): Length in x-direction
        W (float): Width in y-direction
        thickness (float): Thickness in z-direction (for 2D case)
        output_dir (str): Directory to write blockMeshDict
        
    Returns:
        str: Content of blockMeshDict file
    """
    
    # Calculate cell sizes
    dx = L / Nx
    dy = W / Ny
    
    # Define vertices (8 vertices for a hexahedral block)
    # Bottom face (z = 0)
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = L, 0.0, 0.0
    x2, y2, z2 = L, W, 0.0
    x3, y3, z3 = 0.0, W, 0.0
    
    # Top face (z = thickness)
    x4, y4, z4 = 0.0, 0.0, thickness
    x5, y5, z5 = L, 0.0, thickness
    x6, y6, z6 = L, W, thickness
    x7, y7, z7 = 0.0, W, thickness
    
    blockmesh_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v7                                    |
|   \\\\  /    A nd           | Website:  www.openfoam.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Mesh parameters
// Length (x): {L} m, Width (y): {W} m, Thickness (z): {thickness} m
// Cells: {Nx} x {Ny} x 1
// Cell size: dx = {dx:.6f} m, dy = {dy:.6f} m

scale   1;

vertices
(
    ({x0} {y0} {z0})    // vertex 0: origin bottom
    ({x1} {y1} {z1})    // vertex 1: +x bottom  
    ({x2} {y2} {z2})    // vertex 2: +x+y bottom
    ({x3} {y3} {z3})    // vertex 3: +y bottom
    ({x4} {y4} {z4})    // vertex 4: origin top
    ({x5} {y5} {z5})    // vertex 5: +x top
    ({x6} {y6} {z6})    // vertex 6: +x+y top  
    ({x7} {y7} {z7})    // vertex 7: +y top
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({Nx} {Ny} 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    walls
    {{
        type wall;
        faces
        (
            (3 7 6 2)
            (1 5 4 0)
        );
    }}
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (2 6 5 1)
        );
    }}
    emptyFaces
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write file
    output_path = os.path.join(output_dir, "blockMeshDict")
    with open(output_path, 'w') as f:
        f.write(blockmesh_content)
    
    print(f"blockMeshDict written to: {output_path}")
    print(f"Mesh: {Nx} x {Ny} cells, Domain: {L} x {W} m")
    print(f"Cell size: {dx:.6f} x {dy:.6f} m")
    
    return blockmesh_content

def create_topoSetDict(cell_ids, output_path="system/topoSetDict"):
    """
    Generates an OpenFOAM topoSetDict file to create a cellSet from a list of cell IDs.

    Args:
        cell_ids (list of int): A list of the cell IDs to include in the set.
        output_path (str): The full path for the output file.
                           Defaults to 'system/topoSetDict' for a standard case structure.
    """
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Convert the list of cell IDs into a single space-separated string
    cell_string = ' '.join(map(str, cell_ids))

    # Use an f-string to create the file content
    file_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
    {{
        name    grainCells;
        type    cellSet;
        action  new;
        source  labelToCell;
        sourceInfo
        {{
            value ({cell_string});
        }}
    }}
);

// ************************************************************************* //
"""

    # Write the content to the specified file
    try:
        with open(output_path, 'w') as f:
            f.write(file_content)
        print(f"✅ Successfully created topoSetDict at: {output_path}")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")

def to_1d(values_2d, index_map):
    flat_values = values_2d[index_map >= 0]
    flat_indices = index_map[index_map >= 0]
    
    max_index = flat_indices.max()
    array_1d = np.empty(max_index + 1, dtype=values_2d.dtype)
    array_1d[flat_indices] = flat_values
    return array_1d

def to_2d(array_1d, index_map):
    values_2d = np.full(index_map.shape, fill_value=np.nan, dtype=array_1d.dtype)
    valid = index_map >= 0
    values_2d[valid] = array_1d[index_map[valid]]
    return values_2d


## constants, setting up initial conditions etc
pathVelocity = './pillarsSolidsVelocity'
pathElastic = './pillarsSolidsElastic'
pathTransport = './scalarTransport'

# Read image
mask, openfoam_image_field, Nx, Ny = image_to_openfoam_mask(image_path='ManyCylinders.png', method='mean', invert=False)

# Nx = 360
# Ny = 90
L = 2e-3
W = Ny/Nx * L
dx = L / Nx
dy = W / Ny
thickness = 1e-5

# yield stresses, [N/m^2]
sigAlive = 4000
sigDead = 40
# detachment rate constant, [m/s]
kdet = 1/3600

mu = 1.95e-4       # maximum growth rate for biofilm, [1/s]
Ks = 0.1           # Monod half-saturation constant, [mol/m^3]
Yield = 0.15       # Yield of biomass
kd = 2e-6          # biomass decay constant, [1/s]

# timestep for each removal/growth step, in seconds
dt = 600
no_steps = 3

dt_elastic = 1e-2
dt_velocity = 1e-2
dt_transport = 2

Bcut = 6010

#%%
###############################################################################################################
#
#
#
#
# START OF VELOCITY AND STRESS CALCULATION SCRIPT, INITIAL SETUP
#
#
#
#
###############################################################################################################

caseElastic = FoamCase(pathElastic) # Loads the OpenFOAM case
#caseVelocity[0]["U"].internal_field = [0, 0, 0] # Set the initial velocity field to zero
caseElastic.clean() # clean up case, runs 'clean' in directory

output = create_blockmesh_dict(Nx, Ny, L, W, thickness=thickness, output_dir=os.path.join(pathElastic, "system"))

caseElastic.run("blockMesh") # do blockMesh

# make sure that we set epss back to the uniform value
with caseElastic[0]["epss"] as field: # write epss to file
    field.internal_field = 0

# make sure that we set U back to the uniform value
with caseElastic[0]["U"] as field: # write epss to file
    field.internal_field = [0, 0, 0]

# make sure that we set p back to the uniform value
with caseElastic[0]["p"] as field: # write epss to file
    field.internal_field = 0

# now find all indices which are true, and create a topoSetDict for them
nonzeros = np.nonzero(openfoam_image_field)
cell_id_grains = nonzeros[0]

create_topoSetDict(cell_id_grains.tolist(), output_path=os.path.join(pathElastic, "system/topoSetDict"))

caseElastic.run("topoSet")
caseElastic.run("subsetMesh grainCells -overwrite -patch walls")

# now we need to re-associate each cell with the pixels in the original image. 
# we want a list of cells, each with the i,j location of the pixel in the image, which maps to the mask location
# we also want a 2D image with either the cell #, or -1 for a grain location.
x, y, z = readmesh(pathElastic)
# Ensure x, y are NumPy arrays
x = np.asarray(x)
y = np.asarray(y)
# Compute image indices
xn = (x / dx).astype(int)
yn = (y / dy).astype(int)
# Initialize output arrays
image_cell_values = np.array(mask, dtype=int) - 1
cell_locations_in_image = np.stack((xn, yn), axis=1)
# Assign index i to each corresponding (xn, yn)
# Note: This assumes no duplicate (xn, yn) pairs — last one wins if duplicates exist.
image_cell_values[yn, xn] = np.arange(len(x))
img_array = np.asarray(image_cell_values)
# Optional: normalize to [0, 255] if values exceed 255
if img_array.max() > 255 or img_array.min() < 0:
    img_array = 255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())
    img_array = img_array.astype(np.uint8)
else:
    img_array = img_array.astype(np.uint8)
# Convert to PIL Image and save
img = Image.fromarray(img_array, mode='L')  # 'L' = 8-bit grayscale
img.save("image_cell_values.png")

# now, we want to write some other fields. We want each pixel to have a value of Balive and Bdead, for the amount of biofilm there.
# note that meshgrid in numpy is opposite of matlab, it's (ny,nx) array indexing, like an image
X, Y = np.meshgrid(np.linspace(0,L,Nx), np.linspace(0,W,Ny))
# Parameters
r0 = W/5
x0 = 3*L/5
y0 = W/2
Bvalue = 6000.0
Balive2D = np.zeros_like(mask, dtype=float)
Bdead2D = np.zeros_like(mask, dtype=float)
# put a patch of alive biofilm at a circle
# Balive2D[(X - x0)**2 + (Y - y0)**2 < r0**2] = Bvalue
# put a patch of alive biofilm at the top and bottom
condition = ((Y < W/3) | (Y > 2*W/3)) & (X > L/5)
Balive2D[condition] = Bvalue

# now also setup the other systems
caseVelocity = FoamCase(pathVelocity) # Loads the OpenFOAM case
caseVelocity.clean()
shutil.copytree(os.path.join(pathElastic, "constant/polyMesh"), os.path.join(pathVelocity, "constant/polyMesh"), dirs_exist_ok=True)

caseTransport = FoamCase(pathTransport) # Loads the OpenFOAM case
caseTransport.clean()
shutil.copytree(os.path.join(pathVelocity, "constant/polyMesh"), os.path.join(pathTransport, "constant/polyMesh"), dirs_exist_ok=True)

#%%
# start main loop here
for step in range(no_steps):

    ############################### Solve elastic EQN #########################
    with caseElastic[-1]["epss"] as field: # write epss to file
        field.internal_field = epssField

    caseElastic.run("elasticHBPF_noepss") # Run the case itself
    # update the run time for the next step
    with caseElastic.control_dict as f:
        f["endTime"] = f["endTime"] + dt_elastic

    #%%
    ###############################################################################################################
    # EIKONAL EQUATION SOLUTION
    ###############################################################################################################

    # the epss field is anywhere there is significant amounts of biofilm
    epss2D = 0.1 * (Balive2D + Bdead2D > 0) + 1e-5
    epssField = to_1d(epss2D, image_cell_values)
    BField = to_1d(Balive2D, image_cell_values)

    with caseElastic[-1]["sigmaEq"] as field:
        sigEqField = field.internal_field

    # negative when we have no solid, positive when we have solid, form required for Eikonal solver
    solidIndicator2D = ( epss2D > 0.05 ) - 0.5
    solidIndicator2DMasked = np.ma.masked_array(solidIndicator2D, ~mask)

    sigEq2D = to_2d(sigEqField, image_cell_values)
    F = kdet * sigEq2D/((Balive2D * sigAlive + Bdead2D * sigDead)/(Balive2D + Bdead2D))
    F[np.isnan(F)] = 1e10
    # F = np.ones_like(mask, dtype=float)

    t = skfmm.travel_time(solidIndicator2DMasked, F, dx)

    plt.title('Travel time from the boundary with an obstacle')
    plt.contour(X, Y, solidIndicator2DMasked, [0], linewidths=(3), colors='black')
    plt.contour(X, Y, solidIndicator2DMasked.mask, [0], linewidths=(3), colors='red')
    plt.contour(X, Y, t, np.linspace(0,1000,10))
    plt.colorbar()
    plt.savefig('2d_phi_travel_time_mask.png')
    plt.show()

    # find all points in domain where t < dt
    fieldToRemove = t < dt

    # plt.title('Field to Remove')
    # plt.imshow(fieldToRemove, cmap='viridis', aspect='auto')
    # plt.colorbar(label='Boolean values')
    # plt.show()
    # # plt.savefig('2d_phi_travel_time_mask.png')
    # plt.show()

    # in places we want to remove, set B to zero
    Balive2D[fieldToRemove] = 0
    Bdead2D[fieldToRemove] = 0
    # write epss and B fields
    epss2D = 0.1 * (Balive2D + Bdead2D > 0.1) + 1e-5
    epssField = to_1d(epss2D, image_cell_values)
    BField = to_1d(Balive2D, image_cell_values)
    BdeadField = to_1d(Bdead2D, image_cell_values)

    #%%
    ###############################################################################################################
    # SCALAR TRANSPORT SOLVER
    ###############################################################################################################

    ####### first calculate velocities #################
    # make sure that we set epss back to the uniform value, then re-write
    with caseVelocity[-1]["epss"] as field: # write epss to file
        field.internal_field = 0
    with caseVelocity[-1]["epss"] as field: # write epss to file
        field.internal_field = epssField

    # now run to find the velocity throughout the medium
    caseVelocity.run("elasticHBPF_noepss") # Run the case itself
    # update the run time for the next step
    with caseVelocity.control_dict as f:
        f["endTime"] = f["endTime"] + dt_elastic

    ######## Then calculate the scalar transport rate ##############
    # read velocity from previous calculation
    with caseVelocity[-1]["U"] as field:
        Ufield = field.internal_field

    # write internal fields
    with caseTransport[-1]["epss"] as field: # write epss to file
        field.internal_field = epssField
    with caseTransport[-1]["B"] as field: # write B to file
        field.internal_field = BField
    with caseTransport[-1]["U"] as field: # write velocity
        field.internal_field = Ufield

    caseTransport.run("convectionFoam")
    # update the run time for the next step
    with caseTransport.control_dict as f:
        f["endTime"] = f["endTime"] + dt_elastic

    # %%
    ###############################################################################################################
    # START OF BIOFILM GROWTH MODEL
    ###############################################################################################################

    with caseTransport[-1]["C"] as field: # read concentration field
        CField = field.internal_field

    # for each element of the B field, update based on growth
    solidIndicator2D = ( epss2D > 0.05 ) - 0.5
    solidIndicator2DMasked = np.ma.masked_array(solidIndicator2D, ~mask)
    t = skfmm.travel_time(solidIndicator2DMasked, np.ones_like(solidIndicator2D))

    prefactor = (mu*(CField/(Ks + CField)) - kd)
    BdeadField = BdeadField + kd * BField / prefactor * (np.exp(dt*prefactor) - 1)
    BField = BField * np.exp(prefactor * dt)

    Balive2D = to_2d(BField, image_cell_values)
    Bdead2D = to_2d(BdeadField, image_cell_values)

    # find all cells with values of Balive greater than some cutoff
    # Go through each cell which is growing. 
    # If there's an empty cell adjacent, choose one at random and displace half the biomass there.
    # If there's no empty cell, displace that biomass to a random adjacent cell, and restart
    # here, have some vibe coding, thanks Claude
    Bdivide = (Balive2D > Bcut) & (mask)
    while np.any(Bdivide):
        Bdivide = (Balive2D > Bcut) & (mask)
        print(np.sum(Bdivide))
        target_coords = np.column_stack(np.where(Bdivide))

        # Neighbor offsets (4-connectivity)
        offsets = np.array([[-1,0],  [0,-1], 
                            [0,1],   [1,0]])

        results = []
        for i, j in target_coords:
            # cut the current value in half
            Balive2D[i,j] = Balive2D[i,j]/2

            # Calculate all neighbor positions
            neighbor_coords = np.array([i, j]) + offsets
            
            # Filter valid neighbors (in bounds and not masked)
            valid_mask = ((neighbor_coords[:, 0] >= 0) & 
                            (neighbor_coords[:, 0] < Balive2D.shape[0]) &
                            (neighbor_coords[:, 1] >= 0) & 
                            (neighbor_coords[:, 1] < Balive2D.shape[1]))
            
            valid_neighbors = neighbor_coords[valid_mask]
            
            # Remove masked neighbors
            unmasked = mask[valid_neighbors[:, 0], valid_neighbors[:, 1]]
            final_neighbors = valid_neighbors[unmasked]

            B_neighbors = Balive2D[final_neighbors[:, 0], final_neighbors[:, 1]]
            B_neighbors_zero = B_neighbors < 0.1

            # Filter to only neighbors that satisfy the condition
            valid_indices = np.where(B_neighbors_zero)[0]

            if len(valid_indices) > 0:
                # move half the biomass to an adjacent cell
                random_valid_idx = np.random.choice(valid_indices)
                random_coord = final_neighbors[random_valid_idx]
                random_value = Balive2D[random_coord[0], random_coord[1]]
                Balive2D[random_coord[0], random_coord[1]] = Balive2D[i,j]/2  # Replace with your desired value
                # Or for example:
                # Balive2D[random_coord[0], random_coord[1]] += 100
                # Balive2D[random_coord[0], random_coord[1]] *= 1.5
            else:
                # find the adjacent index with the smallest t, propogate half of the biofilm to that cell
                t_vals = t[final_neighbors[:, 0], final_neighbors[:, 1]]
                t_min_idx = np.argmin(t_vals)
                coords_min_t = final_neighbors[t_min_idx]
                Balive2D[coords_min_t[0], coords_min_t[1]] = Balive2D[coords_min_t[0], coords_min_t[1]] + Balive2D[i,j]/2

    plt.title('B value')
    plt.imshow(Balive2D, cmap='viridis', aspect='auto')
    plt.colorbar()
    # plt.contour(X, Y, Balive2D, 50)
    # plt.savefig('2d_phi_travel_time_mask.png')
    plt.show()

    plt.title('Bdead value')
    plt.imshow(Bdead2D, cmap='viridis', aspect='auto')
    plt.colorbar()
    # plt.contour(X, Y, Balive2D, 50)
    # plt.savefig('2d_phi_travel_time_mask.png')
    plt.show()


    # %%
