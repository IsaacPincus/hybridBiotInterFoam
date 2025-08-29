#%%
#import the class MeshVisu, numpy library and getVolumes function
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
# path to the simulation to load
path = './pillarsSnappySolids'

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

def get_boundaries(binary_image):
    # 4-connectivity (only horizontal and vertical neighbors)
    boundaries = skimage.segmentation.find_boundaries(binary_image.astype(int), 
                                                   connectivity=1,  # 4-connectivity
                                                   mode='inner')
    # 
    # # 8-connectivity (includes diagonal neighbors)
    # boundaries = skimage.segmentation.find_boundaries(binary_image.astype(int), 
    #                                                connectivity=2,  # 8-connectivity  
    #                                                mode='inner')

    boundary_coords = np.column_stack(np.where(boundaries))
    
    return boundary_coords

# some vibe coding to get the neighbour lists
def read_openfoam_polymesh(mesh_dir):
    """
    Read an OpenFOAM polyMesh directory and extract cell information.
    
    Args:
        mesh_dir (str): Path to the polyMesh directory
        
    Returns:
        dict: Dictionary containing:
            - 'cells': list of cell data, each with 'centroid' and 'neighbors'
            - 'num_cells': total number of cells
            - 'points': array of mesh points
            - 'faces': list of face definitions
    """
    
    def parse_openfoam_list(content, data_type=float):
        """Parse OpenFOAM list format and return numpy array or list."""
        # Remove comments and extra whitespace
        lines = [line.split('//')[0].strip() for line in content.split('\n')]
        content_clean = ' '.join(lines)
        
        # Find the list size
        size_match = re.search(r'(\d+)\s*\(', content_clean)
        if not size_match:
            raise ValueError("Could not find list size")
        
        size = int(size_match.group(1))
        
        # Extract data between parentheses
        start_paren = content_clean.find('(')
        end_paren = content_clean.rfind(')')
        
        if start_paren == -1 or end_paren == -1:
            raise ValueError("Could not find list data")
        
        data_str = content_clean[start_paren+1:end_paren].strip()
        
        if data_type == float:
            # For points (vectors)
            if '(' in data_str and ')' in data_str:
                # Vector data like (x y z)
                vectors = re.findall(r'\((.*?)\)', data_str)
                result = []
                for vec_str in vectors:
                    coords = [float(x) for x in vec_str.split()]
                    result.append(coords)
                return np.array(result)
            else:
                # Scalar data
                values = [float(x) for x in data_str.split()]
                return np.array(values)
        else:
            # For faces and cells (lists of integers)
            if '(' in data_str and ')' in data_str:
                # Lists like 4(0 1 2 3)
                lists = re.findall(r'(\d+)\s*\((.*?)\)', data_str)
                result = []
                for count_str, indices_str in lists:
                    indices = [int(x) for x in indices_str.split()]
                    result.append(indices)
                return result
            else:
                # Simple integer list
                values = [int(x) for x in data_str.split()]
                return values

    # Read points file
    points_file = os.path.join(mesh_dir, 'points')
    with open(points_file, 'r') as f:
        points_content = f.read()
    points = parse_openfoam_list(points_content, data_type=float)
    
    # Read faces file
    faces_file = os.path.join(mesh_dir, 'faces')
    with open(faces_file, 'r') as f:
        faces_content = f.read()
    faces = parse_openfoam_list(faces_content, data_type=int)
    
    # Read owner file
    owner_file = os.path.join(mesh_dir, 'owner')
    with open(owner_file, 'r') as f:
        owner_content = f.read()
    owner = parse_openfoam_list(owner_content, data_type=int)
    
    # Read neighbour file
    neighbour_file = os.path.join(mesh_dir, 'neighbour')
    with open(neighbour_file, 'r') as f:
        neighbour_content = f.read()
    neighbour = parse_openfoam_list(neighbour_content, data_type=int)
    
    # Determine number of cells
    num_cells = max(max(owner), max(neighbour) if neighbour else 0) + 1
    
    # Create cell-face connectivity
    cell_faces = defaultdict(list)
    for face_id, cell_id in enumerate(owner):
        cell_faces[cell_id].append(face_id)
    
    for face_id, cell_id in enumerate(neighbour):
        cell_faces[cell_id].append(face_id)
    
    # Create cell-neighbor connectivity
    cell_neighbors = defaultdict(set)
    for face_id, owner_cell in enumerate(owner):
        if face_id < len(neighbour):  # Internal face
            neighbor_cell = neighbour[face_id]
            cell_neighbors[owner_cell].add(neighbor_cell)
            cell_neighbors[neighbor_cell].add(owner_cell)
    
    # Calculate cell centroids
    def calculate_cell_centroid(cell_id):
        """Calculate centroid of a cell using its faces."""
        face_ids = cell_faces[cell_id]
        cell_points = []
        
        for face_id in face_ids:
            face_point_ids = faces[face_id]
            for point_id in face_point_ids:
                cell_points.append(points[point_id])
        
        if cell_points:
            cell_points = np.array(cell_points)
            # Simple average of all points (approximation)
            centroid = np.mean(cell_points, axis=0)
        else:
            centroid = np.array([0.0, 0.0, 0.0])
        
        return centroid
    
    # Build result structure
    cells = []
    for cell_id in range(num_cells):
        centroid = calculate_cell_centroid(cell_id)
        neighbors = list(cell_neighbors[cell_id])
        
        cell_data = {
            'id': cell_id,
            'centroid': {
                'x': float(centroid[0]),
                'y': float(centroid[1]),
                'z': float(centroid[2])
            },
            'neighbors': neighbors
        }
        cells.append(cell_data)
    
    result = {
        'cells': cells,
        'num_cells': num_cells,
        'points': points,
        'faces': faces,
        'owner': owner,
        'neighbour': neighbour
    }
    
    return result


## constants, setting up initial conditions etc
pathVelocity = './pillarsSolidsVelocity'
pathElastic = './pillarsSolidsElastic'
pathTransport = './scalarTransport'

# ## constants, setting up initial conditions etc
# pathVelocity = '/mnt/c/Data/Biofilm_Sims/Aug29_High_conc/pillarsSolidsVelocity'
# pathElastic = '/mnt/c/Data/Biofilm_Sims/Aug29_High_conc/pillarsSolidsElastic'
# pathTransport = '/mnt/c/Data/Biofilm_Sims/Aug29_High_conc/scalarTransport'
# imagePath = '/mnt/c/Data/Biofilm_Sims/Aug29_High_conc/ManyCylindersOld.png'

show_plots = True
rng = np.random.default_rng()

# yield stresses, [N/m^2]
sigAlive = 4000
sigDead = 40

# Read image
mask, openfoam_image_field, Nx, Ny = image_to_openfoam_mask(image_path=imagePath, method='mean', invert=False)

# Nx = 360
# Ny = 90
L = 2e-3
W = Ny/Nx * L
dx = L / Nx
dy = W / Ny
thickness = 1e-5

X, Y = np.meshgrid(np.linspace(0,L,Nx), np.linspace(0,W,Ny))

U_threshold = 1e-6

dt_elastic = 1e-2
dt_velocity = 1e-2
dt_transport = 5

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

caseElastic = FoamCase(pathElastic) # Loads the OpenFOAM case
caseVelocity = FoamCase(pathVelocity) # Loads the OpenFOAM case
caseTransport = FoamCase(pathTransport) # Loads the OpenFOAM case

#%%
###############################################################################################################
# YIELD STRESS SOLUTION
###############################################################################################################

with caseElastic[-1]["sigmaEq"] as field:
    sigEqField = field.internal_field
with caseTransport[-1]["epss"] as field: # read epss from file
    epssField = field.internal_field
with caseTransport[-1]["B"] as field: #read B to file
    BField = field.internal_field
with caseTransport[-1]["Bdead"] as field: # read Bdead to file
    BdeadField = field.internal_field

# the epss field is anywhere there is significant amounts of biofilm
epss2D = to_2d(epssField, image_cell_values)
Balive2D = to_2d(BField, image_cell_values)
Bdead2D = to_2d(BdeadField, image_cell_values)

# negative when we have no solid, positive when we have solid, form required for Eikonal solver
solidIndicator2D = ( epss2D > 0.05 ) - 0.5
solidIndicator2DMasked = np.ma.masked_array(solidIndicator2D, ~mask)
sigEq2D = to_2d(sigEqField, image_cell_values)
yieldStress = ((Balive2D * sigAlive + Bdead2D * sigDead)/(Balive2D + Bdead2D))\
# find all points in domain where Von Mises stress is greater than yield stress
fieldToRemove = sigEq2D > yieldStress

if show_plots:
    plt.title('Travel time from the boundary with an obstacle')
    plt.contour(X, Y, solidIndicator2DMasked, [0], linewidths=(3), colors='black')
    plt.contour(X, Y, solidIndicator2DMasked.mask, [0], linewidths=(3), colors='red')
    plt.contour(X, Y, t, np.linspace(0,1000,10))
    plt.colorbar()
    plt.savefig('2d_phi_travel_time_mask.png')
    plt.show()

# in places we want to remove, set B to zero
Balive2DTemp = Balive2D
Bdead2DTemp = Bdead2D
Balive2DTemp[fieldToRemove] = 0
Bdead2DTemp[fieldToRemove] = 0
# now go through and find regions with completely detached biofilm, no part touching grains
maskLabel = np.float64(~mask)
totalBplusMask = maskLabel + Balive2DTemp + Bdead2DTemp
totalBplusMask[np.isnan(totalBplusMask)] = 1.0
labeled_image = skimage.measure.label(totalBplusMask>0, connectivity=2)
if show_plots:
    plt.title('labeled image')
    plt.imshow(labeled_image, cmap='viridis', aspect='auto')
    plt.colorbar(label='Labels of each region')
    plt.show()
for i in range(1,labeled_image.max()+1):
    # get just the parts of the image with this value
    comp = labeled_image==i
    # check if there's any overlap with the mask
    doesOverlap = np.any(comp&~mask)
    if ~doesOverlap:
        fieldToRemove[comp] = True
if show_plots:
    plt.title('area to remove')
    plt.imshow(fieldToRemove, cmap='viridis', aspect='auto')
    plt.colorbar(label='true or false')
    plt.show()
# in places we want to remove, set B to zero
Balive2DTemp[fieldToRemove] = 0
Bdead2DTemp[fieldToRemove] = 0

# write epss and B fields
epss2DTemp = 0.1 * (Balive2DTemp + Bdead2DTemp > 0.1) + 1e-5
epssFieldTemp = to_1d(epss2DTemp, image_cell_values)
BFieldTemp = to_1d(Balive2DTemp, image_cell_values)
BdeadFieldTemp = to_1d(Bdead2DTemp, image_cell_values)

####### first calculate velocities #################
# make sure that we set epss back to the uniform value, then re-write
with caseVelocity[-1]["epss"] as field: # write epss to file
    field.internal_field = 0
with caseVelocity[-1]["epss"] as field: # write epss to file
    field.internal_field = epssFieldTemp

# now run to find the velocity throughout the medium
caseVelocity.run("elasticHBPF_noepss") # Run the case itself
# update the run time for the next step
with caseVelocity.control_dict as f:
    f["endTime"] = f["endTime"] + dt_velocity

######## Then calculate the scalar transport rate ##############
# read velocity from previous calculation
with caseVelocity[-1]["U"] as field:
    UfieldTemp = field.internal_field

# now we actually remove the regions which are both yielded and where the velocity is above some threshold
# this means that we only remove parts if they can actually be taken away by the flow
U2DTemp = to_2d(UfieldTemp, image_cell_values)
fieldToRemove = fieldToRemove & (U2DTemp > U_threshold)

# in places we want to remove, set B to zero
Balive2D[fieldToRemove] = 0
Bdead2D[fieldToRemove] = 0

# write epss and B fields
epss2D = 0.1 * (Balive2D + Bdead2D > 0.1) + 1e-5
epssField = to_1d(epss2D, image_cell_values)
BField = to_1d(Balive2D, image_cell_values)
BdeadField = to_1d(Bdead2D, image_cell_values)

if show_plots:
    plt.title('area to remove')
    plt.imshow(fieldToRemove, cmap='viridis', aspect='auto')
    plt.colorbar(label='true or false')
    plt.show()
