#import the class MeshVisu, numpy library and getVolumes function
from foamlib import FoamCase
from fluidfoam import MeshVisu, readmesh, readvector
from fluidfoam.readof import getVolumes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from collections import defaultdict
import re
import skfmm
# path to the simulation to load
pathRead = './pillarsSnappySolids'
pathWrite = './scalarTransport'

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


# # Load mesh and create an object called myMesh
# # The box by default is equal to the mesh dimension
# myMesh = MeshVisu( path =  path)

timename = '0.0002'
vel = readvector(pathRead, timename, 'U')

U = np.array(vel)

mesh_data = read_openfoam_polymesh(os.path.join(pathRead,'constant/polyMesh/'))

caseVelocity = FoamCase(pathRead)

with caseVelocity[-1]["U"] as field:
    Ufield = field.internal_field
    print(Ufield[:,0:5])

with caseVelocity[-1]["epss"] as field:
    epssField = field.internal_field
    print(epssField[0:5])

with caseVelocity[-1]["phi"] as field:
    phiField = field.internal_field
    print(phiField[0:5])

with caseVelocity[-1]["sigmaEq"] as field:
    sigEqField = field.internal_field
    print(sigEqField[0:5])

caseTransport = FoamCase(pathWrite) # Loads the OpenFOAM case
#caseVelocity[0]["U"].internal_field = [0, 0, 0] # Set the initial velocity field to zero
caseTransport.clean() # clean up case, runs 'clean' in directory

# now get all the boundary cells, i.e. those next to a wall

with caseTransport[0]["U"] as field: # write velocity
    print(field.dimensions)
    print(field.boundary_field)
    field.internal_field = Ufield

with caseTransport[0]["phi"] as field: # write velocity transfer function
    print(field.dimensions)
    print(field.boundary_field)
    field.internal_field = phiField

with caseTransport[0]["B"] as field: # Load a field
    print(field.dimensions)
    print(field.boundary_field)
    field.internal_field = epssField*1e4

# caseTransport.run()
caseVelocity.run("convectionFoam") # Run the case itself

# Eikonal equation
# speed is given by concentrations and yield stresses
# we should also calculate the boundary as the zero countour of the solid field
# can also specify the grid spacing. We will need to interpolate, and carefully include grains
# the grains are specified by a logical mask array, which will need to be all the grain locations
# the tough bit will be carefully transposing this back to the original grid
# I suspect it will turn out to be better to just remove the grain cells directly, rather than snappy
# but this may make it less usable for an arbitrary mesh? I guess how do we do it anyway?
indicator = solidField - 0.5
indicator = np.ma.MaskedArray(indicator, mask)
F = k * sigEqField/((B*sigDetatchAlive + Bdead*sigDetatchDead)/(B + Bdead))
t = skfmm.travel_time(indicator, F, dx=1)

Umag = np.linalg.norm(U, axis=0)
# print(U.shape)
# print(x.shape)

