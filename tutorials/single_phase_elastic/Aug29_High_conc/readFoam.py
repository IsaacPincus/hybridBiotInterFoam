#import the class MeshVisu, numpy library and getVolumes function
from fluidfoam import MeshVisu, readmesh, readvector
from fluidfoam.readof import getVolumes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from collections import defaultdict
import re
# path to the simulation to load
path = './pillarsSnappySolids'

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

    
try:
    mesh_data = read_openfoam_polymesh(os.path.join(path,'constant/polyMesh/'))
    
    print(f"Number of cells: {mesh_data['num_cells']}")
    print(f"Number of points: {len(mesh_data['points'])}")
    print(f"Number of faces: {len(mesh_data['faces'])}")
    
    # Print first few cells
    for i, cell in enumerate(mesh_data['cells'][:5]):
        print(f"\nCell {cell['id']}:")
        print(f"  Centroid: ({cell['centroid']['x']:.3f}, "
                f"{cell['centroid']['y']:.3f}, {cell['centroid']['z']:.3f})")
        print(f"  Neighbors: {cell['neighbors']}")
        
except FileNotFoundError as e:
    print(f"Error: Could not find mesh files. {e}")
except Exception as e:
    print(f"Error reading mesh: {e}")

# # Load mesh and create an object called myMesh
# # The box by default is equal to the mesh dimension
# myMesh = MeshVisu( path =  path)

x, y, z = readmesh(path)

timename = '0.0002'
vel = readvector(path, timename, 'U')

U = np.array(vel)

Umag = np.linalg.norm(U, axis=0)
# print(U.shape)
# print(x.shape)

print(np.max(Umag))
print(np.min(Umag))

# fig, ax = plt.subplots(figsize=(8.5, 3), dpi=100)
# plt.rcParams.update({'font.size': 10})
# plt.xlabel('x')
# plt.ylabel('y')

# plt.pcolormesh(x, y, Umag, shading='auto')
# plt.colorbar()
# plt.show()

# Create triangulation
triang = tri.Triangulation(x, y)

# Plot with tricontourf (filled contours)
plt.tricontourf(triang, Umag, levels=20)
plt.colorbar()
# plt.scatter(x, y, c='black', s=20)  # Show data points
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')

# Save instead of show
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # Free up memory



# # these functions transfer a cell ID to a pixel location and vice versa.
# def pixel_to_cell_id(i, j, Nx, Ny):
#     """
#     Convert pixel coordinates to OpenFOAM cell ID.
    
#     Args:
#         i (int): x-coordinate (0 to Nx-1)
#         j (int): y-coordinate (0 to Ny-1) 
#         Nx (int): number of cells in x-direction
#         Ny (int): number of cells in y-direction
        
#     Returns:
#         int: OpenFOAM cell ID
#     """
#     if i < 0 or i >= Nx or j < 0 or j >= Ny:
#         raise ValueError(f"Pixel coordinates ({i},{j}) out of bounds for {Nx}x{Ny} mesh")
    
#     return i + j * Nx

# def cell_id_to_pixel(cell_id, Nx, Ny):
#     """
#     Convert OpenFOAM cell ID to pixel coordinates.
    
#     Args:
#         cell_id (int): OpenFOAM cell ID
#         Nx (int): number of cells in x-direction
#         Ny (int): number of cells in y-direction
        
#     Returns:
#         tuple: (i, j) pixel coordinates
#     """
#     total_cells = Nx * Ny
#     if cell_id < 0 or cell_id >= total_cells:
#         raise ValueError(f"Cell ID {cell_id} out of bounds for {Nx}x{Ny} mesh")
    
#     i = cell_id % Nx
#     j = cell_id // Nx
#     return (i, j)

# def image_to_openfoam_field(image_array, field_name="image_data"):
#     """
#     Convert 2D image array to OpenFOAM cell field format.
    
#     Args:
#         image_array (np.ndarray): 2D image array [Ny, Nx]
#         field_name (str): name for the field
        
#     Returns:
#         np.ndarray: 1D array ordered for OpenFOAM cells
#     """
#     Ny, Nx = image_array.shape
    
#     # Create 1D array for OpenFOAM field
#     openfoam_field = np.zeros(Nx * Ny)
    
#     # Map image pixels to OpenFOAM cells
#     for j in range(Ny):
#         for i in range(Nx):
#             cell_id = pixel_to_cell_id(i, j, Nx, Ny)
#             # Note: image array is [row, col] = [j, i]
#             openfoam_field[cell_id] = image_array[j, i]
    
#     return openfoam_field

# def openfoam_field_to_image(openfoam_field, Nx, Ny):
#     """
#     Convert OpenFOAM 1D field to 2D image array.
    
#     Args:
#         openfoam_field (np.ndarray): 1D field from OpenFOAM
#         Nx (int): number of cells in x-direction
#         Ny (int): number of cells in y-direction
        
#     Returns:
#         np.ndarray: 2D image array [Ny, Nx]
#     """
#     if len(openfoam_field) != Nx * Ny:
#         raise ValueError(f"Field size {len(openfoam_field)} doesn't match mesh size {Nx*Ny}")
    
#     image_array = np.zeros((Ny, Nx))
    
#     for cell_id in range(len(openfoam_field)):
#         i, j = cell_id_to_pixel(cell_id, Nx, Ny)
#         image_array[j, i] = openfoam_field[cell_id]
    
#     return image_array