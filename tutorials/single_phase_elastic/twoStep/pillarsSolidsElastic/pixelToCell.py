import numpy as np

def pixel_to_cell_id(i, j, Nx, Ny):
    """
    Convert pixel coordinates to OpenFOAM cell ID.
    
    Args:
        i (int): x-coordinate (0 to Nx-1)
        j (int): y-coordinate (0 to Ny-1) 
        Nx (int): number of cells in x-direction
        Ny (int): number of cells in y-direction
        
    Returns:
        int: OpenFOAM cell ID
    """
    if i < 0 or i >= Nx or j < 0 or j >= Ny:
        raise ValueError(f"Pixel coordinates ({i},{j}) out of bounds for {Nx}x{Ny} mesh")
    
    return i + j * Nx

def cell_id_to_pixel(cell_id, Nx, Ny):
    """
    Convert OpenFOAM cell ID to pixel coordinates.
    
    Args:
        cell_id (int): OpenFOAM cell ID
        Nx (int): number of cells in x-direction
        Ny (int): number of cells in y-direction
        
    Returns:
        tuple: (i, j) pixel coordinates
    """
    total_cells = Nx * Ny
    if cell_id < 0 or cell_id >= total_cells:
        raise ValueError(f"Cell ID {cell_id} out of bounds for {Nx}x{Ny} mesh")
    
    i = cell_id % Nx
    j = cell_id // Nx
    return (i, j)

def image_to_openfoam_field(image_array, field_name="image_data"):
    """
    Convert 2D image array to OpenFOAM cell field format.
    
    Args:
        image_array (np.ndarray): 2D image array [Ny, Nx]
        field_name (str): name for the field
        
    Returns:
        np.ndarray: 1D array ordered for OpenFOAM cells
    """
    Ny, Nx = image_array.shape
    
    # Create 1D array for OpenFOAM field
    openfoam_field = np.zeros(Nx * Ny)
    
    # Map image pixels to OpenFOAM cells
    for j in range(Ny):
        for i in range(Nx):
            cell_id = pixel_to_cell_id(i, j, Nx, Ny)
            # Note: image array is [row, col] = [j, i]
            openfoam_field[cell_id] = image_array[j, i]
    
    return openfoam_field

def openfoam_field_to_image(openfoam_field, Nx, Ny):
    """
    Convert OpenFOAM 1D field to 2D image array.
    
    Args:
        openfoam_field (np.ndarray): 1D field from OpenFOAM
        Nx (int): number of cells in x-direction
        Ny (int): number of cells in y-direction
        
    Returns:
        np.ndarray: 2D image array [Ny, Nx]
    """
    if len(openfoam_field) != Nx * Ny:
        raise ValueError(f"Field size {len(openfoam_field)} doesn't match mesh size {Nx*Ny}")
    
    image_array = np.zeros((Ny, Nx))
    
    for cell_id in range(len(openfoam_field)):
        i, j = cell_id_to_pixel(cell_id, Nx, Ny)
        image_array[j, i] = openfoam_field[cell_id]
    
    return image_array

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

# Example usage and verification
if __name__ == "__main__":
    # Example: 4x3 mesh
    Nx, Ny = 4, 3
    
    print("Cell numbering example (4x3 mesh):")
    print("Pixel (i,j) -> Cell ID")
    for j in reversed(range(Ny)):  # Print top to bottom
        row = []
        for i in range(Nx):
            cell_id = pixel_to_cell_id(i, j, Nx, Ny)
            row.append(f"{cell_id:2d}")
        print(f"j={j}: " + " ".join(row))
    
    print("\nVerification:")
    # Test conversion both ways
    for cell_id in range(Nx * Ny):
        i, j = cell_id_to_pixel(cell_id, Nx, Ny)
        back_to_cell = pixel_to_cell_id(i, j, Nx, Ny)
        print(f"Cell {cell_id:2d} -> Pixel ({i},{j}) -> Cell {back_to_cell}")
        assert cell_id == back_to_cell
    
    # Test with example image
    print("\nExample image conversion:")
    test_image = np.arange(Nx * Ny).reshape(Ny, Nx)
    print("Original image:")
    print(test_image)
    
    openfoam_field = image_to_openfoam_field(test_image)
    print(f"\nOpenFOAM field: {openfoam_field}")
    
    reconstructed_image = openfoam_field_to_image(openfoam_field, Nx, Ny)
    print("\nReconstructed image:")
    print(reconstructed_image)
    
    print(f"\nImages match: {np.array_equal(test_image, reconstructed_image)}")