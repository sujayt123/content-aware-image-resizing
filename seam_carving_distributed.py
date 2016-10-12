from concurrent.futures import ProcessPoolExecutor, wait
import cv2
import math
import numpy as np
import sys

def compute_scoring_matrix(color_image):  
    """
    Computes the energy matrix for the input image.
    param:  color_image     a matrix of RGB values corresponding to the current image
    return: gradient        a matrix of values indicating the "value", or "information", of a pixel in the image
    """
    grayscale = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # Filter high frequency noise out of calculations of the energy matrix. 
    blur = cv2.blur(grayscale,(5,5))
    # Convolve image with Sobel_x kernel.
    sobelx64f = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    abs_sobelx64f = np.absolute(sobelx64f)
    sobelx_8u = np.uint8(abs_sobelx64f)
    # Convolve image with Sobel_y kernel.
    sobely64f = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)
    abs_sobely64f = np.absolute(sobely64f)
    sobely_8u = np.uint8(abs_sobely64f)  
    # Calculate the gradient magnitude as a function of the output of the Sobel filter.  
    gradient = cv2.addWeighted(sobelx_8u, 0.5, sobely_8u, 0.5, 0)
    # Preserve each row, but throw out any of the padded columns from consideration.
    return gradient.astype(int).tolist()

def update(parent, curr, prev, top_left, top, top_right):
    if top_left <=  top and top_left <= top_right:
        curr[j] += top_left
    elif top <=  top_left and top <= top_right:
        parent[iteration][j] = 1
        curr[j] += top
    else:
        parent[iteration][j] = 2
        curr[j] += top_right

def process_inf(num_rows, end_edge_conn):
    for i in range(num_rows):
        end_edge_conn.send(float("inf"))    
        end_edge_conn.recv()
    return

def process_main(num_rows, left_edge_conn, right_edge_conn, energy_matrix, start_idx, end_idx):    
    prev, curr, parent = np.zeros(end_idx - start_idx), np.zeros(end_idx - start_idx), np.zeros((rows, end_idx - start_idx), dtype=uint8)
    for j in range(end_idx - start_idx):
        prev[j] = energy_matrix[0][start_idx + j]
    left_edge_conn.send(prev[0])
    right_edge_conn.send(prev[-1])
    for i in range(1, num_rows):
        # Receive right-edge value from left process and compute first element accordingly.
        left_recv_val = left_edge_conn.recv()
        update(parent, curr, prev, left_recv_val, prev[j], prev[j + 1])
        # Notify the left peer of the left-edge value.
        left_recv_val.send(curr[0])
        # Receive left-edge value from right process and compute last element accordingly.
        right_recv_val  right_edge_conn.recv()
        update(parent, curr, prev, prev[j - 1], prev[j], right_recv_val)        
        # Notify the right peer of the right-edge value.
        right_recv_val.send(curr[-1])
        # Compute internal values.
        for j in range(1, len(prev) - 1):
            curr[j] = energy_matrix[i][start_idx + j]
            update(parent, curr, prev, prev[j - 1], prev[j], prev[j + 1])
        # Swap prev and curr.
        temp = prev
        prev = curr
        curr = temp
    return prev, parent

def detect_seam(energy_matrix):
    """
    Uses dynamic programming to detect a minimum cost contiguous seam from top to bottom of the input image.

    param:  energy_matrix    the cost matrix 
    return: path             a list of the coordinates of the pixels in the minimum cost seam
    """        
    left_to_right, right_to_left = 
    



    # .....





    # Find where the minimum cost path ends in the bottom row of the matrix.
    min_end_idx = np.argmin(prev)
    curr_coord = [rows - 1, min_end_idx]
    path = [(rows - 1, min_end_idx - 1)]
    # Trace back using the parent matrix to determine the actual seam.
    while curr_coord[0] > 0:
        if parent[curr_coord[0]][curr_coord[1]] == 0:
            curr_coord[1] -= 1
        elif parent[curr_coord[0]][curr_coord[1]] == 2:
            curr_coord[1] += 1          
        curr_coord[0] -= 1                          
        # Note that in our output, we must shift all our column values by 1. 
        # This is because of the sentinels in the DP matrix.
        path.append((curr_coord[0], curr_coord[1] - 1))
    return path

def carve_seam(seam, color_image, energy_matrix):
    """
    Removes pixels at coordinates specified in seam array from color_image.

    param:  seam             a list of the coordinates of the pixels in the minimum cost seam
    param:  color_image      the color representation of the image   
    param:  energy_matrix    the cost matrix      
    return: color_image                   
    """     
    for pixel in seam:    
        color_image[pixel[0]].pop(pixel[1])        
    return color_image

def main(path_to_image, path_to_output):    
    """
    Executes the program.

    param:  path_to_image   the path specifying the location of the image on disk
    param:  path_to_output  the path to the desired output video
    """   
    color_image = cv2.imread(path_to_image)         
    video = cv2.VideoWriter(path_to_output, cv2.VideoWriter_fourcc(*'MJPG'), 15, (len(color_image[0]), len(color_image)))   
    # We can use a with statement to ensure threads are cleaned up promptly
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(len(color_image[0]) - 1):
            energy_matrix = compute_scoring_matrix(color_image)        
            seam = detect_seam(energy_matrix, executor)            
            color_image = np.asarray(carve_seam(seam, color_image.tolist(), energy_matrix), dtype=np.uint8)
            video.write(cv2.copyMakeBorder(color_image, 0, 0, 0, i + 1, cv2.BORDER_CONSTANT, value=[255,255,255]))

if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        print("Please specify an appropriate set of command-line arguments. Run with -h flag for more details.", file=sys.stderr)
        sys.exit(1)
    if args[1] == "-h":
        print("seam_carving.py")
        print("Author: Sujay Tadwalkar")
        print()
        print("Command Syntax:")
        print("python seam_carving.py [options]")
        print()
        print("Options:")
        print("-h".ljust(40), "shows this help message".ljust(50))
        print("<input_filepath> <output_filepath>".ljust(40), "input_filepath  is the path (absolute or relative) to the input image".ljust(50))
        print(" " * 41 + "output_filepath is the path (absolute or relative) to the desired output video [must end with a .avi extension]".ljust(50))
        print() 
        sys.exit(0)
    if len(args) <= 2 or args[2].endswith(".avi") is False:    
        print("Please specify an appropriate set of command-line arguments. Run with -h flag for more details.", file=sys.stderr)        
        sys.exit(1)
        
    main(args[1], args[2])
