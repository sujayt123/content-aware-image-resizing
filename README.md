# content-aware-image-resizing
An implementation of seam carving for content-aware image resizing using NumPy and OpenCV.

Input:  An image that you wish to resize
Output: A .avi video illustrating intelligent image resizing as the width of the image is scaled down

Dependencies:
Python 2.7, OpenCV, NumPy, DivX codecs

Usage:
python seam_carving.py [options]

Options:
-h                                       shows this help message                           
<input_filepath> <output_filepath>       input_filepath  is the path (absolute or relative) to the input image
                                         output_filepath is the path (absolute or relative) to the desired output video 
                                         [must end with a .avi extension]

See it in action on Lena.png here:
https://youtu.be/lxo-g1fW6Jk
