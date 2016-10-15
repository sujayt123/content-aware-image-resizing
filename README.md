# content-aware-image-resizing
A reasonably fast, memory-efficient implementation of seam carving for content-aware image resizing using NumPy and OpenCV. 
Read more about it [here](https://en.wikipedia.org/wiki/Seam_carving)

### Dependencies:
- Python 2.7
- OpenCV
- NumPy
- MJPEG codec

### Usage:
`python seam_carving.py [input_image] [output_video]`
- input_filepath is the path to the input image
- output_video is the desired path for the output video
<br>
(Note that only ".avi" output is presently supported)

Update 10/15/2016:
Just for kicks, I created a version that distributes the work across multiple processes [on a single computing node], aptly named "seam_carving_distributed.py". It's not particularly fast because 1) the nature of the task - with its continual synchronization, memory sharing and message passing requirements - doesn't lend itself to multiprocessing very well and 2) the much more desirable alternative of multithreading is rendered unfeasible because of Python's GIL. There is some room for improvement in terms of the amount of memory that needs to be shared, the number of messages that need to be sent, etc, but I'm satisfied with this for now. At the very least, my work served as a nice refresher exercise for parallel computing. 

### Examples:

* [Input](http://optipng.sourceforge.net/pngtech/img/lena.png): lena.png <br>
  [Output](https://www.youtube.com/watch?v=lxo-g1fW6Jk): lena_resized.avi
<br>
<br>
* [Input](https://i.ytimg.com/vi/m5sfxYSPUjI/maxresdefault.jpg): yellowstone.png<br>
  [Output](https://www.youtube.com/watch?v=BfhbGSSQ7tk&feature=youtu.be): yellowstone_resized.avi

