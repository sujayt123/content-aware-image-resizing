# content-aware-image-resizing
An implementation of seam carving for content-aware image resizing using NumPy and OpenCV. 
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
Note that only an "avi" extension is presently supported for the output.

### Examples:

* [Input](http://optipng.sourceforge.net/pngtech/img/lena.png): lena.png <br>
  [Output](https://www.youtube.com/watch?v=lxo-g1fW6Jk): lena_resized.avi
<br>
<br>
* [Input](https://i.ytimg.com/vi/m5sfxYSPUjI/maxresdefault.jpg): yellowstone.png<br>
  [Output](https://www.youtube.com/watch?v=BfhbGSSQ7tk&feature=youtu.be): yellowstone_resized.avi

