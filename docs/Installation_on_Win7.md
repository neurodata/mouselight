# Napari Display Problem
This document reports an issue that is encountered when running the tutorial ```downloading_brains.ipynb```.

The document includes two sections:
1. a brief description of [Issue#127](https://github.com/neurodata/brainlit/issues/127)
2. a detailed code history

### <strong>1. Brief Description of Issue#127:</strong>
If your drivers/operating system are out of the date:
- Windows 7
- python3.7.9

,you may get the following error message:
   
    RuntimeError: Using glBindFramebuffer with no OpenGL context.
In the napari window, the images can be seen loaded but cannot be displayed at the screen as shown in the screenshot below:
![Napari screenshot](https://user-images.githubusercontent.com/66708974/92999637-92c60200-f4f0-11ea-8cad-116a93ae6969.png)

### <strong>2. Detailed Code History</strong>
Input 1:
```from brainlit.utils.session import NeuroglancerSession
from brainlit.utils.swc import graph_to_paths
import napari

dir = "s3://mouse-light-viz/precomputed_volumes/brain1"
dir_segments = "s3://mouse-light-viz/precomputed_volumes/brain1_segments"
mip = 0
v_id = 0
radius = 75

# get image and center point
ngl_sess = NeuroglancerSession(mip = mip, url = dir, url_segments=dir_segments)
img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius)
print(f"\n\nDownloaded volume is of shape {img.shape}, with total intensity {sum(sum(sum(img)))}.")
```
Output 1:
```
Downloading: 100%|██████████| 1/1 [00:00<00:00, 13.70it/s]
Downloading: 46it [00:38,  1.19it/s]

Downloaded volume is of shape (151, 151, 151), with total intensity 4946609.
```
Input 2:
```
G_sub = ngl_sess.get_segments(2, bbox)
paths = graph_to_paths(G_sub)
print(f"Selected volume contains {G_sub.number_of_nodes()} nodes and {len(paths)} paths")
```
Output 2:
```
Downloading: 100%|██████████| 1/1 [00:00<00:00,  3.47it/s]
Selected volume contains 6 nodes and 2 paths
```
Input 3:
```
with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img)
    viewer.add_shapes(data=paths, shape_type='path', edge_width=0.1, edge_color='blue', opacity=0.1)
    viewer.add_points(vox, size=1, opacity=0.5)
```
Output 3:
```
ERROR:root:Unhandled exception:
Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\app\backends\_qt.py", line 825, in paintGL
    self._vispy_canvas.events.draw(region=None)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 455, in __call__
    self._invoke_callback(cb, event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 475, in _invoke_callback
    self, cb_event=(cb, event))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 471, in _invoke_callback
    cb(event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\context.py", line 175, in flush_commands
    self.shared.parser.parse([('CURRENT', 0, fbo)])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 819, in parse
    self._parse(command)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 743, in _parse
    self._gl_initialize()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 851, in _gl_initialize
    if this_version < '2.1':
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\distutils\version.py", line 52, in __lt__
    c = self._cmp(other)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\distutils\version.py", line 335, in _cmp
    if self.version == other.version:
AttributeError: 'LooseVersion' object has no attribute 'version'

ERROR:root:Unhandled exception:
Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 53, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native
AttributeError: 'function' object has no attribute '_native'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 72, in _get_gl_func
    func = getattr(_lib, name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 377, in __getattr__
    func = self.__getitem__(name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 382, in __getitem__
    func = self._FuncPtr((name_or_ordinal, self))
AttributeError: function 'glBindFramebuffer' not found

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\app\backends\_qt.py", line 825, in paintGL
    self._vispy_canvas.events.draw(region=None)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 455, in __call__
    self._invoke_callback(cb, event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 475, in _invoke_callback
    self, cb_event=(cb, event))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 471, in _invoke_callback
    cb(event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\context.py", line 175, in flush_commands
    self.shared.parser.parse([('CURRENT', 0, fbo)])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 819, in parse
    self._parse(command)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 745, in _parse
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, args[0])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 55, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native = _get_gl_func("glBindFramebuffer", None, (ctypes.c_uint, ctypes.c_uint,))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 87, in _get_gl_func
    raise RuntimeError('Using %s with no OpenGL context.' % name)
RuntimeError: Using glBindFramebuffer with no OpenGL context.

WARNING: Error drawing visual <Volume at 0x21be1648>
WARNING:vispy:Error drawing visual <Volume at 0x21be1648>
ERROR:root:Unhandled exception:
Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 53, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native
AttributeError: 'function' object has no attribute '_native'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 72, in _get_gl_func
    func = getattr(_lib, name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 377, in __getattr__
    func = self.__getitem__(name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 382, in __getitem__
    func = self._FuncPtr((name_or_ordinal, self))
AttributeError: function 'glBindFramebuffer' not found

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\app\backends\_qt.py", line 825, in paintGL
    self._vispy_canvas.events.draw(region=None)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 455, in __call__
    self._invoke_callback(cb, event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 475, in _invoke_callback
    self, cb_event=(cb, event))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 471, in _invoke_callback
    cb(event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 217, in on_draw
    self._draw_scene()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 266, in _draw_scene
    self.draw_visual(self.scene)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 304, in draw_visual
    node.draw()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\visuals.py", line 99, in draw
    self._visual_superclass.draw(self)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\visual.py", line 443, in draw
    self._vshare.index_buffer)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\shaders\program.py", line 101, in draw
    Program.draw(self, *args, **kwargs)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\program.py", line 533, in draw
    canvas.context.flush_commands()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\context.py", line 175, in flush_commands
    self.shared.parser.parse([('CURRENT', 0, fbo)])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 819, in parse
    self._parse(command)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 745, in _parse
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, args[0])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 55, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native = _get_gl_func("glBindFramebuffer", None, (ctypes.c_uint, ctypes.c_uint,))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 87, in _get_gl_func
    raise RuntimeError('Using %s with no OpenGL context.' % name)
RuntimeError: Using glBindFramebuffer with no OpenGL context.

WARNING: Error drawing visual <Volume at 0x21be1648>
WARNING:vispy:Error drawing visual <Volume at 0x21be1648>
ERROR:root:Unhandled exception:
Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 53, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native
AttributeError: 'function' object has no attribute '_native'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 72, in _get_gl_func
    func = getattr(_lib, name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 377, in __getattr__
    func = self.__getitem__(name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 382, in __getitem__
    func = self._FuncPtr((name_or_ordinal, self))
AttributeError: function 'glBindFramebuffer' not found

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\app\backends\_qt.py", line 825, in paintGL
    self._vispy_canvas.events.draw(region=None)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 455, in __call__
    self._invoke_callback(cb, event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 475, in _invoke_callback
    self, cb_event=(cb, event))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 471, in _invoke_callback
    cb(event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 217, in on_draw
    self._draw_scene()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 266, in _draw_scene
    self.draw_visual(self.scene)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 304, in draw_visual
    node.draw()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\visuals.py", line 99, in draw
    self._visual_superclass.draw(self)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\visual.py", line 443, in draw
    self._vshare.index_buffer)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\shaders\program.py", line 101, in draw
    Program.draw(self, *args, **kwargs)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\program.py", line 533, in draw
    canvas.context.flush_commands()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\context.py", line 175, in flush_commands
    self.shared.parser.parse([('CURRENT', 0, fbo)])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 819, in parse
    self._parse(command)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 745, in _parse
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, args[0])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 55, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native = _get_gl_func("glBindFramebuffer", None, (ctypes.c_uint, ctypes.c_uint,))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 87, in _get_gl_func
    raise RuntimeError('Using %s with no OpenGL context.' % name)
RuntimeError: Using glBindFramebuffer with no OpenGL context.

WARNING: Error drawing visual <Volume at 0x21be1648>
WARNING:vispy:Error drawing visual <Volume at 0x21be1648>
ERROR:root:Unhandled exception:
Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 53, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native
AttributeError: 'function' object has no attribute '_native'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 72, in _get_gl_func
    func = getattr(_lib, name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 377, in __getattr__
    func = self.__getitem__(name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 382, in __getitem__
    func = self._FuncPtr((name_or_ordinal, self))
AttributeError: function 'glBindFramebuffer' not found

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\app\backends\_qt.py", line 825, in paintGL
    self._vispy_canvas.events.draw(region=None)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 455, in __call__
    self._invoke_callback(cb, event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 475, in _invoke_callback
    self, cb_event=(cb, event))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 471, in _invoke_callback
    cb(event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 217, in on_draw
    self._draw_scene()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 266, in _draw_scene
    self.draw_visual(self.scene)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 304, in draw_visual
    node.draw()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\visuals.py", line 99, in draw
    self._visual_superclass.draw(self)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\visual.py", line 443, in draw
    self._vshare.index_buffer)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\shaders\program.py", line 101, in draw
    Program.draw(self, *args, **kwargs)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\program.py", line 533, in draw
    canvas.context.flush_commands()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\context.py", line 175, in flush_commands
    self.shared.parser.parse([('CURRENT', 0, fbo)])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 819, in parse
    self._parse(command)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 745, in _parse
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, args[0])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 55, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native = _get_gl_func("glBindFramebuffer", None, (ctypes.c_uint, ctypes.c_uint,))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 87, in _get_gl_func
    raise RuntimeError('Using %s with no OpenGL context.' % name)
RuntimeError: Using glBindFramebuffer with no OpenGL context.

WARNING: Error drawing visual <Volume at 0x21be1648>
WARNING:vispy:Error drawing visual <Volume at 0x21be1648>
ERROR:root:Unhandled exception:
Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 53, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native
AttributeError: 'function' object has no attribute '_native'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 72, in _get_gl_func
    func = getattr(_lib, name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 377, in __getattr__
    func = self.__getitem__(name)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\ctypes\__init__.py", line 382, in __getitem__
    func = self._FuncPtr((name_or_ordinal, self))
AttributeError: function 'glBindFramebuffer' not found

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\app\backends\_qt.py", line 825, in paintGL
    self._vispy_canvas.events.draw(region=None)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 455, in __call__
    self._invoke_callback(cb, event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 475, in _invoke_callback
    self, cb_event=(cb, event))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\util\event.py", line 471, in _invoke_callback
    cb(event)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 217, in on_draw
    self._draw_scene()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 266, in _draw_scene
    self.draw_visual(self.scene)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\canvas.py", line 304, in draw_visual
    node.draw()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\scene\visuals.py", line 99, in draw
    self._visual_superclass.draw(self)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\visual.py", line 443, in draw
    self._vshare.index_buffer)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\visuals\shaders\program.py", line 101, in draw
    Program.draw(self, *args, **kwargs)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\program.py", line 533, in draw
    canvas.context.flush_commands()
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\context.py", line 175, in flush_commands
    self.shared.parser.parse([('CURRENT', 0, fbo)])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 819, in parse
    self._parse(command)
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\glir.py", line 745, in _parse
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, args[0])
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\_gl2.py", line 55, in glBindFramebuffer
    nativefunc = glBindFramebuffer._native = _get_gl_func("glBindFramebuffer", None, (ctypes.c_uint, ctypes.c_uint,))
  File "C:\ProgramData\Miniconda3\envs\brainlit\lib\site-packages\vispy\gloo\gl\gl2.py", line 87, in _get_gl_func
    raise RuntimeError('Using %s with no OpenGL context.' % name)
RuntimeError: Using glBindFramebuffer with no OpenGL context.
```