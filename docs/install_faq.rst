Common Installation Issues
--------------------------

Mac OS
^^^^^^

Obstacles Encountered During downloading_brains Tutorial (macOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(contact jduva4@jhu.edu or akodba1@jhu.edu for related questions)

1. Issues with using a jupyter notebook
    * `fixes <https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html>`_

2. If using ``virtualenv`` to create the environment rather than ``conda``, make sure that you have Python 3 installed outside of Anaconda (call ``python --version``) because many systems will not. Make sure that ``pip`` references Python 3 (the ``pip --version`` command should show ``3.xx`` in the path), otherwise ``pip`` installs could be updating Python 2 exclusively. 

4. May run into a schema-related error when importing napari in Step 1: “This is specifically a suppressible warning because if you’re using a schema other than the ones in SUPPORTED_VERSIONS, it’s possible that not all functionality will be supported properly. If you don’t want to see these messages, add a warningfilter to your code.” (Source: https://github.com/cwacek/python-jsonschema-objects/issues/184)

5. Not exclusive to macOS but make sure aws .json file has no dollar signs in the strings and is being edited/saved within the terminal using a program like Nano or Vim. Do not use external editors like Sublime.

6.  AWS Credendials Issues
    * See below

7. Section (2) of downloading_brains notebook, Create a Neuroglancer instance and download the volume: make sure variables are correct and functions have correct inputs
    * For Example:
        * Wrong: `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius, radius, radius)`
        * Right:  `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius)`
    
8. Section (4) of downloading_brains notebook, View the volume: the iPyNb kernel may consistently die when running, not allowing napari to be viewed
    * In terminal, type `pip install opencv-contrib-python-headless`
    * Or try including ``%gui qt`` just above the ``import napari`` line. 

9. When installing ``brainlit`` on Mac OS BigSur, make sure you are using ``python==3.9.0`` and not ``python==3.9.1``. This is a known `issue <https://github.com/napari/napari/issues/1393#issuecomment-745615931>`. Please report any other Mac OS BigSur compatibility issues.


Windows
^^^^^^^

Napari Display Problem
~~~~~~~~~~~~~~~~~~~~~~
This document reports an issue that is encountered when running the tutorial ``downloading_brains.ipynb``.

The document includes two sections:
1. a brief description of `Issue#127 <https://github.com/neurodata/brainlit/issues/127>`_
2. a detailed code history

1. Brief Description of Issue#127:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If your drivers/operating system are out of the date:
- Windows 7
- python3.7.9

You may get the following error message:
   
    RuntimeError: Using glBindFramebuffer with no OpenGL context.
In the napari window, the images can be seen loaded but cannot be displayed at the screen as shown in the screenshot below:
`Napari screenshot <https://user-images.githubusercontent.com/66708974/92999637-92c60200-f4f0-11ea-8cad-116a93ae6969.png>`_

2. Detailed Code History
~~~~~~~~~~~~~~~~~~~~~~~~
Input 1:

.. code-block::

    from brainlit.utils.session import NeuroglancerSession
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

Output 1:

.. code-block::

    Downloading: 100%|██████████| 1/1 [00:00<00:00, 13.70it/s]
    Downloading: 46it [00:38,  1.19it/s]

    Downloaded volume is of shape (151, 151, 151), with total intensity 4946609.

Input 2:

.. code-block::

    G_sub = ngl_sess.get_segments(2, bbox)
    paths = graph_to_paths(G_sub)
    print(f"Selected volume contains {G_sub.number_of_nodes()} nodes and {len(paths)} paths")

Output 2:

.. code-block::

    Downloading: 100%|██████████| 1/1 [00:00<00:00,  3.47it/s]
    Selected volume contains 6 nodes and 2 paths

Input 3:

.. code-block::

    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(img)
        viewer.add_shapes(data=paths, shape_type='path', edge_width=0.1, edge_color='blue', opacity=0.1)
        viewer.add_points(vox, size=1, opacity=0.5)

Output 3:

.. code-block::

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

WSL 2
^^^^^

WSL2 Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Windows 10 users that prefer Linux functionality without the speed sacrifice of a Virtual Machine, Brainlit can be installed and run on WSL2.
WSL2 is a fully functional Linux kernel that can run ELF64 binaries on a Windows Host.
- OS Specifications: Version 1903, Build 18362 or higher
- `Installation Instructions <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_
- Any Linux distribution can be installed. Ubuntu16.04.3 was used for this tutorial.

Install python required libraries and build tools. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the below commands to configure the WSL2 environment. See `here <https://stackoverflow.com/questions/8097161/how-would-i-build-python-myself-from-source-code-on-ubuntu/31492697>`_ for more information. 


.. code-block::

    $ sudo apt update && sudo apt install -y build-essential git libexpat1-dev libssl-dev zlib1g-dev
    $ libncurses5-dev libbz2-dev liblzma-dev
    $ libsqlite3-dev libffi-dev tcl-dev linux-headers-generic libgdbm-dev
    $ libreadline-dev tk tk-dev


Install a python version management tool, and create/activate a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Pyenv WSL2 Install <https://gist.github.com/monkut/35c2ef098b871144b49f3f9979032cee>`_ (easiest for WSL2)
- `Anaconda WSL2 Install <https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da>`_

Install brainlit
~~~~~~~~~~~~~~~~

- See `installation section <https://github.com/NeuroDataDesign/brainlit/blob/wsl2-tutorial/README.md#installation>`_ of README.md

Create and save AWS Secrets file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- See AWS Secrets file section below


Configure jupyter notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install jupyter notebook: ``$ python -m pip install jupyter notebook`` and add the following line to your ``~/.bashrc`` script: 


.. code-block::

    export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0 

To launch jupyter notebook, you need to type ``$ jupyter notebook --allow-root``, not just ``$ jupyter notebook``
Then copy and paste one of the URLs outputted into your web browser.  
If your browser is unable to connect, try unblocking the default jupyter port via this command: ``$ sudo ufw allow 8888 ``

Configure X11 Port Forwarding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Install `VcXsrv Windows X Server <https://sourceforge.net/projects/vcxsrv/>`_ on your Windows host machine
- Let VcXsrv through your Public & Private windows firewall. 
(Control Panel -> System and Security -> Windows Defender Firewall -> Allowed Apps -> Change Settings)
- Run XLaunch on your Windows Host Machine with default settings AND select the "Disable Access Control" option
- To confim X11 Port Forwarding is configured, run ``xclock`` on the subsystem.  This should launch on your windows machine. 

Exceptions
~~~~~~~~~~

- The Napari viewer cannot be fully launched (only launches a black screen), because `OpenGL versions>1.5 are not currently supported by WSL2 <https://discourse.ubuntu.com/t/opengl-on-ubuntu-on-wsl-2-timeline/17599>`_.  This should be resolved in upcoming WSL2 updates.



AWS Credentials Issues
^^^^^^^^^^^^^^^^^^^^^^
 
:warning: **SECURITY DISCLAIMER** :warning:

Do **NOT** push any official AWS credentials to any repository. These posts are a good reference to get a sense of what pushing AWS credentials implies:

1. *I Published My AWS Secret Key to GitHub* by Danny Guo `here <https://www.dannyguo.com/blog/i-published-my-aws-secret-key-to-github/>`_
2. *Exposing your AWS access keys on Github can be extremely costly. A personal experience.* by Guru `here <https://medium.com/@nagguru/exposing-your-aws-access-keys-on-github-can-be-extremely-costly-a-personal-experience-960be7aad039>`_
3. *Dev put AWS keys on Github. Then BAD THINGS happened* by Darren Pauli `here <https://www.theregister.com/2015/01/06/dev_blunder_shows_github_crawling_with_keyslurping_bots/>`_


Brainlit can access data volumes stored in `AWS S3 <https://aws.amazon.com/free/storage/s3/?trk=ps_a134p000006BgagAAC&trkCampaign=acq_paid_search_brand&sc_channel=ps&sc_campaign=acquisition_US&sc_publisher=google&sc_category=storage&sc_country=US&sc_geo=NAMER&sc_outcome=acq&sc_detail=aws%20s3&sc_content=S3_e&sc_segment=432339156183&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Storage|Product|US|EN|Text&s_kwcid=AL!4422!3!432339156183!e!!g!!aws%20s3&ef_id=CjwKCAjwkoz7BRBPEiwAeKw3q7yLVNTPLORSa7QUsB5aGT0wAKrnrlnkwNPex8vdqYMVBPqgjlZV2RoCIdgQAvD_BwE:G:s&s_kwcid=AL!4422!3!432339156183!e!!g!!aws%20s3>`_ through the `CloudVolume <https://github.com/seung-lab/cloud-volume>`_ package. As specified in the `docs <https://github.com/seung-lab/cloud-volume#credentials>`_, AWS credentials have to be stored in a file called ``aws-secret.json`` inside the ``~.cloudvolume/secrets/`` folder.

Prerequisites to successfully troubleshoot errors related to AWS credentials:

- The data volume is hosted on S3 (i.e. the link looks like ``s3://your-bucket-name/some-path/some-folder``).
- Familiarity with `IAM Roles <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html>`_ and `how to create them <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html>`_.
- An ``AWS_ACCESS_KEY_ID`` and an ``AWS_SECRET_ACCESS_KEY`` with adequate permissions, provided by an AWS account administrator. Brainlit does not require the IAM user associated with the credentials to have access to the AWS console (i.e. it can be a service account).

Here is a collection of known issues, along with their troubleshoot guide:

Missing ``AWS_ACCESS_KEY_ID``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Error message:

.. code-block::

    python
    ~/opt/miniconda3/envs/brainlit/lib/python3.8/site-packages/cloudvolume/connectionpools.py in _create_connection(self)
        99       return boto3.client(
        100         's3',
    --> 101         aws_access_key_id=self.credentials['AWS_ACCESS_KEY_ID'],
        102         aws_secret_access_key=self.credentials['AWS_SECRET_ACCESS_KEY'],
        103         region_name='us-east-1',

    KeyError: 'AWS_ACCESS_KEY_ID'


This error is thrown when the `credentials` object has an empty ``AWS_ACCESS_KEY_ID` entry. This probably indicates that ``aws-secret.json``  is not stored in the right folder and it cannot be found by CloudVolume. Make sure your credential file is named correctly and stored in ``~.cloudvolume/secrets/``. If you are a Windows user, the output of this Python snippet is the expansion of ``~`` for your system:

.. code-block::

    python
    import os
    HOME = os.path.expanduser('~')
    print(HOME)


example output:

.. code-block::

    bash
    Python 3.8.3 (v3.8.3:6f8c8320e9)
    >>> import os
    >>> HOME = os.path.expanduser('~')
    >>> print(HOME)
    C:\Users\user


Empty ``AKID`` (Access Key ID)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Error message:

.. code-block::

    python
    /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/botocore/client.py in _make_api_call(self, operation_name, api_params)
        654             error_code = parsed_response.get("Error", {}).get("Code")
        655             error_class = self.exceptions.from_code(error_code)
    --> 656             raise error_class(parsed_response, operation_name)
        657         else:
        658             return parsed_response
    ClientError: An error occurred (AuthorizationHeaderMalformed) when calling the GetObject operation: The authorization header is malformed; a non-empty Access Key (AKID) must be provided in the credential.


This error is thrown when your ``aws-secret.json`` file is stored and loaded correctly, and it looks like this:

.. code-block::

    json
    {
    "AWS_ACCESS_KEY_ID": "",
    "AWS_SECRET_ACCESS_KEY": ""
    }


Even though the bucket itself may be public, `boto3 <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`_ requires some non-empty AWS credentials to instantiante the S3 API client.

Access denied
~~~~~~~~~~~~~

.. code-block::

    python
    /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/botocore/client.py in _make_api_call(self, operation_name, api_params)
        654             error_code = parsed_response.get("Error", {}).get("Code")
        655             error_class = self.exceptions.from_code(error_code)
    --> 656             raise error_class(parsed_response, operation_name)
        657         else:
        658             return parsed_response
    ClientError: An error occurred (AccessDenied) when calling the GetObject operation: Access Denied


This error is thrown when:

1. The AWS credentials are stored and loaded correctly but are not allowed to access the data volume. A check with an AWS account administrator is required.

2. There is a typo in your credentials. The content of ``aws-secret.json`` should look like this:


.. code-block::

    json
    {
    "AWS_ACCESS_KEY_ID": "$YOUR_AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "$AWS_SECRET_ACCESS_KEY"
    }


where the ``$`` are placeholder characters and should be replaced along with the rest of the string with the official AWS credentials.