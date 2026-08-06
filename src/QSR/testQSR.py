from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace

qsrlib = QSRlib()

# clean data, make sure human and object available
# remove frame drops or ignore when object is out of frame
# if bounding box under threshold or mask under threshold remove object
# make world trace, see how dict needs to be restructured for it
# call qsr, decide what calculus qtc may not be appropriate for 3d inference