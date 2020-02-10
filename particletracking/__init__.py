import filehandling 
from labvision import video 

class ParticleTracker:
    
    def __init__(self, vidname):
        self.vidname = vidname
        self.filename = filehandling.remove_ext(vidname)
        self.dataname = self.filename + ".hdf5"
        
    def track(self):
        self._get_video_info()
        
    def _get_video_info(self):
        vid = video.ReadVideo(self.vidname)
        self.num_frames = vid.num_frames
        self.fps = vid.fps 
        
        # Get width and height of a processed frame (may be different)
        frame = vid.read_next_frame()
        new_frame, _, _ = self.ip.process(frame)
        