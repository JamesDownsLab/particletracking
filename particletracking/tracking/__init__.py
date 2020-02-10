import filehandling
from labvision import video, images
from particletracking import preprocessing, configurations, dataframes
import multiprocessing as mp
from tqdm import tqdm
import os


class ParticleTracker:

    def __init__(self, vidname, manager, multiprocess=False, link=False):
        self.manager = manager
        self.vidname = vidname
        self.filename = filehandling.remove_ext(vidname)
        self.dataname = self.filename + ".hdf5"
        self.multiprocess = multiprocess
        cpus = mp.cpu_count()
        self.num_processes = cpus // 2 if self.multiprocess else 1
        self.link = link

    def track(self):
        self._get_video_info()
        if self.multiprocess:
            self._track_multiprocess()
        else:
            self._track_process(0)
        self.manager.extra_steps(self.filename)

    def _get_video_info(self):
        vid = video.ReadVideo(self.vidname)
        self.num_frames = vid.num_frames
        self.fps = vid.fps
        self.frame_div = self.num_frames // self.num_processes

        # Get width and height of a processed frame (may be different)
        frame = vid.read_next_frame()
        new_frame, _, _ = self.manager.process(frame)

    def _track_process(self, group_number):
        data_name = (str(group_number) + '.hdf5'
                     if self.multiprocess else self.dataname)
        with dataframes.DataStore(data_name, load=False) as data:
            data.add_metadata('number of frames', self.num_frames)
            data.add_metadata('video_filename', self.vidname)
            data.add_metadata('crop', self.manager.preprocessor.crop)
            start = self.frame_div * group_number
            vid = video.ReadVideo(self.vidname)
            vid.set_frame(start)
            if group_number == 3:
                missing = self.num_frames - self.num_processes * (
                    self.frame_div)
                frame_div = self.frame_div + missing
            else:
                frame_div = self.frame_div

            # Iterate over frames
            for f in tqdm(range(frame_div), 'Tracking'):
                frame = vid.read_next_frame()
                info, boundary, info_headings = self.manager.analyse_frame(
                    frame)
                data.add_tracking_data(start + f, info,
                                       col_names=info_headings)
                if f == 0:
                    data.add_metadata('boundary', boundary)

    def _track_multiprocess(self):
        """Splits processing into chunks"""
        p = mp.Pool(self.num_processes)
        p.map(self._track_process, range(self.num_processes))
        p.close()
        p.join()
        self._cleanup_intermediate_dataframes()

    def _cleanup_intermediate_dataframes(self):
        """Concatenates and removes intermediate dataframes"""
        dataframe_list = ["{}.hdf5".format(i) for i in
                          range(self.num_processes)]
        dataframes.concatenate_datastore(dataframe_list,
                                         self.dataname)
        for file in dataframe_list:
            os.remove(file)


class ExampleManager:

    def __init__(self):
        self.parameters = configurations.EXAMPLE_CHILD_PARAMETERS
        self.preprocessor = preprocessing.PreProcessor(self.parameters)

    def process(self, frame):
        frame, boundary, cropped_frame = self.preprocessor.process(frame)
        return frame, boundary, cropped_frame

    def analyse_frame(self, frame):
        new_frame, boundary, cropped_frame = self.preprocessor.process(frame)
        info = images.find_circles(
            new_frame,
            self.parameters['min_dist'][0],
            self.parameters['p_1'][0],
            self.parameters['p_2'][0],
            self.parameters['min_rad'][0],
            self.parameters['max_rad'][0]
        )
        info_headings = ['x', 'y', 'r']
        return info, boundary, info_headings

    def extra_steps(self, name):
        pass


if __name__ == "__main__":
    file = "/home/ppxjd3/Videos/short.MP4"
    tracker = ParticleTracker(file, ExampleManager())
    tracker.track()
