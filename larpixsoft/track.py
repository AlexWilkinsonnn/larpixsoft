from larpixsoft.detector import Detector

class Track():
  def __init__(self, track, detector : Detector):
    self.x_start, self.x_end = track['x_start'], track['x_end']
    self.y_start, self.y_end = track['y_start'], track['y_end'] 
    self.z_start, self.z_end = track['z_start'], track['z_end'] 
    self.t_start, self.t_end = track['t_start'], track['t_end'] 
    self.x, self.y, self.z, self.t = track['x'], track['y'], track['z'], track['t']
    self.electrons = track['n_electrons']

    self.detector = detector

  def drift_time_lowerz(self):
    start_t = ((self.z_start - self.detector.get_zlims()[0])/self.detector.vdrift)*(1/self.detector.time_sampling) 
    mid_t = ((self.z - self.detector.get_zlims()[0])/self.detector.vdrift)*(1/self.detector.time_sampling) 
    end_t = ((self.z_end - self.detector.get_zlims()[0])/self.detector.vdrift)*(1/self.detector.time_sampling) 

    return start_t, mid_t, end_t
