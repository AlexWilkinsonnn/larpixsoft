import math

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

  def segments(self, segment_length):
    segments = []

    k_x = self.x_end - self.x_start
    k_y = self.y_end - self.y_start
    k_z = self.z_end - self.z_start

    line_length = math.sqrt(k_x**2 + k_y**2 + k_z**2)
    N = math.ceil(line_length/segment_length)

    for n in range(N):
      x = k_x * (n/N) + self.x_start
      y = k_y * (n/N) + self.y_start
      z = k_z * (n/N) + self.z_start

      segments.append((x, y, z))

    return segments

  def drift_time_lowerz(self, z):
    return ((z - self.detector.get_zlims()[0])/self.detector.vdrift)*(1/self.detector.time_sampling) 
