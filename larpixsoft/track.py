import math
from scipy import stats

from larpixsoft.detector import Detector

class Track():
  def __init__(self, track, detector : Detector, id=-1):
    self.x_start, self.x_end = track['x_start'], track['x_end']
    self.y_start, self.y_end = track['y_start'], track['y_end'] 
    self.z_start, self.z_end = track['z_start'], track['z_end'] 
    self.t_start, self.t_end = track['t_start'], track['t_end'] 
    self.x, self.y, self.z, self.t = track['x'], track['y'], track['z'], track['t']
    self.electrons = track['n_electrons']
    self.pdg = track['pdgId']
    self.trackid = track['trackID']
    self.dE = track['dE']
    self.eventid = track['eventID']

    self.detector = detector

    self.objectid = id

  def __eq__(self, other):
    if type(other) == type(self):
      if self.objectid != -1:
        return self.objectid == other.objectid
      else:
        return super(Track, self).__eq__(other)

    else:
      return False

  def __hash__(self):
    if self.objectid != -1:
      return hash(self.objectid)
    else:
      return super(Track, self).__hash__()

  def segments(self, segment_length, equal_split=True, fake_fluctuations=0.0,  drift_time='none'):
    if (not equal_split and drift_time != 'none') or (equal_split and fake_fluctuations):
      raise NotImplementedError

    segments = []

    k_x = self.x_end - self.x_start
    k_y = self.y_end - self.y_start
    k_z = self.z_end - self.z_start
    k_t = self.t_start - self.t_end

    line_length = math.sqrt(k_x**2 + k_y**2 + k_z**2)

    # Split track into equal segments of length < segment_length
    if equal_split:
      N = math.ceil(line_length/segment_length)
      for n in range(N):
        segment = {}

        segment['x_start'] = k_x * (n/N) + self.x_start
        segment['y_start'] = k_y * (n/N) + self.y_start
        segment['z_start'] = k_z * (n/N) + self.z_start

        segment['x'] = k_x * ((n + 0.5)/N) + self.x_start
        segment['y'] = k_y * ((n + 0.5)/N) + self.y_start
        segment['z'] = k_z * ((n + 0.5)/N) + self.z_start

        segment['x_end'] = k_x * ((n + 1)/N) + self.x_start
        segment['y_end'] = k_y * ((n + 1)/N) + self.y_start
        segment['z_end'] = k_z * ((n + 1)/N) + self.z_start

        segment['t_start'] = k_t * (n/N) + self.t_start
        segment['t'] = k_t * ((n + 0.5)/N) + self.t_start
        segment['t_end'] = k_t * ((n + 1)/N) + self.t_start

        segment['length'] = math.sqrt((segment['x_end'] - segment['x_start'])**2 + 
          (segment['y_end'] - segment['y_start'])**2 + (segment['z_end'] - segment['z_start'])**2)

        segment['electrons'] = round(self.electrons/N)
        segment['dE'] = self.dE/N

        if drift_time == 'upper':
          segment['drift_time_upperz'] = ((((self.detector.get_zlims()[1] - segment['z']) / 
            self.detector.vdrift)*(1/self.detector.time_sampling) + segment['t']/1000))
        elif drift_time == 'lower':
          segment['drift_time_lowerz'] = ((((segment['z'] - self.detector.get_zlims()[0]) / 
            self.detector.vdrift)*(1/self.detector.time_sampling) + segment['t']/1000))

        segments.append(segment)

    # Split track into segments of length segment_length and a remainder with length < segment_length
    else:
      N = int(line_length // segment_length)
      try:
        step = segment_length / line_length
      except ZeroDivisionError:
        print(N)
        step = 0
      step_small = 1 - (step * N)

      for n in range(N + 1):
        segment = {}

        if n < N:
          start_factor = n * step
          mid_factor = (n + 0.5) * step
          end_factor = (n + 1) * step

          if fake_fluctuations:
            fluctuation_factor = max(0.1, min(stats.norm.rvs(loc=1.0, scale=fake_fluctuations), 1.9))
            segment['electrons'] = round(self.electrons * step * fluctuation_factor)
            segment['dE'] = self.dE * step * fluctuation_factor

          else:
            segment['electrons'] = round(self.electrons * step)
            segment['dE'] = self.dE * step

        elif n == N:
          start_factor = n * step
          mid_factor = (n * step) + (step_small * 0.5)
          end_factor = (n * step) + step_small
  
          if fake_fluctuations:
            fluctuation_factor = max(0.1, min(stats.norm.rvs(loc=1.0, scale=fake_fluctuations), 1.9))
            segment['electrons'] = round(self.electrons * step_small * fluctuation_factor)
            segment['dE'] = self.dE * step_small * fluctuation_factor

          else:
            segment['electrons'] = round(self.electrons * step_small)
            segment['dE'] = self.dE * step_small

        segment['x_start'] = k_x * start_factor + self.x_start
        segment['y_start'] = k_y * start_factor + self.y_start
        segment['z_start'] = k_z * start_factor + self.z_start

        segment['x'] = k_x * mid_factor + self.x_start
        segment['y'] = k_y * mid_factor + self.y_start
        segment['z'] = k_z * mid_factor + self.z_start

        segment['x_end'] = k_x * end_factor + self.x_start
        segment['y_end'] = k_y * end_factor + self.y_start
        segment['z_end'] = k_z * end_factor + self.z_start

        segment['t_start'] = k_t * start_factor + self.t_start
        segment['t'] = k_t * mid_factor + self.t_start
        segment['t_end'] = k_t * end_factor + self.t_start

        segment['length'] = math.sqrt((segment['x_end'] - segment['x_start'])**2 + 
          (segment['y_end'] - segment['y_start'])**2 + (segment['z_end'] - segment['z_start'])**2)

        segments.append(segment)

    return segments

  def drift_time_lowerz(self, z):
    return ((z - self.detector.get_zlims()[0])/self.detector.vdrift)*(1/self.detector.time_sampling) + self.t/1000

  def drift_time_upperz(self, z):
    return ((self.detector.get_zlims()[1] - z)/self.detector.vdrift)*(1/self.detector.time_sampling) + self.t/1000
