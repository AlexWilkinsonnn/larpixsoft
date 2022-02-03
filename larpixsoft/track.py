import math

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
      if self.id != -1:
        return self.objectid == other.objectid
      else:
        return super(Track, self).__eq__(other)

    else:
      return False

  def __hash__(self):
    if self.id != -1:
      return hash(self.objectid)
    else:
      return super(Track, self).__hash__()

  def segments(self, segment_length, drift_time='none'):
    segments = []

    k_x = self.x_end - self.x_start
    k_y = self.y_end - self.y_start
    k_z = self.z_end - self.z_start
    k_t = self.t_start - self.t_end

    line_length = math.sqrt(k_x**2 + k_y**2 + k_z**2)
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

    return segments

  def drift_time_lowerz(self, z):
    return ((z - self.detector.get_zlims()[0])/self.detector.vdrift)*(1/self.detector.time_sampling) + self.t/1000

  def drift_time_upperz(self, z):
    return ((self.detector.get_zlims()[1] - z)/self.detector.vdrift)*(1/self.detector.time_sampling) + self.t/1000
