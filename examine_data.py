import os, argparse

def main(INPUT_DIR, N):
  n = 0
  for entry in os.scandir(INPUT_DIR):
    if N and n >= N:
      break

    if entry.name.endswith('.txt'):
      with open(entry.path, 'r') as f:
        line = f.readline().strip('\n')
        x_min = round(float(line.split('x_min:')[1].split(',')[0]), 2)
        x_max = round(float(line.split('x_max:')[1].split(',')[0]), 2)
        y_min = round(float(line.split('y_min:')[1].split(',')[0]), 2)
        y_max = round(float(line.split('y_max:')[1].split(',')[0]), 2)
        z_min = round(float(line.split('z_min:')[1].split(',')[0]), 2)
        z_max = round(float(line.split('z_max:')[1].split(',')[0]), 2)
        t_min = round(float(line.split('t_min:')[1].split(',')[0]), 2)
        t_max = round(float(line.split('t_max:')[1].split(',')[0]), 2)

      print("x: [{: <6},{: <6}] y: [{: <7},{: <7}] z: [{: <7},{: <7}], t: [{: <6},{: <6}]".format(
        x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max))
      print("delta_x: {: <7} delta_y: {: <7} delta_z: {: <7} delta_t: {: <7}".format(
        round(x_max - x_min, 2), round(y_max - y_min, 2), round(z_max - z_min, 2), round(t_max - t_min, 2)))
      print()
      n += 1
      
        

def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("input_dir")

  parser.add_argument("-n", type=int, default=0)

  args = parser.parse_args()

  return (args.input_dir, args.n)

if __name__ == '__main__':
  arguments = parse_arguments()
  main(*arguments)

