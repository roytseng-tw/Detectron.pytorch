import socket
from datetime import datetime

def get_run_name():
  return datetime.now().strftime('%b%d-%H-%M-%S') + '_' + socket.gethostname()