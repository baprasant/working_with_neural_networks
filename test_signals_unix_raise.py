import signal
import os
import time

def send_signal():
    print('Sending signal in current thread')
    os.kill(os.getpid(), signal.SIGUSR1)
    
time.sleep(4)
send_signal()
