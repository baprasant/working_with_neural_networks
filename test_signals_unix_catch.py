import signal
import os

# Our signal handler
def signal_handler(signum, frame):
    print("Signal Number:", signum, " Frame: ", frame)

# Register our signal handler with SIGUSR1
signal.signal(signal.SIGUSR1, signal_handler)

signal.pause()
