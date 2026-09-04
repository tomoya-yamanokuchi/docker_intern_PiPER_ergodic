import time
from piper_sdk import C_PiperInterface_V2

piper = C_PiperInterface_V2("can0")
piper.ConnectPort()
time.sleep(0.1)

print("Sending emergency stop...")
piper.EmergencyStop(0x01)
print("Emergency stop sent.")
