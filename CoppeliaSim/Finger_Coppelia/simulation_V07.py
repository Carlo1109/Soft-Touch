from coppelia_V07 import *
from weart import *
import keyboard, os
import time
from filterpy.kalman import KalmanFilter

def mapping_index(closure):
    l1 = 0.51
    l2 = 0.31
    l3 = 0.23
    offset = 0.396
    max_dist = l1 + l2 + l3 + offset
    distance = max_dist * (1 - closure)
    return distance



def simulation(copp: CoppeliaConnector,weart: WeartConnector, openxr):
    print("Starting simulation...")
    copp.start_simulation()
    weart.start_listeners()
    
    
    dis = mapping_index(weart.get_index_closure())
    
    d_back = []
    
    try:
        while not keyboard.is_pressed("ì"):
            distance = mapping_index(weart.get_index_closure())
            """
            for i in range(N):
                distance = mapping_index(weart.get_index_closure())
                d += 1/N *(distance)
            """
            copp.ini()
            copp.move_index(distance, d_back)
            copp.sensing()
            copp.step_simulation()
            d_back.append(distance)
            if(len(d_back) > 6):
                d_back.pop(0)
            force = copp.get_contact_force()
            weart.apply_force(force)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping simulation...")
        copp.stop_simulation()

if __name__ == "__main__":
    os.system("cls")
    print("Starting script...\n")

    print("Connecting to Coppelia...")
    copp = CoppeliaConnector()
    print("Connected.\n")
    with WeartConnector() as weart:
        print("Connected. Calibrating...")

        weart.calibrate()
        print("Calibrated.\n")

        # openxr code
        simulation(copp, weart, None)
    
