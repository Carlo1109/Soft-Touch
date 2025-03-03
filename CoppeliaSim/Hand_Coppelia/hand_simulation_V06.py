from hand_coppelia_V06 import *
import keyboard, os
from threading import Thread

def simulation(copp: CoppeliaConnector, weart: WeartConnector, openxr):
    print("Starting simulation.")
    copp.start_simulation()
    weart.start_listeners()
    
    # Simulation loop. Press "esc" to stop simulation (Note: On Mac "esc" may not work)
    try:
        while not keyboard.is_pressed("esc"):
            # Computation of distance references
            distance_thumb = copp.mapping(weart.get_closure("thumb"))
            distance_index = copp.mapping(weart.get_closure("index"))
            distance_middle = copp.mapping(weart.get_closure("middle"))
            
            # Thread creation and starting for simultaneous movements
            thread_thumb = Thread(target=copp.actuation(distance_thumb, "thumb", weart))
            thread_index = Thread(target=copp.actuation(distance_index, "index", weart))
            thread_middle = Thread(target=copp.actuation(distance_middle, "middle", weart))

            thread_thumb.start()
            thread_index.start()
            thread_middle.start()

            thread_thumb.join()     
            thread_index.join()
            thread_middle.join()   

            # Execute simulation step
            copp.step_simulation()
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping simulation...")
        copp.stop_simulation()


if __name__ == "__main__":
    # Clear previous terminal
    os.system("cls")

    print("Connecting to Coppelia...")
    copp = CoppeliaConnector()
    with WeartConnector() as weart:
        print("Connected. Calibrating...")

        weart.calibrate()
        print("Calibrated.\n")

        simulation(copp, weart, None)