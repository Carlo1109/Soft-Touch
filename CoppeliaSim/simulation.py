from coppelia_V07 import *
from weart import *
import keyboard, os

# Maps closure value, that is between 0 and 1, in the range [min_dist, max_dist]
def mapping_finger(closure):
    l1 = 0.51
    l2 = 0.31
    l3 = 0.23
    offset = 0.396
    max_dist = l1 + l2 + l3 + offset
    distance = max_dist * (1 - closure)
    return distance

def simulation(copp: CoppeliaConnector,weart: WeartConnector):
    print("Starting simulation...")
    copp.start_simulation()
    weart.start_listeners()
    
    # Simulation loop. Press "esc" to stop simulation (Note: On Mac "esc" may not work)
    try:
        while not keyboard.is_pressed("esc"):
            # Computation of distance references
            distance = mapping_finger(weart.get_finger_closure())

            copp.initial_set()
            copp.move_finger(distance)
            
            # Create the graph that tracks tip distance
            copp.sensing()
            
            # Execute simulation step
            copp.step_simulation()

            # Application of contact force
            force = copp.get_contact_force()
            weart.apply_force(force)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping simulation...")
        copp.stop_simulation()

if __name__ == "__main__":
    # Clear previous terminal
    os.system("cls")
    print("Starting script...\n")

    print("Connecting to Coppelia...")
    copp = CoppeliaConnector()
    print("Connected.\n")
    with WeartConnector() as weart:
        print("Connected. Calibrating...")

        weart.calibrate()
        print("Calibrated.\n")

        simulation(copp, weart)
    
