# local code
import time
from interfaces import *
from mujoco_connector import MujocoConnector, MujocoSimpleVisualizer
from mujoco_xr import MujocoXRVisualizer
from weart import WeartConnector, HAPTIC_FINGERS
from guis import TUI
from benchmarking import Benchmarker, Plotter
from hand import Hand
from math import cos

# libraries
from contextlib import nullcontext
from threading import Thread
from colorama import just_fix_windows_console, Fore, Style

just_fix_windows_console()

def simulation(engine: Engine,
                weart: WeartConnector | None,
                visualizer: Visualizer,
                hand_provider: HandPoseProvider | None,
                gui: GUI,
                hands: tuple[Hand, Hand]):
    print("Starting simulation.")
    engine.start_simulation()

    # If you want to use benchmarking decomment the relative commands in this file
        # perf_bench = Benchmarker(title = "Performance profiler")
        # frame_bench = Benchmarker(title = "Performance profiler")
        # force_plot = Plotter(title = "Applied force graph")

    def loop():
        time.sleep(5)
        print("Starting visualization...")
        visualizer.start_visualization()
        gui.start_gui(engine, visualizer)
        # If VR is being used, creation of benchmarking
            # if isinstance(visualizer, MujocoXRVisualizer):
            #     visualizer.add_perf_counters(perf_bench, frame_bench)

        print(Style.BRIGHT + Fore.GREEN, f"Done!{Style.NORMAL} Everything is up and running.{Fore.RESET}\n")
        try:
            engine.init_task()
            
            while not visualizer.should_exit() and not gui.should_exit():
                    # frame_bench.new_iteration()
                frame_continue, frame_duration = visualizer.wait_frame()
                    # frame_bench.mark("Wait frame")
                    # frame_bench.end_iteration()
                if visualizer.should_exit():
                    break
                if not frame_continue:
                    continue
                    # perf_bench.new_iteration()
                    # force_plot.new_iteration()

                if hand_provider is not None:
                    # Hand movement tracking in VR environment, for each hand if tracking is True 
                    for hand in filter(lambda h: h.tracking, hands):
                        hand_pose = hand_provider.get_hand_pose(hand.id)
                        if hand_pose is not None:
                            engine.move_hand(hand.id, *hand_pose)

                if weart is not None or True:
                    # Kinematic control of each finger, if haptics is True
                    for hand in filter(lambda h: h.haptics, hands):
                        for finger in [ "thumb", "index", "middle", "annular", "pinky"]: #"thumb",
                            # haptic_finger is used to define the closure, finger to define the movement
                            haptic_finger = finger
                            abduction = -1
                            if finger in ["annular", "pinky"]:
                                haptic_finger = "middle"
                            # if finger == "thumb":
                            #     abduction = weart.get_thumb_abduction(hand.id, haptic_finger)
                            closure = weart.get_finger_closure(hand.id, haptic_finger)
                            
                            engine.move_finger(hand.id, finger, closure, abduction)

                engine.step_simulation(frame_duration)
                    # perf_bench.mark("Step simulation")

                visualizer.render_frame()
                    # perf_bench.mark("Render")

                for hand in filter(lambda h: h.haptics, hands):
                    # Application of haptic feedback
                    for finger in HAPTIC_FINGERS:
                        force, temperature, texture = engine.get_contact_force(hand.id, finger)
                            # perf_bench.mark("Contact force")
                            # force_plot.plot(force, f"{finger} hand {hand}")

                        if weart is not None:
                            weart.apply_force(hand.id, finger, force, temperature, texture)
                            # perf_bench.mark("Apply force to finger")

            # force_plot.end_iteration()
            # perf_bench.end_iteration()

            # d, t = frame_bench.get_data(from_date=datetime.datetime.now() - datetime.timedelta(seconds=1))
            # print(len(d["Wait frame"]))

        except KeyboardInterrupt:
            pass # To exit gracefully. Even though we swallow the error, we still exit the loop.
        finally:
                # force_plot.stop()
                # perf_bench.stop()
                # perf_bench.export_csv("benchmark.csv", include_time=True)

            print("\nStopping visualization...")
            gui.stop_gui()
            visualizer.stop_visualization()

            print("Stopping simulation...")
            engine.stop_simulation()

            print("\nCiao!\n")

    threaded = False
    if threaded:
        t = Thread(target = loop)
        t.start()
        # we must run the loop in another thread because the graph can only be visualized in the main thread...
            # perf_bench.graph_viz(max_points=1000, use_time=True)
            # force_plot.graph_viz(max_points=10000, y_axis="Force")
        t.join()
    else:
        loop()

if __name__ == "__main__":
    # CHANGEABLE PARAMETERS

    used_engine = "mujoco"
    used_viz = "openxr"
    use_weart = True
    used_gui = "tui"
    
    # If you want to upload another scene, it needs absolute path
    scene_path = "C:/Users/alien/Documents/Medical-Robotics-Project/MujocoVR_V12/assets/hand.xml"

    # In order to not visualize a hand, set tracking and haptics to false 
    hands = (
        Hand(id = 0, side = "left", tracking = False, haptics = False, controller_rotation = 0),
        Hand(id = 1, side = "right", tracking = True, haptics = True, controller_rotation = 0)
    )

    # SCRIPT
    print("Starting script...\n")

    engine = mujoco = visualizer = weart = hand = gui = None

    match used_engine:
        case "mujoco":
            # Creation of MuJoCo connector
            print("Loading MuJoCo...")
            engine = mujoco = MujocoConnector(scene_path, hands)
            print("Loaded.\n")
        case "coppelia":
            print("Connecting to CoppeliaSim...")
            engine = CoppeliaConnector()
            print("Connected.\n")
            used_viz = None
        case _:
            raise RuntimeError("Invalid engine name")

    match used_viz:
        case None:
            visualizer_ctx = nullcontext(Visualizer())
        case "simple":
            assert mujoco is not None
            # If engine MuJoCo, open MuJoCo window
            visualizer_ctx = nullcontext(MujocoSimpleVisualizer(mujoco))
        case "openxr":
            assert mujoco is not None
            print("Loading Virtual Reality...")
            # If engine MuJoCo, use VR
            visualizer_ctx = hand = MujocoXRVisualizer(mujoco, mirror_window = True, fullscreen = True, samples = 8, fps_counter = False)
        case _:
            raise RuntimeError("Invalid visualizer name")
        
    if use_weart:
            print("Connecting to WEART...")
            # Creation of WeArt connector
            enabled_hands_haptic = [hand.id for hand in hands if hand.haptics]
            weart_ctx = WeartConnector(enabled_hands_haptic)
    else:
        weart_ctx = nullcontext()
        print(Fore.RED + Style.BRIGHT, "WARNING:", Style.NORMAL + "You have not enabled WEART.\n", Style.RESET_ALL)

    match used_gui:
        case "tui":
            gui = TUI()
        # TODO: add another GUI, maybe a TK window?
        case _:
            raise RuntimeError("Invalid GUI name")

    with visualizer_ctx as visualizer:
        print("Visualizer created.\n")

        with weart_ctx as weart:
            
            # Everything is initialized at this point
            simulation(engine, weart, visualizer, hand, gui, hands)