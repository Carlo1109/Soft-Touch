from weartsdk.WeArtHapticObject import WeArtHapticObject
from weartsdk.WeArtCommon import HandSide, ActuationPoint
from weartsdk.WeArtTemperature import WeArtTemperature
from weartsdk.WeArtTexture import WeArtTexture
from weartsdk.WeArtForce import WeArtForce
from weartsdk.WeArtEffect import TouchEffect
from weartsdk.WeArtTrackingCalibration import WeArtTrackingCalibration
from weartsdk.WeArtThimbleTrackingObject import WeArtThimbleTrackingObject

from weartsdk.WeArtClient import WeArtClient
import weartsdk.WeArtCommon
import time
import logging

class WeartConnector(object):
    def __init__(self):
        self._client = WeArtClient(weartsdk.WeArtCommon.DEFAULT_IP_ADDRESS, weartsdk.WeArtCommon.DEFAULT_TCP_PORT, log_level=logging.INFO)

        # Haptic for a finger
        self._hapticObject = WeArtHapticObject(self._client)
        self._hapticObject.handSideFlag = HandSide.Left.value
        self._hapticObject.actuationPointFlag = ActuationPoint.Index
        self._touchEffect = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self._hapticObject.AddEffect(self._touchEffect)
    
    # Start WeArtClient
    def __enter__(self):
        self._client.Run()
        self._client.Start()
        return self

    # Stop WeArtClient
    def __exit__(self, exc_type, exc_value, traceback):
        self._client.StopRawData()
        self._client.Stop()
        self._client.Close()
    
    # Start calibration, once simulation starts
    def calibrate(self):
        calibration = WeArtTrackingCalibration()
        self._client.AddMessageListener(calibration)
        self._client.StartCalibration()

        while(not calibration.getResult()):
            time.sleep(1)
        
        self._client.StopCalibration()
    
    # Creation of listeners for the finger
    def start_listeners(self):
        self._thumbThimbleTracking = WeArtThimbleTrackingObject(HandSide.Left, ActuationPoint.Index)
        self._client.AddThimbleTracking(self._thumbThimbleTracking)

    # Get finger closure
    def get_finger_closure(self):
        return self._thumbThimbleTracking.GetClosure()
    
    # Force application for haptic feedback
    def apply_force(self, force_value):
        self._touchEffect.Set(self._touchEffect.getTemperature(), WeArtForce(True, force_value), self._touchEffect.getTexture())
        self._hapticObject.UpdateEffects()