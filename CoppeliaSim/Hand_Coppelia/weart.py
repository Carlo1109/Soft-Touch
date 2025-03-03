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

        # Haptic for the thumb
        self._hapticObjectThumb = WeArtHapticObject(self._client)
        self._hapticObjectThumb.handSideFlag = HandSide.Left.value
        self._hapticObjectThumb.actuationPointFlag = ActuationPoint.Thumb
        self._touchEffectThumb = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self._hapticObjectThumb.AddEffect(self._touchEffectThumb)

        # Haptic for the index
        self._hapticObjectIndex = WeArtHapticObject(self._client)
        self._hapticObjectIndex.handSideFlag = HandSide.Left.value
        self._hapticObjectIndex.actuationPointFlag = ActuationPoint.Index
        self._touchEffectIndex = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self._hapticObjectIndex.AddEffect(self._touchEffectIndex)

        # Haptic for the middle        
        self._hapticObjectMiddle = WeArtHapticObject(self._client)
        self._hapticObjectMiddle.handSideFlag = HandSide.Left.value
        self._hapticObjectMiddle.actuationPointFlag = ActuationPoint.Middle
        self._touchEffectMiddle = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self._hapticObjectMiddle.AddEffect(self._touchEffectMiddle)
    
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

    # Creation of listeners for each finger
    def start_listeners(self):
        self._thumbThimbleTracking = WeArtThimbleTrackingObject(HandSide.Left, ActuationPoint.Thumb)
        self._indexThimbleTracking = WeArtThimbleTrackingObject(HandSide.Left, ActuationPoint.Index)
        self._middleThimbleTracking = WeArtThimbleTrackingObject(HandSide.Left, ActuationPoint.Middle)
        self._client.AddThimbleTracking(self._thumbThimbleTracking)
        self._client.AddThimbleTracking(self._indexThimbleTracking)
        self._client.AddThimbleTracking(self._middleThimbleTracking)

    # Get finger closure
    def get_closure(self, finger):
        closure = 0
        match finger:
            case "index": 
                closure =  self._indexThimbleTracking.GetClosure()
            case "middle": 
                closure = self._middleThimbleTracking.GetClosure()
            case "thumb":
                closure = self._thumbThimbleTracking.GetClosure()
        return closure, finger
    
    # Force application for haptic feedback
    def apply_force(self, force_value, finger):
        match finger:
            case "thumb":
                self._touchEffectThumb.Set(self._touchEffectThumb.getTemperature(), WeArtForce(True, force_value), self._touchEffectThumb.getTexture())
                self._hapticObjectThumb.UpdateEffects()
            case "index":
                self._touchEffectIndex.Set(self._touchEffectIndex.getTemperature(), WeArtForce(True, force_value), self._touchEffectIndex.getTexture())
                self._hapticObjectIndex.UpdateEffects()
            case "middle":
                self._touchEffectMiddle.Set(self._touchEffectMiddle.getTemperature(), WeArtForce(True, force_value), self._touchEffectMiddle.getTexture())
                self._hapticObjectMiddle.UpdateEffects()

            