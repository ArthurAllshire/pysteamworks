from magicbot import StateMachine, state, timed_state
from components.geardepositiondevice import GearDepositionDevice
from components.gearalignmentdevice import GearAlignmentDevice
from networktables import NetworkTable
from components.vision import Vision
from components.range_finder import RangeFinder

class ManipulateGear(StateMachine):
    gearalignmentdevice = GearAlignmentDevice
    geardepositiondevice = GearDepositionDevice
    range_finder = RangeFinder
    sd = NetworkTable
    aligned = False
    vision = Vision
    
    @state(first=True, must_finish=True)
    def align_peg(self):
        # do something to align with the peg
        # now move to the next state
        #move forward
        self.sd.putString("state", "unloadingGear")
        self.put_dashboard()

    @timed_state(duration=0.5, next_state="forward_open", must_finish=True)
    def forward_closed(self):
        self.put_dashboard()
        self.geardepositiondevice.push_gear()

    @timed_state(duration=2.0, next_state="backward_open", must_finish=True)
    def forward_open(self):
        self.put_dashboard()
        self.geardepositiondevice.drop_gear()

    @timed_state(duration=0.5, next_state="backward_closed", must_finish=True)
    def backward_open(self):
        self.put_dashboard()
        self.geardepositiondevice.retract_gear()

    @state
    def backward_closed(self):
        self.put_dashboard()
        self.geardepositiondevice.lock_gear()

        self.sd.putString("state", "stationary")
        self.done()

    def put_dashboard(self):
        """Update all the variables on the smart dashboard"""
        self.sd.putNumber("vision_x", self.vision.x)
        self.sd.putNumber("smoothed_vision_x", self.vision.smoothed_x)
        self.sd.putNumber("vision_y", self.range_finder.getDistance())
