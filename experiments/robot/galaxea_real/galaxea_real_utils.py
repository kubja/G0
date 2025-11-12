from robot_interface_ros2 import GalaxeaInferfaceConfig, GalaxeaInterface
from wrapper import Wrapper

def get_wrapped_env(config: GalaxeaInferfaceConfig):
    interface = GalaxeaInterface(config=config)
    return Wrapper(interface)

def get_env(config: GalaxeaInferfaceConfig):
    interface = GalaxeaInterface(config=config)
    return interface