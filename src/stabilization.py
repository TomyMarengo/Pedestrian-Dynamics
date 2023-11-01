
from vidstab import VidStab

stabilizer = VidStab(kp_method='FAST', threshold=500, nonmaxSuppression=False)
stabilizer.stabilize(input_path='../Targets_720_0_1000.mov', output_path='stable_video.avi')