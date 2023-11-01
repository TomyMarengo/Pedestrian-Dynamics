
from vidstab import VidStab

stabilizer = VidStab(kp_method='FAST', threshold=500, nonmaxSuppression=False)
stabilizer.stabilize(input_path='video/original_video.mov', output_path='video/stable_video.avi')