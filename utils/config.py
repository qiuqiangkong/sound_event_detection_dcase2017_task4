sample_rate = 32000
audio_duration = 10     # Audio clips have durations of 10 seconds
audio_length = sample_rate * audio_duration

mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
frames_per_second = sample_rate // hop_size
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

# ID of classes
ids = ['/m/0284vy3', '/m/05x_td', '/m/02mfyn', '/m/02rhddq', '/m/0199g', 
       '/m/06_fw', '/m/012n7d', '/m/012ndj', '/m/0dgbq', '/m/04qvtq', 
       '/m/03qc9zr', '/m/0k4j', '/t/dd00134', '/m/01bjv', '/m/07r04', 
       '/m/04_sv', '/m/07jdr']

# Name of classes
labels = ['Train horn', 'Air horn, truck horn', 'Car alarm', 'Reversing beeps', 
       'Bicycle', 'Skateboard', 'Ambulance (siren)', 
       'Fire engine, fire truck (siren)', 'Civil defense siren', 
       'Police car (siren)', 'Screaming', 'Car', 'Car passing by', 'Bus', 
       'Truck', 'Motorcycle', 'Train']
       
# Number of training samples of sound classes
samples_num = [441, 407, 273, 337, 624, 2399, 2399, 1506, 744, 2020, 1617, 
    25744, 3724, 3745, 7090, 3291, 2301]
    
classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}