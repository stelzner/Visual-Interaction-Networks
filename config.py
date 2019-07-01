class VinConfig:

    # Directories
    img_folder = "./img/"  # image folder
    traindata = 'billards_balls_training_data.mat'
    testdata = 'billards_balls_testing_data.mat'
    checkpoint_dir = "./checkpoint/"
    log_dir = "./log"

    # Model config
    visual = False  # If False, use states as input instead of images
    discount_factor = 0.995  # discount factor for loss from rollouts
    num_visible = 6  # Number of visible frames
    num_rollout = 16  # Number of rollout frames
    frame_step = 1  # Stepsize when observing frames
    cl = 16  # state code length per object

    # Training config
    load = True  # Load parameters from checkpoint file
    train_with_v = False
    num_epochs = 100
    batch_size = 128

    # Data config
    num_episodes = 1000  # The number of episodes
    num_frames = 100  # The number of frames per episode
    width = 32
    height = 32
    channels = 3
    num_obj = 3  # the number of object
