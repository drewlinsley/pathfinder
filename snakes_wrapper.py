import time
import sys
import numpy as np
import os
import snakes


class Args:
    def __init__(self,
                 contour_path = './contour', batch_id=0, n_images = 200000,
                 window_size=[256,256], antialias_scale = 4,
                 contour_length=15, distractor_length=5, num_distractor_contours=6, use_single_paddles=400,
                 max_target_contour_retrial = 4, max_distractor_contour_retrial = 4, max_paddle_retrial=2,
                 continuity = 1.4, paddle_length=5, paddle_thickness=1.5, paddle_margin_list=[4], paddle_contrast_list=[1.],
                 pause_display=False, save_images=True, save_gt=False, save_metadata=True):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.antialias_scale = antialias_scale

        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.num_distractor_contours = num_distractor_contours
        self.use_single_paddles = use_single_paddles

        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial

        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list # if multiple elements in a list, a number will be sampled in each IMAGE
        self.paddle_contrast_list = paddle_contrast_list # if multiple elements in a list, a number will be sampled in each PADDLE

        self.pause_display = pause_display
        self.save_images = save_images
        self.save_gt = save_gt
        self.save_metadata = save_metadata

t = time.time()
args = Args()
## Constraints
num_machines = int(sys.argv[1])
start_id = int(sys.argv[2])
args.batch_id = start_id + int(sys.argv[3]) - 1
total_images = int(sys.argv[4])
args.n_images = total_images/num_machines

dataset_root = '/gpfs/data/tserre/data/PSST/curvy_decoy'

args.contour_length = 12 # from 6~18
args.antialias_scale = 2
args.continuity = 1.6  # from 2.4~0.8
args.distractor_length = args.contour_length / 3
args.num_distractor_contours = int(33 * (9. / args.contour_length))
args.use_single_paddles = False

################################# DS: BASELINE
dataset_subpath = 'curv_baseline'
args.contour_path = os.path.join(dataset_root, dataset_subpath)
snakes.from_wrapper(args)
dataset_subpath = 'curv_baseline_neg'
args.contour_path = os.path.join(dataset_root, dataset_subpath)
args.contour_length = args.contour_length / 3
args.num_distractor_contours += 3
snakes.from_wrapper(args)

################################# DS: snake length
# length from 10 to 30
for cl in [6, 18]:
    dataset_subpath = 'curv_contour_length_' + str(cl)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.contour_length = cl
    args.distractor_length = cl/3
    args.num_distractor_contours = int(30*(9./args.contour_length))
    snakes.from_wrapper(args)
    dataset_subpath = 'curv_contour_length_' + str(cl) + '_neg'
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.contour_length = cl/3
    args.num_distractor_contours += 3
    snakes.from_wrapper(args)
args.contour_length = 15
args.num_distractor_contours = 0

################################# DS: snake inter-paddle continuity
# continuity from 1 to 2.6 (expect occasional failures at high values)
for ct in [2.4, 2.0, 1.2, 1.0]:
    dataset_subpath = 'curv_continuity_' + str(ct)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.continuity = ct
    #snakes.from_wrapper(args)
args.continuity = 1.4

################################# DS: (REST OF THE 2-way MATRIX)
# NOT IMPLEMENTED YET
for cl in [10, 20, 25, 30]:
    for ct in [1.0, 1.8, 2.2, 2.6]:
        print('not implemented.')

################################# DS: BASELINE
dataset_subpath = 'curv_negative_scatter'
args.contour_path = os.path.join(dataset_root, dataset_subpath)
args.contour_length = 1
args.distractor_length = 1
# snakes.from_wrapper(args)

################################# DS: snake paddle contrast
contrast_interval = 0.02
# if nc = 10, contrast is sampled from 11 values (nc+1)
for nc in [10, 20, 30, 40]:
    # Positive
    dataset_subpath = 'num_paddle_contrasts_' + str(nc)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.paddle_contrast_list = list(np.linspace(1., 1.-contrast_interval*nc, num=nc+1, endpoint=True))
    #snakes.from_wrapper(args)
    # Negative
    dataset_subpath = 'num_paddle_contrasts_' + str(nc) + '_neg'
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.contour_length = 1
    #snakes.from_wrapper(args)
    args.contour_length = 15
args.paddle_contrast_list = [1.]

################################# DS: snake margin range
contrast_interval = 0.02
# if nc = 10, contrast is sampled from 11 values (nc+1)
for pm in [[3,4,5], [2,3,4,5,6], [1,2,3,4,5,6,7]]:
    # Positive
    dataset_subpath = 'num_paddle_margins_' + str(len(pm))
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.paddle_margin_list = pm
    #snakes.from_wrapper(args)
    # Negative
    dataset_subpath = 'num_paddle_margins_' + str(len(pm)) + '_neg'
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.contour_length = 1
    #snakes.from_wrapper(args)
    args.contour_length = 15
args.paddle_margin_list = [4]



elapsed = time.time() - t
print('n_totl_imgs (per condition) : ', str(total_images))
print('ELAPSED TIME OVER ALL CONDITIONS : ', str(elapsed))