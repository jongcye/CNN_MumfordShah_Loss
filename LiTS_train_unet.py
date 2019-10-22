import time
import os
import numpy as np
from options.train_options import TrainOptions
from LiTS_getDatabase_unet import DataProvider_LiTS
from models.models import create_model
from util.visualizer import Visualizer
from math import *
from util import html
import ipdb


opt = TrainOptions().parse()

input_min       = 0
input_max       = 400

data_train = DataProvider_LiTS(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                               opt.dataroot, a_min=input_min, a_max=input_max, mode="train")
dataset_size = data_train.n_data
print('#training images = %d' % dataset_size)
training_iters = int(ceil(data_train.n_data/float(opt.batchSize)))

total_steps = 0
model = create_model(opt)
visualizer = Visualizer(opt)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    """ Train """
    for step in range(1, training_iters+1):
        batch_x, batch_y, path = data_train(opt.batchSize)
        data = {'A': batch_x, 'B': batch_y,
                'A_paths': path, 'B_paths': path}
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += 1
        model.set_input(data)
        model.optimize_parameters()

        if step % opt.display_step == 0:
            save_result = step % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, step, training_iters, errors, t, 'Train')

        if step % opt.plot_step == 0:
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, step / float(training_iters), opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()
