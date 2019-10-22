import numpy as np
import matplotlib.pyplot as plt
import os
import ntpath
import time
import ipdb
from . import util
from . import html
import scipy.ndimage as ndimage


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save\

    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)


    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, iters, errors, t, mode):
        message = '(%s - epoch: %d | iters: %d/%d | time: %.3f) ' % (mode, epoch, i, iters, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        # short_path = ntpath.basename(image_path[0])
        # name = os.path.splitext(short_path)[0]

        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_data_plt(self, webpage, visuals, pred_gt, pred, image_path):
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            img = image_numpy[0].cpu().float().numpy()
            fig = plt.imshow(img[0, ...])
            fig.set_cmap('gray')
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)


        image_name = '%s_%s.png' % (name, 'pred_gt')
        save_path = os.path.join(image_dir, image_name)
        img = pred_gt.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        ims.append(image_name)
        txts.append('pred_gt')
        links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_result_fig(self, img, imgName, image_dir, image_path):
        # image_dir = webpage.get_image_dir()
        short_path = image_path.split('\\')
        name = short_path[-1]
        image_name = '%s_%s.png' % (name, imgName)
        save_path = os.path.join(image_dir, image_name)
        img = img.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

    def predict_output(self, label, visuals):
        label = np.array(label) > 0

        prediction_ = (visuals['fake_B'][0,0].cpu().float().numpy())
        # prediction = prediction_ >= (np.amax(prediction_))
        # prediction = prediction_ > ((np.amax(label_))/float(2))

        # Threshold: whole_F = 0.02 / core_F = -0.25 / enhance_C = -0.7
        prediction = prediction_ > -0.4
        return label, prediction

    def predict_output_v2(self, label, visuals):
        label = np.array(label) > 0

        prediction_ = (visuals['fake_B_CE'][0,0,:,:].cpu().float().numpy())
        # prediction = prediction_ >= (np.amax(prediction_))
        # prediction = prediction_ > ((np.amax(label_))/float(2))

        # Threshold: whole_F = 0.02 / core_F = -0.25 / enhance_C = -0.7
        prediction = prediction_ > 0.5
        return label, prediction

    def predict_output_seg_only(self, label, visuals):
        label = np.array(label) > 0

        prediction_ = (visuals['fake_B'][0,0,:,:].cpu().float().numpy())
        # prediction = prediction_ >= (np.amax(prediction_))
        # ipdb.set_trace()

        # Threshold: whole_F = -0.7
        prediction = prediction_ > 0.7
        return label, prediction

    def morphology_operation_liver(self, pred):
        pred = pred.astype(int)
        struct1 = ndimage.generate_binary_structure(2, 1)
        # pred = ndimage.binary_dilation(pred, structure=struct1, iterations=1)
        # struct = ndimage.generate_binary_structure(2, 2)
        # pred = ndimage.binary_erosion(pred, structure=struct, iterations=1)

        # pred = ndimage.binary_opening(pred, structure=struct1, iterations=2)
        pred = ndimage.binary_closing(pred, structure=struct1, iterations=2)
        # pred = ndimage.binary_opening(pred, structure=struct1, iterations=2)
        # pred = ndimage.binary_closing(pred, structure=struct1, iterations=1)
        # pred = ndimage.binary_fill_holes(pred)

        # pred = ndimage.binary_closing(pred, structure=struct, iterations=2)
        # pred = ndimage.binary_dilation(pred, structure=struct1, iterations=1)
        # ipdb.set_trace()
        return pred

    def morphology_operation_tumor(self, pred):
        pred = pred.astype(int)
        struct1 = ndimage.generate_binary_structure(2, 1)
        # pred = ndimage.binary_dilation(pred, structure=struct1, iterations=1)
        # struct = ndimage.generate_binary_structure(2, 2)
        # pred = ndimage.binary_erosion(pred, structure=struct, iterations=1)
        # pred = ndimage.binary_fill_holes(pred)
        pred = ndimage.binary_opening(pred, structure=struct1, iterations=2)
        pred = ndimage.binary_closing(pred, structure=struct1, iterations=1)
        pred = ndimage.binary_opening(pred, structure=struct1, iterations=2)

        # pred = ndimage.binary_opening(pred, structure=struct1, iterations=1)
        # pred = ndimage.binary_closing(pred, structure=struct1, iterations=1)

        # pred = ndimage.binary_closing(pred, structure=struct, iterations=2)
        # pred = ndimage.binary_dilation(pred, structure=struct1, iterations=1)
        # ipdb.set_trace()
        return pred


    def calculate_score(self, label, pred,  score):
        if score == "dice":
            intersect = float(np.sum(pred.astype(int) * label.astype(int)))
            union = float(np.sum(pred.astype(int)) + np.sum(label.astype(int)))
            return (2 * intersect) / union

        elif score == "iou":
            intersect = float(np.sum(pred.astype(int) * label.astype(int)))
            union = float(np.sum(np.logical_or(pred, label).astype(int)))
            return (intersect / union)

        elif score == "voe":
            intersect = float(np.sum(pred.astype(int) * label.astype(int)))
            union = float(np.sum(np.logical_or(pred, label).astype(int)))
            return 1 - (intersect / union)

        elif score == "sens":
            TP = float(np.sum(pred.astype(int) * label.astype(int)))
            FN = float(np.sum((~pred).astype(int) * (label).astype(int)))
            if TP + FN == 0:
                return 0
            return TP/(TP+FN)

        elif score == "ppv":
            TP = float(np.sum(pred.astype(int) * label.astype(int)))
            FP = float(np.sum(pred.astype(int) * (~label).astype(int)))
            if TP + FP == 0:
                return 0

            return TP / (TP + FP)






