"""
Contains methods for YOLOv3 models for object detection.

Trained on COCO dataset.
"""

import cv2
import re
import numpy as np
from collections import defaultdict

from edge_ai_tester.core.model import ModelWrapper
from edge_ai_tester.datasets.open_images_dataset import DectObject


class TVMDarknetCOCOYOLOV3(ModelWrapper):
    def __init__(self, modelpath, dataset, from_file):
        self.thresh = 0.2
        self.iouthresh = 0.5
        super().__init__(modelpath, dataset, from_file)

    def load_model(self, modelpath):
        self.keyparams = {}
        self.perlayerparams = defaultdict(list)
        keyparamsrgx = re.compile(r'(width|height|classes)=(\d+)')
        perlayerrgx = re.compile(r'(mask|anchors|num)=((\d+,?)+)')

        with open(self.modelpath.with_suffix('.cfg'), 'r') as config:
            for line in config:
                line = line.replace(' ', '')
                res = keyparamsrgx.match(line)
                if res:
                    self.keyparams[res.group(1)] = int(res.group(2))
                    continue
                res = perlayerrgx.match(line)
                if res:
                    self.perlayerparams[res.group(1)].append(res.group(2))
        self.perlayerparams = {
            k: [np.array([int(x) for x in s.split(',')]) for s in v]
                for k, v in self.perlayerparams.items()
        }

    def prepare_model(self):
        self.load_model(self.modelpath)

    def get_input_spec(self):
        return {
            'data': (
                1, 3, self.keyparams['width'], self.keyparams['height']
            )
        }, 'float32'

    def preprocess_input(self, X):
        return np.array(X)

    def convert_to_dectobject(self, entry):
        # array x, y, w, h, classid, score
        x1 = entry[0] - entry[2] / 2
        x2 = entry[0] + entry[2] / 2
        y1 = entry[1] - entry[3] / 2
        y2 = entry[1] + entry[3] / 2
        return DectObject(
            self.dataset.classnames[entry[4]],
            x1, y1, x2, y2,
            entry[5]
        )

    def parse_outputs(self, data):
        # get all bounding boxes with objectness score over given threshold
        boxdata = []
        for i in range(len(data)):
            ids = np.asarray(np.where(data[i][:, 4, :, :] > self.thresh))
            ids = np.transpose(ids)
            if ids.shape[0] > 0:
                ids = np.append([[i]] * ids.shape[0], ids, axis=1)
                boxdata.append(ids)

        if len(boxdata) > 0:
            boxdata = np.concatenate(boxdata)

        # each entry in boxdata contains:
        # - layer id
        # - det id
        # - y id
        # - x id

        bboxes = []
        for box in boxdata:
            # x and y values from network are coordinates in a chunk
            # to get the actual coordinates, we need to compute
            # new_coords = (chunk_coords + out_coords) / out_resolution
            x = (box[3] + data[box[0]][box[1], 0, box[2], box[3]]) / data[box[0]].shape[2]
            y = (box[2] + data[box[0]][box[1], 1, box[2], box[3]]) / data[box[0]].shape[3]

            # width and height are computed using following formula:
            # w = anchor_w * exp(out_w) / input_w
            # h = anchor_h * exp(out_h) / input_h
            # anchors are computed based on dataset analysis
            maskid = self.perlayerparams['mask'][2 - box[0]][box[1]]
            anchors = self.perlayerparams['anchors'][box[0]][2 * maskid:2 * maskid + 2]
            w = anchors[0] * np.exp(data[box[0]][box[1], 2, box[2], box[3]]) / self.keyparams['width']
            h = anchors[1] * np.exp(data[box[0]][box[1], 3, box[2], box[3]]) / self.keyparams['height']

            # get objectness score
            objectness = data[box[0]][box[1], 4, box[2], box[3]]

            # get class with highest probability
            classid = np.argmax(data[box[0]][box[1], 5:, box[2], box[3]])

            # compute final class score (objectness * class probability
            score = objectness * data[box[0]][box[1], classid + 5, box[2], box[3]]

            # drop the bounding box if final score is below threshold
            if score < self.thresh:
                continue

            bboxes.append([x, y, w, h, classid, score])

        # sort the bboxes by score descending
        bboxes.sort(key=lambda x: x[5], reverse=True)

        bboxes = [self.convert_to_dectobject(b) for b in bboxes]
        
        # group bboxes by class to perform NMS sorting
        grouped_bboxes = defaultdict(list)
        for item in bboxes:
            grouped_bboxes[item.clsname].append(item)

        # perform NMS sort to drop overlapping predictions for the same class
        cleaned_bboxes = []
        for clsbboxes in grouped_bboxes.values():
            for i in range(len(clsbboxes)):
                # if score equals 0, the bbox is dropped
                if clsbboxes[i].score == 0:
                    continue
                # add current bbox to final results
                cleaned_bboxes.append(clsbboxes[i])

                # look for overlapping bounding boxes with lower probability
                # and IoU exceeding specified threshold
                for j in range(i + 1, len(clsbboxes)):
                    if self.dataset.compute_iou(clsbboxes[i], clsbboxes[j]) > self.iouthresh:
                        clsbboxes[j] = clsbboxes[j]._replace(score=0)
        return cleaned_bboxes

    def postprocess_outputs(self, y):
        # YOLOv3 has three stages of outputs
        # each one contains:
        # - real output
        # - masks
        # - biases
        
        # TVM-based model output provides 12 arrays
        # Those are subdivided into three groups containing
        # - actual YOLOv3 output
        # - masks IDs
        # - anchors
        # - 6 integers holding number of dects per cluster, actual output
        #   number of channels, actual output height and width, number of
        #   classes and unused parameter

        # iterate over each group
        lastid = 0
        outputs = []
        for i in range(3):
            # first extract the actual output
            # each output layer shape follows formula:
            # (BS, B * (4 + 1 + C), w / (8 * (i + 1)), h / (8 * (i + 1)))
            # BS is the batch size
            # w, h are width and height of the input image
            # the resolution is reduced over the network, and is 8 times
            # smaller in each dimension for each output
            # the "pixels" in the outputs are responsible for the chunks of
            # image - in the first output each pixel is responsible for 8x8
            # squares of input image, the second output covers objects from
            # 16x16 chunks etc.
            # Each "pixel" can predict up to B bounding boxes.
            # Each bounding box is described by its 4 coordinates,
            # objectness prediction and per-class predictions
            outshape = (
                self.dataset.batch_size,
                len(self.perlayerparams['mask'][i]),
                4 + 1 + self.dataset.numclasses,
                self.keyparams['width'] // (8 * 2 ** i),
                self.keyparams['height'] // (8 * 2 ** i)
            )

            outputs.append(
                y[lastid:(lastid + np.prod(outshape))].reshape(outshape)
            )

            # drop additional info provided in the TVM output
            # since it's all 4-bytes values, ignore the insides
            lastid += (
                np.prod(outshape)
                    + len(self.perlayerparams['mask'][i])
                    + len(self.perlayerparams['anchors'][i])
                    + 6  # layer parameters
            )

        # change the dimensions so the output format is
        # batches layerouts dets params width height
        perbatchoutputs = []
        for i in range(outputs[0].shape[0]):
            perbatchoutputs.append([
                outputs[0][i],
                outputs[1][i],
                outputs[2][i]
            ])
        result = []
        # parse the combined outputs for each image in batch, and return result
        for out in perbatchoutputs:
            result.append(self.parse_outputs(out))

        return result

    def convert_input_to_bytes(self, inputdata):
        return inputdata.tobytes()

    def convert_output_from_bytes(self, outputdata):
        return np.frombuffer(outputdata, dtype='float32')

    def get_framework_and_version(self):
        return ('darknet', 'alexeyab')
