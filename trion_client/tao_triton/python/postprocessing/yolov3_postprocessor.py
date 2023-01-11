# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Simple class to run post processing of Triton Inference outputs."""

import os
import numpy as np
from PIL import Image, ImageDraw

from tao_triton.python.postprocessing.postprocessor import Postprocessor
from tao_triton.python.postprocessing.utils import pool_context


    
def trt_output_process_fn(y_encoded):
    "function to process TRT model output."
    keep_k, boxes, scores, cls_id = y_encoded
    result = []
    
    for idx, k in enumerate(keep_k.reshape(-1)):
        mul = np.array([960,
                        544,
                        960,
                        544])
        
        loc = boxes.reshape(-1, 4)[idx][:k] * mul
        cid = cls_id.reshape(-1, 1)[idx][:k]
        conf = scores.reshape(-1, 1)[idx][:k]
        result.append(np.concatenate((cid, conf, loc), axis=-1))
    return result


class YOLOv3Postprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self,batch_size, frames, output_path, data_format,class_list):
        """Initialize a post processor class for a yolov3 model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        self.class_list=class_list
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_names = ["BatchedNMS",
                             "BatchedNMS_1",
                             "BatchedNMS_2",
                             "BatchedNMS_3"]
        self.threshold = 0.8
        self.keep_aspect_ratio = True
        self.class_mapping = {index:class_name for index,class_name in enumerate(self.class_list)}

    def _get_bbox_and_kitti_label_single_img(
        self, img, img_ratio, y_decoded,
        is_draw_img, is_kitti_export
    ):
        """helper function to draw bbox on original img and get kitti label on single image.

        Note: img will be modified in-place.
        """
        kitti_txt = ""
        draw = ImageDraw.Draw(img)
        color_list = ['Black', 'Red', 'Blue', 'Gold', 'Purple']
        for i in y_decoded:
            if float(y_decoded[1]) < self.threshold:
                continue

            if self.keep_aspect_ratio:
                y_decoded[2:6] *= img_ratio
            else:
                orig_w, orig_h = img.size
                ratio_w = float(orig_w) / self.model_input_width
                ratio_h = float(orig_h) / self.model_input_height
                y_decoded[2] *= ratio_w
                y_decoded[3] *= ratio_h
                y_decoded[4] *= ratio_w
                y_decoded[5] *= ratio_h
            
            if is_kitti_export:
                kitti_txt += self.class_mapping[int(y_decoded[0])] + ' 0 0 0 ' + \
                    ' '.join([str(x) for x in y_decoded[2:6]])+' 0 0 0 0 0 0 0 ' + str(y_decoded[1])+'\n'

            if is_draw_img:
                draw.rectangle(
                    ((y_decoded[2], y_decoded[3]), (y_decoded[4], y_decoded[5])),
                    outline=color_list[int(y_decoded[0]) % len(color_list)]
                )
                # txt pad
                draw.rectangle(((y_decoded[2],y_decoded[3]), (y_decoded[2] + 100, y_decoded[3]+10)),
                               fill=color_list[int(y_decoded[0]) % len(color_list)])

                draw.text((y_decoded[2], y_decoded[3]), "{0}: {1:.2f}".format(self.class_mapping[int(y_decoded[0])], y_decoded[1]))


        return img, kitti_txt


    def apply(self, results, this_id, render=True, batching=True):
        """Apply the post processor to the outputs to the yolov3 outputs."""

        #output_array = {}
        output_array = []      
  
        for output_name in self.output_names:
            #print(results.as_numpy(output_name))
            output_array.append(results.as_numpy(output_name))

        #y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in output_array]
        y_pred = [i.reshape(1, -1)[:1] for i in output_array]

        y_pred_decoded = trt_output_process_fn(y_pred)

        
        for image_idx in range(self.batch_size):
            current_idx = (int(this_id) - 1) * self.batch_size + image_idx
            if current_idx >= len(self.frames):
                break
            current_frame = self.frames[current_idx]
            filename = os.path.basename(current_frame._image_path)

            img = Image.open(current_frame._image_path)
            orig_w, orig_h = img.size
            ratio = min(current_frame.w/float(orig_w), current_frame.h/float(orig_h))
            new_w = int(round(orig_w*ratio))
            ratio = float(orig_w)/new_w

            output_label_file = os.path.join(
                self.output_path, "infer_labels",
                "{}.txt".format(os.path.splitext(filename)[0])
            )
            output_image_file = os.path.join(
                self.output_path, "infer_images",
                "{}.jpg".format(os.path.splitext(filename)[0])
            )
            if not os.path.exists(os.path.dirname(output_label_file)):
                os.makedirs(os.path.dirname(output_label_file))
            if not os.path.exists(os.path.dirname(output_image_file)):
                os.makedirs(os.path.dirname(output_image_file))

            img, kitti_txt = self._get_bbox_and_kitti_label_single_img(img, ratio, y_pred_decoded[0], output_image_file, output_label_file)

            img.save(output_image_file)

            open(output_label_file, 'w').write(kitti_txt)

