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

from tao_triton.python.postprocessing.postprocessor import Postprocessor


class PoseClassificationPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, sequences, output_path):
        """Initialize a post processor class for a poseclassification model.
        
        Args:
            batch_size (int): Number of sequences in the batch.
            sequences (list): List of sequences.
            output_path (str): Unix path to the output rendered sequences and labels.
        """
        self.batch_size = batch_size
        self.sequences = sequences
        self.output_path = output_path
        self.output_name = "fc_pred"

    def apply(self, output_tensors, this_id, render=True, batching=True, action_data=[]):
        """Apply the post processor to the outputs to the poseclassification outputs."""
        output_array = output_tensors.as_numpy(self.output_name)
        if len(output_array) != self.batch_size:
            raise Exception("expected {} results, got {}".format(
                batch_size, len(output_array)))

        # Include special handling for non-batching models
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Output a JSON file
        if len(action_data) > 0:
            for idx in range(self.batch_size):
                results = output_array[idx]
                current_idx = (int(this_id) - 1) * self.batch_size + idx
                if current_idx < len(self.sequences):
                    for b in range(len(action_data)):
                        for f in range(len(action_data[b]["batches"])):
                            frame_num = action_data[b]["batches"][f]["frame_num"]
                            for p in range(len(action_data[b]["batches"][f]["objects"])):
                                object_id = action_data[b]["batches"][f]["objects"][p]["object_id"]
                                segment_id = action_data[b]["batches"][f]["objects"][p]["segment_id"]
                                if segment_id == current_idx:
                                    if not batching:
                                        results = [results]
                                    result = results[0]
                                    if output_array.dtype.type == np.object_:
                                        cls = "".join(chr(x) for x in result).split(':')
                                    else:
                                        cls = result.split(':')
                                    action_data[b]["batches"][f]["objects"][p]["action"] = cls[2]

        # Output a text file of results
        else:
            output_file = os.path.join(self.output_path, "results.txt")
            with open(output_file, "a") as wfile:
                for idx in range(self.batch_size):
                    results = output_array[idx]
                    current_idx = (int(this_id) - 1) * self.batch_size + idx
                    if current_idx < len(self.sequences):
                        wfile.write("{}".format(current_idx))
                        if not batching:
                            results = [results]
                        for result in results:
                            if output_array.dtype.type == np.object_:
                                cls = "".join(chr(x) for x in result).split(':')
                            else:
                                cls = result.split(':')
                            wfile.write(
                                ", {:0.4f}({})={}".format(
                                    float(cls[0]), cls[1], cls[2]
                                )
                            )
                        wfile.write("\n")
                    else:
                        break
