import subprocess
from logger import  console_logger
import os
import json
from math import ceil

import docker
import re
import requests
import time
import numpy as np
class Tao_Model_Tao_plan:
    def __init__(self,model_name):
        self.model_name=model_name
        
    def configruation_docker_path(self):
        try:
            if self.check_config_files():
                os.environ['KEY']="diycam_models"
                os.environ['NUM_GPUS']="1"

                #local paths
                os.environ["LOCAL_MODEL_DIR"]=os.path.join(os.getcwd(),"models",self.model_name)
                os.environ["LOCAL_SAVE_DIR"]=os.path.join(os.getcwd(),"model_respository",self.model_name,"1")
                #docker path
                os.environ["SAVE_DIR"]="/opt/tritonserver/model_plane/"
                os.environ["MODEL_DIR"]="/opt/tao_models/model"

                mounts_file=os.path.expanduser("~/.tao_mounts.json")

                drive_map={
                    "Mounts":[
                        {
                            "source":os.environ["LOCAL_MODEL_DIR"],
                            "destination":os.environ["MODEL_DIR"]
                        },
                        {
                            "source": os.environ["LOCAL_SAVE_DIR"],
                            "destination": os.environ["SAVE_DIR"]
                        },
                        
                    ],
                        "DockerOptions":{
                            "user": "{}:{}".format(os.getuid(), os.getgid())
                            # "entrypoint":""
                        }
                }
                with open(mounts_file,"w") as mfile:
                    json.dump(drive_map,mfile,indent=4)
                return True
            else:
                console_logger.debug("check configle files------------------")
                return False
        except:
            return False

    def check_config_files(self):
        file_name=["/labels.txt","/resnet18_detector.etlt","/clustering_config.prototxt","/pgie_config.txt"]
        mode_path=os.path.join(os.getcwd(),"models",self.model_name)
        for file in file_name:
            if os.path.isfile(mode_path+file):
                pass
            elif file =="/clustering_config.prototxt":
                console_logger.debug(f"This file creating ")
                self.clustering_config_creation(mode_path+file)
            else:
                console_logger.debug(f"Missing {file} file Kindly check config directory")
                return False
        console_logger.debug((file_name,"are present"))
        return True

    def clustering_config_creation(self,path):
        data='''linewidth: 4\nstride: 16\n'''
        with open(os.path.join(os.getcwd(),"models",self.model_name,"labels.txt"),"r") as label_name:
            label_names=[label.split("\n")[0] for label in label_name.readlines() ]
        for label_name in label_names:
            random_color=list(np.random.choice(range(255),size=3))
            label_name='''classwise_clustering_config{
            key: "'''+label_name+'''" 
            value:{
                coverage_threshold:0.005
                minimum_bounding_box_height:4
                dbscan_config{
                    dbscan_eps:0.3
                    dbscan_min_samples:0.05
                    dbscan_confidence_threshold: 0.9
                }
                bbox_color{
                    R:'''+str(random_color[0])+'''
                    G:'''+str(random_color[1])+'''
                    B:'''+str(random_color[2])+'''
                }
            }
        }
        '''
            data=data+label_name
        with open(path,"a")as writes_data:
            writes_data.write(data)

    def create_triton_config_file(self):
        try:
            if self.configruation_docker_path():
                os.makedirs(os.path.join(os.getcwd(),"model_respository",self.model_name,"1"),exist_ok=True)            
                dim="3,384,1248"
                with open(os.path.join(os.getcwd(),"models",self.model_name,"pgie_config.txt"),"r") as read_pige:
                    conf_read=read_pige.read()
                out_config=conf_read[re.search("output-blob-names",conf_read).end()+1:].split("\n")[0].replace(";",",")
                key_name=conf_read[re.search("tlt-model-key",conf_read).end()+1:].split("\n")[0]
                console_logger.debug(os.path.isfile(os.path.join(os.getcwd(),"model_respository",self.model_name,"1","model.plan")))
                if os.path.isfile(os.path.join(os.getcwd(),"model_respository",self.model_name,"1","model.plan")):
                    console_logger.debug("Already Plan file present ....")
                    return True
                else:
                    p_tao = subprocess.Popen(["tao","converter", "/opt/tao_models/model/resnet18_detector.etlt", "-k",key_name, "-o",out_config, "-m", "32","-i","nc","-d","3,384,1248","-p","Input,1x3x384x1248,8x3x384x1248,16x3x384x1248","-e", "/opt/tritonserver/model_plane/model.plan"], stdout=subprocess.PIPE)
                    p_tao.wait()
                    out, err = p_tao.communicate()
                    out = out.decode('UTF-8')
                    console_logger.debug(out)
                    dim=[int(i) for i in dim.split(",")]
                    self.trion_config_file(dim)
                return True
            else:
                console_logger.debug("check docker configration file")
                return False
        except Exception as e:
            console_logger.debug(e)
            return False

    def trion_config_file(self, dim):
        try:
            model_stride=16            
            with open(os.path.join(os.getcwd(),"models",self.model_name,"labels.txt")) as clss_name:
                class_name=[cls.split("\n")[0] for cls in clss_name.readlines() ]
            config_format='''name: "'''+self.model_name+'''"
    platform: "tensorrt_plan"
    max_batch_size: 32
    input [
    {
        name: "input_1"
        data_type: TYPE_FP32
        format: FORMAT_NCHW
        dims: [ '''+(str(dim[0]))+''', '''+(str(dim[1]))+''', '''+(str(dim[2]))+''' ]
    }
    ]
    output [
    {
        name: "output_bbox/BiasAdd"
        data_type: TYPE_FP32
        dims: [ '''+str(len(class_name)*4)+''', '''+str(ceil(dim[1]/model_stride))+''', '''+str(ceil(dim[2]/model_stride))+''' ]
    },
    {
        name: "output_cov/Sigmoid"
        data_type: TYPE_FP32
        dims: [ '''+str(len(class_name))+''', '''+str(ceil(dim[1]/model_stride))+''', '''+str(ceil(dim[2]/model_stride))+''' ]
    }
    ]
    dynamic_batching { }
            '''
            with open(os.path.join(os.getcwd(),"model_respository",self.model_name,"config.pbtxt"),"w")as config_write:
                config_write.write(config_format)
            with open(os.path.join(os.getcwd(),"models",self.model_name,"labels.txt"),"r") as label_name:
                label_name=",".join([label.split("\n")[0] for label in label_name.readlines()])
            with open(os.path.join(os.getcwd(),"model_respository",self.model_name,"labels.txt"),"w") as labels:
                labels.write(label_name)
            return True
        except Exception as e:
            console_logger.debug(e)
            return False

    def triton_server_start(self):
        if self.create_triton_config_file() and self.check_config_files:
            client=docker.from_env()
            gpu=docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])
            self.container_name="tao_triton_server"
            if bool(client.containers.list(filters={"name":"tao_triton_server","status":"running"})):
                client.containers.list(filters={"name":"tao_triton_server","status":"running"})[0].stop()
                console_logger.debug("tao_triton_server alreay stop wait  to close")               
                client.containers.run("nvcr.io/nvidia/tritonserver:22.05-py3",command="tritonserver --model-repository=/opt/tritonserver/models --strict-model-config=false --grpc-infer-allocation-pool-size=16 --model-control-mode=explicit ",auto_remove=True,device_requests=[gpu],ipc_mode="host",ports={"8000":8000,"8001":8001,"8002":8002},volumes=[f"{os.getcwd()}/model_respository:/opt/tritonserver/models"],detach=True,name=self.container_name)
                console_logger.debug("Trion Server Already Start")
            else:
                 client.containers.run("nvcr.io/nvidia/tritonserver:22.05-py3",command="tritonserver --model-repository=/opt/tritonserver/models --strict-model-config=false --grpc-infer-allocation-pool-size=16 ",auto_remove=True,device_requests=[gpu],ipc_mode="host",ports={"8000":8000,"8001":8001,"8002":8002},volumes=[f"{os.getcwd()}/model_respository:/opt/tritonserver/models"],detach=True,name=self.container_name)
                 console_logger.debug("Triton Server Started .... ") 
                 time.sleep(5)    
        else:
            console_logger.debug("Something wrong in config files or model plan file")
        

    def triton_server_stop(self):
        client=docker.from_env()
        client.containers.list(filters={"name":"tao_triton_server","status":"running"})[0].stop()
        print(self.container_name,"stop")



if __name__=="__main__":
    tao_server_pipeline=Tao_Model_Tao_plan(model_name="fire")
    # tao_server_pipeline.create_triton_config_file()
    tao_server_pipeline.triton_server_start()




