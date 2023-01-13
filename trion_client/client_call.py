import os
import requests
from logger import console_logger
import tao_client_run
import json
import re
import requests
from math import ceil
import numpy as np
import subprocess

import tritonclient.grpc as grpcclient
import sys
sys.path.insert(1,"..")
from trion_server.tao_model_trion_model import Triton_Server

class Tao_Model_Tao_plan():
    def __init__(self, dicst,model_name):
        self.dicst=dicst
        self.model_name=model_name
    def configruation_docker_path(self):
        try:
            if self.check_config_files():
                os.environ['KEY']="diycam_models"
                os.environ['NUM_GPUS']="1"
                os.makedirs(os.path.join(os.getcwd(),"model_respository",self.model_name,"1"),exist_ok=True)
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
    def check_triton_files(self):
        file_name=["/1/model.plan","/config.pbtxt","/labels.txt"]
        triton_model_path=os.path.join(os.getcwd(),"model_respository",self.model_name)
        missing_file=set()
        for file in file_name:
            if os.path.isfile(triton_model_path+file):
                pass
            else:
                missing_file.add(file)
        if len(missing_file)!=0:
            console_logger.debug("triton files not present build plan files")
            self.create_triton_plan_file()
        else:
            console_logger.debug("All Triton config file Present!!")
            requests.request("POST",f"http://localhost:8000/v2/repository/models/{dicts['model_name']}/load")
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
        self.create_triton_plan_file()

    def create_triton_plan_file(self):
        try:
            if self.configruation_docker_path():                
                with open(os.path.join(os.getcwd(),"models",self.model_name,"pgie_config.txt"),"r") as read_pige:
                    conf_read=read_pige.read()
                # dims=conf_read[re.search("uff-input-dims",conf_read)]
                # console_logger.debug((dims))
                dim="3,384,1248"                
                out_config=conf_read[re.search("output-blob-names",conf_read).end()+1:].split("\n")[0].replace(";",",")
                key_name=conf_read[re.search("tlt-model-key",conf_read).end()+1:].split("\n")[0]
                # console_logger.debug(os.path.isfile(os.path.join(os.getcwd(),"model_respository",self.model_name,"1","model.plan")))
                if os.path.isfile(os.path.join(os.getcwd(),"model_respository",self.model_name,"1","model.plan")):
                    console_logger.debug("Already Plan file present ....")
                    dim=[int(i) for i in dim.split(",")]
                    self.trion_config_file(dim)
                    return True
                else:
                    p_tao = subprocess.Popen(["tao","converter", "/opt/tao_models/model/resnet18_detector.etlt", "-k",key_name, "-o",out_config, "-m", "32","-d","3,384,1248","-p","Input,1x3x384x1248,8x3x384x1248,16x3x384x1248","-e", "/opt/tritonserver/model_plane/model.plan"], stdout=subprocess.PIPE)
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
            if self.dicst["mode"].lower()=="yolov3":                
                model_load=requests.request("POST",f"http://localhost:8000/v2/repository/models/{dicts['model_name']}/load")
                console_logger.debug((model_load.status_code==200))
                if model_load.status_code==200:
                    triton_client=grpcclient.InferenceServerClient(url="localhost:8001")
                    model_config=triton_client.get_model_config(model_name=self.dicst["model_name"],model_version="1", as_json=True)
                    config='''name:"'''+model_config["config"]["name"]+'''"
                        platform:"'''+model_config["config"]["platform"]+'''"
                        max_batch_size: '''+str(model_config["config"]["max_batch_size"])+'''
                        input [
                        {
                            name: "'''+model_config["config"]["input"][0]["name"]+'''"
                            data_type: '''+model_config["config"]["input"][0]["data_type"]+'''
                            format: FORMAT_NCHW
                            dims: ['''+",".join(str(e) for e in model_config["config"]["input"][0]["dims"])+''']
                        }
                        ]
                        output [
                        {
                            name:"'''+model_config["config"]["output"][0]["name"]+'''"
                            data_type:'''+model_config["config"]["output"][0]["data_type"]+'''
                            dims: ['''+",".join(str(e) for e in model_config["config"]["output"][0]["dims"])+''']
                        },
                        {
                            name:"'''+model_config["config"]["output"][1]["name"]+'''"
                            data_type:'''+model_config["config"]["output"][1]["data_type"]+'''
                            dims: ['''+",".join(str(e) for e in model_config["config"]["output"][1]["dims"])+''']
                        },
                        {
                            name:"'''+model_config["config"]["output"][2]["name"]+'''"
                            data_type:'''+model_config["config"]["output"][2]["data_type"]+'''
                            dims: ['''+",".join(str(e) for e in model_config["config"]["output"][2]["dims"])+''']
                        },
                        {
                            name:"'''+model_config["config"]["output"][3]["name"]+'''"
                            data_type:'''+model_config["config"]["output"][3]["data_type"]+'''
                            dims: ['''+",".join(str(e) for e in model_config["config"]["output"][3]["dims"])+''']
                        }
                        ]
                        dynamic_batching { }
                            '''
                    with open(os.path.join(os.getcwd(),"model_respository",self.dicst["model_name"],"config.pbtxt"),"w")as config_yolo:
                        config_yolo.write(config)
                elif self.dicst["mode"].lower()=="detectnet_v2":
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
                requests.request("POST",f"http://localhost:8000/v2/repository/models/{dicts['model_name']}/unload")
                requests.request("POST",f"http://localhost:8000/v2/repository/models/{dicts['model_name']}/load")
                return True
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            console_logger.debug(exc_type, fname, exc_tb.tb_lineno)
            console_logger.debug(e)
            return False

        
class Triton_Inference_Client():
    '''verbose(boolean) = get the logs,
    asynce_set(boolean)= ,
    streaming(boolean)= ,
    MODEL_NAME(str)= Model name,
    MODEL_VERSION(int)=version,
    BATCH_SIZE(int)=batch size,
    mode(str)= choice any one {Classification,DetectNet_v2,LPRNet,YOLOv3,Peoplesegnet,Retinanet,Multitask_classification,Pose_classification}
    URL(str)= localhost:8000 depend on protocal(http or grpc)
    PROTOCOL(str)=
    class_list(str)=list of class eg cat,dog
    output_path(str)=path of save output
    postprocessing_config=path of clustering config.prototxt 
    dataset_convert_config= ,
    image_filename(str)=path of input images files    
    '''
   
    @staticmethod
    def deploy_healthcheck(dicts):
        try:
            tao_models=Tao_Model_Tao_plan(dicts,model_name=dicts["model_name"])
            r = requests.get("http://localhost:8000/v2/health/ready",timeout=(20))
            console_logger.debug({r.status_code:"Trion Server Succefully Running ..."})
            if r.status_code == 200:
                    tao_models.check_triton_files()
                    Triton_Inference_Client.tao_client(dicts,model_name=dicts["model_name"])
                    return True
            # else:
            #     tritonserver=Triton_Server()
            #     tritonserver.triton_server_start()
        except requests.exceptions.ConnectionError as e:
            console_logger.debug(e)
            # Triton_Inference_Client.deploy_healthcheck(dicts)   
            tritonserver=Triton_Server()
            tritonserver.triton_server_start()

    def tao_client(dicst,model_name):
        with open(os.path.join(os.getcwd(),"model_respository",model_name,"labels.txt"),"r") as labels_name:
            label_name=labels_name.read().splitlines()
        label_names=",".join(label_name)
        dicst["postprocessing_config"]=f"{os.getcwd()}/../trion_server/models/{model_name}/clustering_config.prototxt"
        dicst["class_list"]=label_names
        # self.dicts={"verbose":False,"async_set":False,"streaming":False,"model_name":"hat","model_version":str(1),"batch_size":1,"mode":"DetectNet_v2","url":"localhost:8000","protocol":"HTTP","image_filename":f"{os.getcwd()}/input_image","class_list":"with_hat,without_hat","output_path":"out","postprocessing_config":f"{os.getcwd()}/models/hat/clustering_config.prototxt","dataset_convert_config":""}
        tao_client_run.main(dicst)


        
if __name__=="__main__":
    image_path=[f"{os.getcwd()}/input_image",f"{os.getcwd()}/input_image"]
    dicts={"verbose":False,"async_set":True,"streaming":True,"model_name":"yolov3","model_version":str(1),"batch_size":16,"mode":"yolov3","url":"localhost:8001","protocol":"grpc","image_filename":image_path[0],"class_list":"","output_path":f"out","postprocessing_config":"","dataset_convert_config":""}
    
    client_call=Triton_Inference_Client()
    # client_call.deploy_healthcheck()
    client_call.deploy_healthcheck(dicts)
    