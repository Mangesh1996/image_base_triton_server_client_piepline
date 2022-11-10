import subprocess
from logger import  console_logger
import os
import json
from math import ceil
import tao_client_run
import docker
import re
import requests
import time

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
        file_name=["/labels.txt","/resnet18_detector.etlt","/config.pbtxt","/clustering_config.prototxt","/pgie_config.txt"]
        mode_path=os.path.join(os.getcwd(),"models",self.model_name)

        for file in file_name:
            if os.path.isfile(mode_path+file):
                console_logger.debug((file,"are present"))
            else:
                console_logger.debug(f"Missing {file} file Kindly check config directory")
                return False
        return True


    def create_triton_config_file(self):
        try:
            if self.configruation_docker_path():
                os.makedirs(os.path.join(os.getcwd(),"model_respository",self.model_name,"1"),exist_ok=True)            
                dim="3,720,1280"
                with open(os.path.join(os.getcwd(),"models",self.model_name,"pgie_config.txt"),"r") as read_pige:
                    conf_read=read_pige.read()
                out_config=conf_read[re.search("output-blob-names",conf_read).end()+1:].split("\n")[0].replace(";",",")
                key_name=conf_read[re.search("tlt-model-key",conf_read).end()+1:].split("\n")[0]
                if os.path.isfile(os.path.join(os.getcwd(),"model_respository",self.model_name,"1","model.plan")):
                    console_logger.debug("Already Plan file present we are create again")
                else:
                    p_tao = subprocess.Popen(["tao","converter", "/opt/tao_models/model/resnet18_detector.etlt", "-k",key_name, "-d",dim, "-o",out_config, "-m", "16", "-e", "/opt/tritonserver/model_plane/model.plan"], stdout=subprocess.PIPE)
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
                class_name=clss_name.readlines()
            class_name=[cls.split("\n")[0] for cls in class_name ]
            config_format='''name: "'''+self.model_name+'''"
    platform: "tensorrt_plan"
    max_batch_size: 16
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
                label_name=label_name.readlines()
            label_name=[label.split("\n")[0] for label in label_name ]
            label_name=",".join(label_name)
            with open(os.path.join(os.getcwd(),"model_respository",self.model_name,"labels.txt"),"w") as labels:
                labels.write(label_name)
            return True
        except Exception as e:
            console_logger.debug(e)
            return False
    def triton_server_start(self):
        if self.create_triton_config_file and self.trion_config_file:
            client=docker.from_env()
            gpu=docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])
            self.container_name="tao_triton_server"
            if bool(client.containers.list(filters={"name":"tao_triton_server","status":"running"})):
                client.containers.list(filters={"name":"tao_triton_server","status":"running"})[0].stop()
                console_logger.debug("tao_triton_server alreay stop wait  to close")               
                client.containers.run("nvcr.io/nvidia/tritonserver:22.05-py3",command="tritonserver --model-repository=/opt/tritonserver/models --strict-model-config=false --grpc-infer-allocation-pool-size=16",auto_remove=True,device_requests=[gpu],ipc_mode="host",ports={"8000":8000,"8001":8001,"8002":8002},volumes=[f"{os.getcwd()}/model_respository:/opt/tritonserver/models"],detach=True,name=self.container_name)
            else:
                 client.containers.run("nvcr.io/nvidia/tritonserver:22.05-py3",command="tritonserver --model-repository=/opt/tritonserver/models --strict-model-config=false --grpc-infer-allocation-pool-size=16",auto_remove=True,device_requests=[gpu],ipc_mode="host",ports={"8000":8000,"8001":8001,"8002":8002},volumes=[f"{os.getcwd()}/model_respository:/opt/tritonserver/models"],detach=True,name=self.container_name)        
        else:
            console_logger.debug("Something wrong in config files or model plan file")
        time.sleep(5)

    def triton_server_stop(self):
        client=docker.from_env()
        client.containers.list(filters={"name":"tao_triton_server","status":"running"})[0].stop()
        print(self.container_name,"stop")


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
    def __init__(self, dicst,model_name):
        self.dicst=dicst
        self.model_name=model_name

    def deploy_healthcheck(self):
        try:
            r = requests.get("http://localhost:8000/v2/health/ready",timeout=(60))
            print(r.status_code)
            if r.status_code == 200 or r.status_code == 400:
                    self.tao_client()
                    return True
        except requests.exceptions.ConnectionError as e:
            console_logger.debug(e)
            self.deploy_healthcheck()

    def tao_client(self):
        with open(os.path.join(os.getcwd(),"model_respository",self.model_name,"labels.txt"),"r") as labels_name:
            label_name=labels_name.read().splitlines()
        label_names=",".join(label_name)
        self.dicst["postprocessing_config"]=f"{os.getcwd()}/models/{self.model_name}/clustering_config.prototxt"
        self.dicst["class_list"]=label_names
        self.dicst["model_name"]=self.model_name
        # self.dicts={"verbose":False,"async_set":False,"streaming":False,"model_name":"hat","model_version":str(1),"batch_size":1,"mode":"DetectNet_v2","url":"localhost:8000","protocol":"HTTP","image_filename":f"{os.getcwd()}/input_image","class_list":"with_hat,without_hat","output_path":"out","postprocessing_config":f"{os.getcwd()}/models/hat/clustering_config.prototxt","dataset_convert_config":""}
        tao_client_run.main(self.dicst)
        
    


if __name__=="__main__":
    tao_server_pipeline=Tao_Model_Tao_plan(model_name="hat")
    tao_server_pipeline.create_triton_config_file()
    tao_server_pipeline.triton_server_start()
    dicts={"verbose":False,"async_set":True,"streaming":True,"model_name":"","model_version":str(1),"batch_size":1,"mode":"DetectNet_v2","url":"localhost:8001","protocol":"grpc","image_filename":f"{os.getcwd()}/input_image","class_list":"","output_path":"out","postprocessing_config":"","dataset_convert_config":""}
    tao_client=Triton_Inference_Client(dicts,model_name="hat")
    tao_client.deploy_healthcheck()
    # # tao_client.tao_client()




