import os
import requests
from logger import console_logger
import tao_client_run
from multiprocessing import Process


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
    # def __init__(self, dicst,model_name):
    #     self.dicst=dicst
    #     self.model_name=model_name
    @staticmethod
    def deploy_healthcheck(dicst):
        try:
            r = requests.get("http://localhost:8000/v2/health/ready",timeout=(60))
            console_logger.debug({r.status_code:"Trion Server Succefully Running ..."})
            if r.status_code == 200 or r.status_code == 400:
                    Triton_Inference_Client.tao_client(dicst,model_name=dicst["model_name"])
                    return True
        except requests.exceptions.ConnectionError as e:
            console_logger.debug(e)
            Triton_Inference_Client.deploy_healthcheck(dicst)

    def tao_client(dicst,model_name):
        with open(os.path.join(os.getcwd(),"model_respository",model_name,"labels.txt"),"r") as labels_name:
            label_name=labels_name.read().splitlines()
        label_names=",".join(label_name)
        dicst["postprocessing_config"]=f"{os.getcwd()}/models/{model_name}/clustering_config.prototxt"
        dicst["class_list"]=label_names
        # self.dicts={"verbose":False,"async_set":False,"streaming":False,"model_name":"hat","model_version":str(1),"batch_size":1,"mode":"DetectNet_v2","url":"localhost:8000","protocol":"HTTP","image_filename":f"{os.getcwd()}/input_image","class_list":"with_hat,without_hat","output_path":"out","postprocessing_config":f"{os.getcwd()}/models/hat/clustering_config.prototxt","dataset_convert_config":""}
        tao_client_run.main(dicst)


        
if __name__=="__main__":
    image_path=[f"{os.getcwd()}/input_image",f"{os.getcwd()}/input_image"]
    dicts={"verbose":False,"async_set":True,"streaming":True,"model_name":"hat","model_version":str(1),"batch_size":32,"mode":"DetectNet_v2","url":"localhost:8001","protocol":"grpc","image_filename":image_path[0],"class_list":"","output_path":f"out","postprocessing_config":"","dataset_convert_config":""}
    client_call=Triton_Inference_Client()
    # client_call.deploy_healthcheck()
    client_call.deploy_healthcheck(dicts)