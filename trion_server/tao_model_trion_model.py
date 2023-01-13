import docker
from logger import  console_logger
import os
import time

class Triton_Server:
    def triton_server_start(self):
        client=docker.from_env()
        gpu=docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])
        self.container_name="tao_triton_server"
        if bool(client.containers.list(filters={"name":"tao_triton_server","status":"running"})):
            client.containers.list(filters={"name":"tao_triton_server","status":"running"})[0].stop()
            console_logger.debug("tao_triton_server alreay stop wait  to close")               
            client.containers.run("nvcr.io/nvidia/tritonserver:22.05-py3",command="tritonserver --model-repository=/opt/tritonserver/models --strict-model-config=false --grpc-infer-allocation-pool-size=16 --model-control-mode=explicit ",auto_remove=True,device_requests=[gpu],ipc_mode="host",ports={"8000":8000,"8001":8001,"8002":8002},volumes=[f"{os.getcwd()}/model_respository:/opt/tritonserver/models"],detach=True,name=self.container_name)
            console_logger.debug("Trion Server Already Start")
        else:
                client.containers.run("nvcr.io/nvidia/tritonserver:22.05-py3",command="tritonserver --model-repository=/opt/tritonserver/models --strict-model-config=false --grpc-infer-allocation-pool-size=16 --model-control-mode=explicit ",auto_remove=True,device_requests=[gpu],ipc_mode="host",ports={"8000":8000,"8001":8001,"8002":8002},volumes=[f"{os.getcwd()}/model_respository:/opt/tritonserver/models"],detach=True,name=self.container_name)
                console_logger.debug("Triton Server Started .... ") 
                time.sleep(5)    

    def triton_server_stop(self):
        client=docker.from_env()
        client.containers.list(filters={"name":"tao_triton_server","status":"running"})[0].stop()
        print(self.container_name,"stop")



if __name__=="__main__":
    tao_server_pipeline=Triton_Server()
    # tao_server_pipeline.create_triton_config_file()
    tao_server_pipeline.triton_server_start()




