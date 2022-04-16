# import azureml
# from azureml.core import Workspace
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.runconfig import DockerConfiguration
from azureml.core.environment import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Dataset, Datastore
from azureml.core import ScriptRunConfig
import os, shutil

# !pip install -q datasets
# !pip install -q transformers


#Standard_NC6_Promo

ws = Workspace.from_config(path='config.json')
print(ws.name, ws.location, ws.resource_group, sep='\t')

# gpu-nc6-promo
# STANDARD_NC6_Promo
# cluster_name = "gpu-cluster" # STANDARD_NC6
cluster_name = "gpu-nc6-promo" # STANDARD_NC6_Promo

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6_Promo',
                                                           max_nodes=2)    
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# setup the python env on the cluster
# restored_environment = Environment.get(workspace=ws,name="tensorflow-2.4-gpu")
# if restored_environment is not None:
#     tf_env = restored_environment
# else:

# tf_env = Environment(name='tensorflow-2.4-gpu')
tf_env = Environment.get(workspace=ws,name='AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu')

# define packages for image
cd = CondaDependencies.create(pip_packages=['azureml-dataset-runtime[pandas,fuse]', 
                                            'azureml-defaults', 
                                            'tensorflow==2.4.0',
                                            'matplotlib', 'scikit-learn', 'transformers', 'datasets'],)

tf_env.python.conda_dependencies = cd

# # Specify a docker image to use.
# tf_env.docker.base_image = (
#     "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04"
# )
# AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu


# Register environment to re-use later
# tf_env = tf_env.register(workspace = ws)

# docker_config = DockerConfiguration(use_docker=True)

# Folder setup
project_folder = './src'
# os.makedirs(project_folder, exist_ok=True)
# shutil.copy('train.py', project_folder)

experiment = Experiment(workspace=ws, name='sri-experiment')

# config = ScriptRunConfig(source_directory=project_folder, script='train.py', compute_target=compute_target, environment=tf_env,
#                       docker_runtime_config=docker_config)
config = ScriptRunConfig(source_directory=project_folder, script='lymebertweet.py', compute_target=compute_target, environment=tf_env)

run = experiment.submit(config)
aml_url = run.get_portal_url()
# print(aml_url)
# If we want to see the log directly in terminal in case of failures instead of going to the azure portal
run.wait_for_completion(show_output=True) 



