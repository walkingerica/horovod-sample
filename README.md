# horovod-sample


Configuring CUDA Versions

You can verify the CUDA version by running NVIDIA's nvcc program.

nvcc --version

You can select and verify a particular CUDA version with the following bash command:

sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda


原本的 PATH：
/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin/:/usr/local/cuda-11.8/bin/:/usr/local/cuda-11.8/include:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

原本的： LD_LIBRARY_PATH
/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda-12.1/lib:/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1:/usr/local/cuda-12.1/targets/x86_64-linux/lib/:/usr/local/cuda-12.1/extras/CUPTI/lib64:/usr/local/lib:/usr/lib

改成：
/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda-11.7/lib:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:/usr/local/cuda-11.7/extras/CUPTI/lib64:/usr/local/lib:/usr/lib

NCCL version 2.16.2

sudo apt install libnccl2 libnccl-dev
sudo apt install libnccl2=2.16.2-1+cuda11.8 libnccl-dev=2.16.2-1+cuda11.8 

pip3 install -U virtualenv
virtualenv horovodenv
source horovodenv/bin/activate

pip install tensorflow==2.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple 

Horovod Installation 
pip install horovod[tensorflow,keras,pytorch,mxnet,spark]


https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-tempinstance
https://aws.amazon.com/cn/blogs/machine-learning/horovod-mxnet-distributed-training/

mpirun -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py

mpirun -np 2 \
	-H node1:1,node2:1
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
