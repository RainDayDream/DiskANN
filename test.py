from ctypes import sizeof
import numpy as np
import diskannpy as dap
import struct
import os
import psutil
import GPUtil
import time
import threading
from datetime import datetime
import pandas as pd


def monitor_resources_real_time(process, monitoring_data, interval=1):
    """
    实时监控进程的资源使用情况(CPU, 内存, I/O, GPU)。
    :param process: 要监控的进程对象(psutil.Process())
    :param monitoring_data: 用于存储监控数据的 DataFrame
    :param interval: 监控间隔时间(秒)
    """
    while not stop_event.is_set():
        # 获取 CPU 和内存使用情况
        cpu_percent = process.cpu_percent(interval=interval)
        memory_info = process.memory_info().rss / (1024 * 1024)
        io_counters = process.io_counters()
        io_read_bytes = io_counters.read_bytes
        io_write_bytes = io_counters.write_bytes

        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        sys_info_data = {
            "时间": formatted_time,
            "CPU 使用率": cpu_percent,
            "内存使用 (MB)": memory_info,
            "I/O 读取字节数": io_read_bytes,
            "I/O 写入字节数": io_write_bytes
        }

        # 添加每个 GPU 的数据
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_name = f"{gpu.name}-{gpu.uuid}"
            sys_info_data[f"{gpus_id[gpu_name]}-负载 (%)"] = gpu.load * 100
            sys_info_data[f"{gpus_id[gpu_name]}-显存使用 (MB)"] = gpu.memoryUsed
            sys_info_data[f"{gpus_id[gpu_name]}-显存总量 (MB)"] = gpu.memoryTotal
            sys_info_data[f"{gpus_id[gpu_name]}-显存用量 (%)"] = gpu.memoryUsed / gpu.memoryTotal * 100

        monitoring_data.append(sys_info_data)

def monitor_function(func, csv_file_path, *args, **kwargs):
    # 获取当前进程对象
    process = psutil.Process()

    # 定义表头列表
    sys_info_header = [
        "时间", "CPU 使用率", "内存使用 (MB)", "I/O 读取字节数", "I/O 写入字节数"
    ]

    global gpus_id
    gpus_id = {}
    # 为每个 GPU 添加专属的列
    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        gpu_name = f"{gpu.name}-{gpu.uuid}"
        gpu_simplified_name = f"GPU{i}"
        gpus_id[gpu_name] = gpu_simplified_name
        sys_info_header.extend([
            f"{gpus_id[gpu_name]}-负载 (%)",
            f"{gpus_id[gpu_name]}-显存使用 (MB)",
            f"{gpus_id[gpu_name]}-显存总量 (MB)",
            f"{gpus_id[gpu_name]}-显存用量 (%)"
        ])

    # 用于存储监控数据的列表
    monitoring_data = []

    # 创建停止事件
    global stop_event
    stop_event = threading.Event()

    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_resources_real_time, args=(process, monitoring_data))
    monitor_thread.start()

    # 执行目标函数
    result = func(*args, **kwargs)

    # 停止监控线程
    stop_event.set()
    monitor_thread.join()

    # 将监控数据转换为 DataFrame
    df = pd.DataFrame(monitoring_data, columns=sys_info_header)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(csv_file_path, index=False, encoding="utf-8")

    return result




def read_fvecs(file_path):
    nodes = 0
    with open(file_path, 'rb') as f:
        points = struct.unpack('i', f.read(4))[0]  # 读取向量个数
        dim = struct.unpack('i', f.read(4))[0]    # 读取向量维度
        file_size = os.path.getsize(file_path)
        expected_size = 8 + points * dim * 4
        if file_size != expected_size:
            raise ValueError(f"File size {file_size} does not match expected size {expected_size}.")
        else:
            print(f"points: {points}, dim: {dim}")
        print(f"points: {points}, dim: {dim}")
        while nodes < points:  # 读取向量的各个分量
            vector = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            yield vector
            # print(f"node: {nodes}, vector: {vector}")
            nodes += 1
        data = f.read(1)
        if not data:  # 检测 EOF
            print("End of file reached.")
        else:
            raise ValueError(f"Extra data found in file: {data}")
        # exit()


def read_bvecs(file_path):
    nodes = 0
    with open(file_path, 'rb') as f:
        points = struct.unpack('i', f.read(4))[0]  # 读取向量个数
        dim = struct.unpack('i', f.read(4))[0]    # 读取向量维度
        file_size = os.path.getsize(file_path)
        expected_size = 8 + points * dim  # 每个向量 dim 个 unsigned char
        if file_size != expected_size:
            raise ValueError(f"File size {file_size} does not match expected size {expected_size}.")
        else:
            print(f"points: {points}, dim: {dim}")
        while nodes < points:  # 读取向量的各个分量
            vector = np.frombuffer(f.read(dim), dtype=np.uint8)  # 每个分量是 unsigned char
            yield vector
            nodes += 1
        data = f.read(1)
        if not data:  # 检测 EOF
            print("End of file reached.")
        else:
            raise ValueError(f"Extra data found in file: {data}")

def read_ivecs(file_path):
    nodes = 0
    with open(file_path, 'rb') as f:
        points = struct.unpack('i', f.read(4))[0]  # 读取向量个数
        dim = struct.unpack('i', f.read(4))[0]    # 读取向量维度
        file_size = os.path.getsize(file_path)
        expected_size = 8 + points * dim * 4  # 每个向量 dim 个 int
        if file_size != expected_size:
            raise ValueError(f"File size {file_size} does not match expected size {expected_size}.")
        else:
            print(f"points: {points}, dim: {dim}")
        while nodes < points:  # 读取向量的各个分量
            vector = np.frombuffer(f.read(dim * 4), dtype=np.int32)  # 每个分量是 int
            yield vector
            nodes += 1
        data = f.read(1)
        if not data:  # 检测 EOF
            print("End of file reached.")
        else:
            raise ValueError(f"Extra data found in file: {data}")


def read_vectors(file_path, dtype, element_size):
    nodes = 0
    with open(file_path, 'rb') as f:
        points = struct.unpack('i', f.read(4))[0]  # 读取向量个数
        dim = struct.unpack('i', f.read(4))[0]    # 读取向量维度
        file_size = os.path.getsize(file_path)
        expected_size = 8 + points * dim * element_size
        if file_size != expected_size:
            raise ValueError(f"File size {file_size} does not match expected size {expected_size}.")
        else:
            print(f"points: {points}, dim: {dim}")
        while nodes < points:  # 读取向量的各个分量
            vector = np.frombuffer(f.read(dim * element_size), dtype=dtype)
            yield vector
            nodes += 1
        data = f.read(1)
        if not data:  # 检测 EOF
            print("End of file reached.")
        else:
            raise ValueError(f"Extra data found in file: {data}")


#index保存路径
index_dir = "./index/" # you can put this wherever you want, but make sure it's a place you can write to
os.makedirs(index_dir, exist_ok=True)

#需要建立索引的向量数据所在路径
base_vec_dir = "/home/xym/anns/DiskANN/build/data/gist/gist_learn.fbin"
#query向量数据所在路径
query_vec_dir = "/home/xym/anns/DiskANN/build/data/gist/gist_query.fbin"


# Commonalities
my_dtype = np.float32  # or np.uint8 or np.int8 ONLY
my_set_of_vectors: np.typing.NDArray[my_dtype] = np.array(list(read_fvecs(base_vec_dir))) # your vectors come from somewhere - you need to bring these!
# index_to_identifiers_map: np.typing.NDArray[str] = ... # your vectors likely have some kind of external identifier - 
# # you need to keep track of the external identifier -> index relationship somehow
# identifiers_to_index_map: dict[str, np.uint32|np.uint.64] = ... # your map of your external id to the `diskannpy` internal id
# # diskannpy `query` responses will contain the _internal id only_, and if you don't have these maps you won't be able to 
# # know what this relates to


# #向量归一化，长度变为1，方便计算余弦相似度
# vecs = my_set_of_vectors / np.linalg.norm(my_set_of_vectors, axis=1)  # useful if your intention is to rank by a directionless 
# # cosine angle distance
#计算
vecs = my_set_of_vectors

# print("向量:")
# print("-----------------------------------------")
# print(vecs)
# print("-----------------------------------------")

#建立索引并保存
dap.build_disk_index(
    # data=vecs,
    data=base_vec_dir,
    distance_metric="l2", # can also be cosine, especially if you don't normalize your vectors like above
    index_directory=index_dir,
    complexity=128,  # the larger this is, the more candidate points we consider when ranking
    graph_degree=64,  # the beauty of a vamana index is it's ability to shard and be able to transfer long distances across the grpah without navigating the whole thing. the larger this value is, the higher quality your results, but the longer it will take to build 
    search_memory_maximum=64.00, # a floating point number to represent how much memory in GB we want to optimize for @ query time
    build_memory_maximum=128.00, # a floating point number to represent how much memory in GB we are allocating for the index building process
    num_threads=0,  # 0 means use all available threads - but if you are in a shared environment you may need to restrict how greedy you are
    vector_dtype=my_dtype,  # we specified this in the Commonalities section above
    index_prefix="gist_disk_index_ann",  # ann is the default anyway. all files generated will have the prefix `ann_`, in the form of `f"{index_prefix}_"`
    pq_disk_bytes=0  # using product quantization of your vectors can still achieve excellent recall characteristics at a fraction of the latency, but we'll do it without PQ for now
)

#读取索引并载入
index = dap.StaticDiskIndex(
    index_directory=index_dir,
    num_threads=0,
    num_nodes_to_cache=1_000_000,
    index_prefix="gist_disk_index_ann",
    distance_metric = "l2",
    vector_dtype = my_dtype
)

#读取查询向量
query_vectors : np.typing.NDArray[my_dtype] = np.array(list(read_fvecs(query_vec_dir)))


#单个查询
some_index: np.uint32 = 0 # the index in our `q` array of points that we will be using to query on an individual basis
my_query_vector: np.typing.NDArray[my_dtype] = query_vectors[some_index] # make sure this is a 1-d array of the same dimensionality as your index!
# normalize if required by my_query_vector /= np.linalg.norm(my_query_vector)
internal_indices, distances = index.search(
    query=my_query_vector,
    k_neighbors=25,
    complexity=50,  # must be as big or bigger than `k_neighbors`
) 

print("Internal indices:", internal_indices)
print("Distances:", distances)

# actual_identifiers = index_to_identifiers_map[internal_indices]  # using np fancy indexing (advanced indexing?) to map them all to ids you actually understand



# 多个查询进行batch查询
import multiprocessing

internal_indices, distances = index.batch_search(
    queries=query_vectors,
    k_neighbors=1,
    complexity=50,
    num_threads=multiprocessing.cpu_count(), # there's a current bug where this is not handling the value 0 properly
    beam_width=8 # beamwidth is the parameter that indicates our parallelism of individual searches, whereas num_threads 
    # indicates the number of threads *per* query item in the batch
)
# note that in batch_query form, our internal_indices and distances are 2d arrays

print("查询结果:")
print("-----------------------------------------")
print(internal_indices)
print("-----------------------------------------")
print(distances)
# actual_neighbors = np.full(shape=internal_indices.shape, dtype=str, fill_value="")
# for row in range(internal_indices.shape[0]):
#     actual_neighbors[row] = index_to_identifiers_map[internal_indices[row]]