import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import tensorrt as trt
import time
import os
import matplotlib.pyplot as plt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        binding_index = engine.get_binding_index(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        size = trt.volume(engine.get_binding_shape(binding)) * dtype.itemsize
        if engine.binding_is_input(binding):
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(size)
            inputs.append({'host': host_mem, 'device': cuda_mem})
        else:
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(size)
            outputs.append({'host': host_mem, 'device': cuda_mem})

        bindings.append(int(cuda_mem))

    return inputs, outputs, bindings, stream

def do_inference(engine, inputs, outputs, bindings, stream, input_data):
    cuda.memcpy_htod_async(inputs[0]['device'], input_data, stream)
    engine.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host']

def inference(engine, input_data, batch_size):
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    latencies = []
    throughputs = []
    total_time = 0.0

    for i in range(batch_size):
        start_time = time.time()
        result = do_inference(engine, inputs, outputs, bindings, stream, input_data)
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_time += elapsed_time

        latencies.append(elapsed_time)
        throughputs.append(1.0 / elapsed_time)

        print(f"Batch {i} inference time: {elapsed_time:.2f} seconds")

    return latencies, throughputs, total_time

def plot_metrics(latencies, throughputs, total_time, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Plot latency
    plt.figure()
    plt.plot(range(len(latencies)), latencies, marker='o', linestyle='-', color='b')
    plt.title('Latency per Batch')
    plt.xlabel('Batch Index')
    plt.ylabel('Latency (seconds)')
    plt.savefig(os.path.join(output_dir, 'latency_per_batch.png'))
    plt.close()

    # Plot throughput
    plt.figure()
    plt.plot(range(len(throughputs)), throughputs, marker='o', linestyle='-', color='r')
    plt.title('Throughput per Second')
    plt.xlabel('Batch Index')
    plt.ylabel('Throughput (batches/second)')
    plt.savefig(os.path.join(output_dir, 'throughput_per_sec.png'))
    plt.close()

    # Plot total execution time
    plt.figure()
    plt.bar(['Total Execution Time'], [total_time], color='g')
    plt.title('Total Execution Time')
    plt.ylabel('Time (seconds)')
    plt.savefig(os.path.join(output_dir, 'total_execution_time.png'))
    plt.close()

    print(f"Graphs saved to {output_dir}")

if __name__ == "__main__":
    engine_path = 'model.trt'  # Path to your TensorRT engine file
    batch_size = 10  # Adjust based on your needs
    engine = load_engine(engine_path)
    input_data = np.random.random(size=(1, 3, 224, 224)).astype(np.float32)
    latencies, throughputs, total_time = inference(engine, input_data, batch_size)
    plot_metrics(latencies, throughputs, total_time, output_dir='graphs')

