# Smart Queue Monitoring System - App testing on the Intel® DevCloud
### Project as a part of Udacity Intel® Edge AI for IoT Developers Nanodegree Program

## Overview
Write Python script and job submission script to request Intel **IEI Tank-870** edge node and run inference on the different hardware types (CPU, GPU, VPU, FPGA). 

## Objectives
* Build out the **smart queuing application** to test its performance on the **Intel® DevCloud** using multiple hardware types.
* Submit inference jobs to **Intel® DevCloud** using the `qsub` command.
* Retrieve and review the results.
* Propose hardware device for **retail**, **manufacturing** and **transportation** scenarios.

#### CPU job was submitted to the 
   - Edge Compute Node [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE processor](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-).
#### GPU job was submitted to the
   - Edge Compute Node with a CPU and GPU [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE processor](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). 
   The inference workload was run on the **Intel® HD Graphics 530** integrated GPU.
#### VPU job was submitted to the
   - Edge Compute Node [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE CPU](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). 
   The inference workload was run on an **[Intel® Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick)** installed in this  node.
#### FPGA job was submited to the
   - Edge Compute Node [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE CPU](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). 
   The inference workload was run on the **[IEI Mustang-F100-A10](https://www.ieiworld.com/mustang-f100/en/)** FPGA card installed in this node.
## Results
**The example**

Comparision of the performance person-detection-retail-0013/FP16/ model across 4 devices for retail scenario. 
The following timings for the model are compared across all 4 devices (CPU, GPU, VPU, FPGA):
- Model Loading Time   [retail scenario bar graph]()
- Average Inference Time   [retail scenario bar graph](https://github.com/ireneuszcierpisz/smart-queue-monitoring/blob/master/retail_model-inference-time.png)
- FPS    [retail scenario bar graph](https://github.com/ireneuszcierpisz/smart-queue-monitoring/blob/master/retail_model-FPS.png)

To view the 'Retail CPU' output_video.mp4  [please click here](https://github.com/ireneuszcierpisz/smart-queue-monitoring/blob/master/output_video(1).mp4).
