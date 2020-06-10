# Smart Queue Monitoring System - Retail Scenario

### Project as a part of Udacity Intel® Edge AI for IoT Developers Nanodegree Program

## Overview
Write Python script and job submission script to request Intel **IEI Tank-870** edge node and run inference on the different hardware types (CPU, GPU, VPU, FPGA). 

## Objectives
* Build out the smart queuing application to test its performance on the **DevCloud** using multiple hardware types.
* Submit inference jobs to **Intel's DevCloud** using the `qsub` command.
* Retrieve and review the results.
* Propose hardware device for retail, manufacturing and transportation scenarios.

### Submit CPU job to an [IEI Tank* 870-Q170 edge compute node]("https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core")  with an [Intel® Core™ i5-6500TE processor]("https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-").
### Submit a job to an Edge Compute Node with a CPU and IGPU
submit GPU job to an <a href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank* 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel® Core i5-6500TE</a>. The inference workload should run on the **Intel® HD Graphics 530** integrated GPU.
## Submit to an Edge Compute Node with an Intel® Neural Compute Stick 2
submit VPU job to an <a href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core i5-6500te CPU</a>. The inference workload should run on an <a 
    href="https://software.intel.com/en-us/neural-compute-stick">Intel Neural Compute Stick 2</a> installed in this  node.
## Submit to an Edge Compute Node with IEI Mustang-F100-A10
submit FPGA job to an <a href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core™ i5-6500te CPU</a> . The inference workload will run on the <a href="https://www.ieiworld.com/mustang-f100/en/"> IEI Mustang-F100-A10 </a> FPGA card installed in this node.
## Assess Performance
Compare the performance person-detection-retail-0013/FP16/ model across 4 devices. 
The following timings for the model are being comapred across all 4 devices (CPU, GPU, VPU, FPGA):
- Model Loading Time   retail scenario bar chart
- Average Inference Time   retail scenario bar chart
- FPS    retail scenario bar chart
To view the 'Retail CPU' output video  please click here.
