---
layout: post
title: "TFS Improvements - Part1"
date: 2017-09-22
---

# Tensorflow Serving in Enterprise Applications: Our Experience and Workarounds - Part 1

*Sujoy Roy, Srinivasa Byaiah Ramachandra Reddy, Per Goncalves Da Silva*

**SAP Leonardo ML Foundation**

One of the major challenges in realising an intelligent enterprise is production-ready deployment of machine learning models. The need for a scalable, dynamic and fault-tolerant serving framework is paramount. In this article, we present our hands-on experiences working with Tensorflow Serving (TFS) - a flexible, high-performance serving system for machine learning models, designed for production environments. We specifically highlight several work arounds we successfully implemented to make it more enterprise ready.

One of the reasons we have been looking at TFS as our choice serving framework is because it makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. Lifecycle management, version control, flexibility to address all kinds of ML applications are just a few in a list of crucial advantages of the serving framework. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be “easily” (as claimed in the documentation) extended to serve other types of models and data (our experiments with custom deployment is work in progress).

Now what do we mean by “making tensorflow serving enterprise ready”? Tensorflow serving in its original avatar (r0.5) allowed us to serve a single model as part of a deployable service. To ensure fault-tolerance and effective management of deployments, each service could be containerized and deployed in a Kubernetes (K8S) cluster. How to do this is explained in the tensorflow serving documentation (https://www.tensorflow.org/serving/serving_inception) with the example of serving an inception model for image classification. But this approach fails to effectively utilize expensive resources like GPUs and adds to resource costs due to Kubernetes constraint of one container per node. Also using this approach makes it hard to manage multiple clients and their multiple applications. In this post we present our solution which has been committed to the TFS code base from version r0.5 onwards.

Before we go into the details of what changes we made to TFS it is important to understand what is happening behind the hood in TFS. So first we present the sequence of steps that take place behind the scenes in bringing up a service with reference to the TFS code base. The TFS model server is started with the following call:

```
    bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
    --enable_batching --port=9000 --model_name=inception   
    --model_base_path=/serving/inception
```

The entry point for the tensorflow_model_server is the file main.cc (model_servers/main.cc) which accepts a set of flags (minimally the model_name, model_base_path and the port) which define the default behaviour of the server. The main.cc uses this information to create a server_model_config proto. Then it creates a ServerCore (ref Fig. 1) using this proto to start a service instance.

Figure 1 reproduces the TFS architecture/block level control flow diagram [TFS-arch](https://www.tensorflow.org/serving/architecture_overview) with overlapping references to the functions from the code base that handles each of these blocks.

<!--
![Figure 1](http://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure1.png)
-->

<p align="center">
  <img src="https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure1.png">
</p>

Figure 1. TFS architecture mapped with essential functions from code base.

|  Component |  Type | Code Reference  |
| ------------ | ------------ | ------------ |
| Server Core  | ServerCore	  | model_servers/server_core*  |
| Manager  | DynamicManager  |  core/aspired_versions_manager* |
| Source  |  Source | sources/storage_path/file_system_storage_path_source*  |
| Source Adapter  | SourceAdapter  | servables/tensorflow/saved_model_bundle_source_adapter*, core/source_adapter* -> StoragePathSourceAdapter |
|  Version Policy | VersionPolicy  | core/aspired_version_policy*, core/availability_preserving_policy*  |
| Loader  | Loader  | core/loader*|
| Server Handle  | ServerHandle  | core/servable_handle*  |

Table 1: Mapping architecture components to code base.

The service instance comprises of a **manager**, **source adapter / loader** and a **source**. The **source** polls the file system for model version servables in the model_base_path and uses the **source adapter** method to create a loader instance, which is sent as an aspired version to the **manager** to take a call. Thus the **source adapter /loader** (which can validate servables) is used to create loader objects for loading servables on the instruction of the manager. The **manager** is the ultimate authority in deciding whether a servable can be loaded based on understanding resource constraints. Figure 2. explains the above components and their interactions.

![Figure 2](https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure2.jpg)

<!--
<p align="center">
  <img src="https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure2.jpg">
</p>
-->

Figure 2: Component Interactions within *ServerCore*. ServerCore(2) first creates AspiredVersion Manager(3), then a Source Adapter is created(4) and connected to the manager(5). Further we create a Source(6) with Source Config pointing to the file system(7). Source is connected to the Source Adapter(8). Manager maintains the servable handle(9). Server is now ready and listening for incoming requests from the gRPC clients(10).

Figure 3. explains the decision process to load a new version of a servable involving the manager, source adapter and source.

![Figure 3](https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure3.jpg)

<!--
<p align="center">
  <img src="https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure3.jpg">
</p>
-->
Figure 3: Process for loading a new version at runtime

## Serving A Single Model

TFS uses *BuildSingleModelConfig* method in *main.cc* to build a single model configuration (when no such configuration is provided) based on the *model_base_path*, *model_version_policy* and *model_name* provided as arguments to the model server. But this works for only single model all/latest version.

For serving multiple versions we introduced the concept of a *model_config* file in TFS which can be used to pass all the meta information need by the server about the model and its versions. An example format of a *model_config* (based on the existing *model_server_config.proto*) file is as follow:

```
    >r0.5
    model_config_list: {
        config: {
            name: "model1",
            base_path: "/serving/models/model1",
            model_platform: "tensorflow",
            version_policy: 1 //serves all versions; 0 serves latest version
        }
    }
    >r1.0
    model_config_list: {
        config: {
            name: "model1",
            base_path: "/serving/models/model1",
            model_platform: "tensorflow",
            version_policy: {
            specific: {
                {versions: 5},
                {versions: 3},
            }
			}
            // instead of specific, we can use all: { } to serve all versions
            // or latest: {num_versions:2} serves latest 2 versions (useful for A/B testing)
        }
    }

```

The version handling for a single model as presented above could use the following policy schemes:
Policy for What to serve?
- Latest Version (folder name with the highest number) (>r0.5)
- All Versions (only >r0.5)
- Specific Version (>r1.0)

Policy for How to serve a servable?
- Availability preserving by setting to AvailabilityPreservingPolicy
- Resource preserving by setting to ResourcePreservingPolicy

The option for how to serve can be set as part of the *ServerCore* initialization in *main.cc*. Set *option.aspired_version_policy* to either of the above.

The gRPC server (using method *RunServer*) serves inbound gRPC request on the port specified as the input flag. For any service request from the client (which comes with model_spec metadata) the server which is a *PredictionServiceImpl* object requests a servable handle from the manager to handle the request. The *PredictionServiceImpl* implements a few kinds of endpoints viz. classify, predict, regress etc.

One of the issues with single model deployment is that each deployment is run on a single node of the K8S cluster, thus taking up all the resources of the node. This is prohibitively costly if we consider a cluster of GPU nodes where 1 GPU is taken up for serving a single deployment of an application.

To address this issue of deploying a containerized service in a K8S cluster we investigated the option of serving multiple models within the same container. Our proposed solution is addressed below.

## Serving Multiple Models

In the last section we introduced our contribution of providing a *model_config* file that contains all the meta-information needed for serving a single model and its multiple versions. The same *model_config* file could be used for serving several models and their multiple versions simultaneously.

We added a new flag *model_config_file* which accepts the file path of the *model_config* file when the server is started. The input file contains model server config in proto format as defined in the *config/model_server_config.proto* (*serving/tensorflow_serving/config/model_server_config.proto*)

The functionality to interpret the model_config file was available in the *ServerCore*. We exposed the use of this untapped functionality through the ability to serve multiple models.  We created utility method *BuildModelConfigFromFile* to read the *model_config* file from input arguments in *main.cc*, which in turn uses *ParseProtoTextFile* to parse proto from a text file.

If the *model_config_flag* is set, the values passed to the *model_name* and *model_base_path* flags are ignored. Following is an example of a *model_config* file.

```
    model_config_list: {
        config: {
            name: "model1",
            base_path: "/serving/models/model1",
            model_platform: "tensorflow",
            verson_policy: {
                latest: { num_versions:2 }
            }
        },
        config: {
            name: "model2",
            base_path: "/serving/models/model2",
            model_platform: "tensorflow",
            verson_policy: {
            latest: { num_versions:2 }
        }
    }
```

## Experimental Analysis

To get some practical experience with TensorFlow Serving's (TFS) performance, we measure the average response time for prediction requests to one or more Inception models served by a TFS instance hosted on an AWS P2 p2.xlarge instance (containing 1 GPU). The TFS server was compiled from source code with GPU enablement.

**Set-up:** For the purposes of controlling the load sent to the server, we simulate a number of users (as threads) sending a predetermined number of requests to the server. The communication is done through gRPC (as this is protocol used by TFS).
For instance, we run an experiment starting from 1 to 10 threads (in steps of 1), with each thread submitting 100 requests against a TFS instance with two models (inception1 and inception2) and waiting 500ms between experiments:
```
    bazel-bin/tensorflow_serving/example/client --server=localhost:9000 --image_dir=./images
    --model_names=inception1,inception2 --num_requests=100 --num_threads_from=1
    --num_threads_to=10 --num_threads_step=1 --experiment_sleep_time=0.5 --out_file=results.txt

```

All requests are directed to one of the inception models being served by the TFS instance without batching. As such, each request contains a model name (e.g. inception1, inception2, etc.) and the image the model should classify. Since the communication is done via gRPC using Protobuf messages, these messages need to be constructed before being sent to the server. Depending on the size of the image being sent for classification, this can take a while. To expedite the experiment, we do an offline pre-processing for all requests (number of models x number of images) and each thread randomly picks from the list which request it will send.

We investigate the average response time using 1, 2 ,5 and 10 models with 200 users making 100 requests each.

**Observations:**

![Figure 4](https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure4.jpg)

<!--
<p align="center">
  <img src="https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure4.jpg">
</p>
-->
Figure 4. Request response wrt number of requests (threads) when serving multiple models in the same model container.

From Figure 4 we can see that the response time running 10 models on the same resource is similar to that of running a single model. This means using the same resource we can serve 10 models instead of 1. We see the obvious that the average response time grows linearly with the number of requests but the variance due to serving different number of models has practically insignificant impact on the response time.

The above solution entails creating a configuration file with the list of models we wish to serve together. However, note that, this framework is static in nature. This means, to introduce a new servable, we would have to bring down the service, re-write the configuration with the new servable added and then start the server. This has the obvious issue of unnecessarily taking down services that are in use and introducing down-time for all active services.

*The solution for addressing this issue will be addressed in our next post.*
