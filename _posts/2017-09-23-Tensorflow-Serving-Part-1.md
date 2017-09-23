---
layout: post
title: "Sujoy Roy, TFS Improvements - Part1"
date: 2017-09-22
---

# Tensorflow Serving in Enterprise Applications: Our Experience and Workarounds - Part 1

*Sujoy Roy, Srinivasa Byaiah Ramachandra Reddy, Per Goncalves Da Silva*

**SAP Leonardo ML Foundation**

One of the major challenges in realising an intelligent enterprise is production-ready deployment of machine learning models. The need for a scalable, dynamic and fault-tolerant serving framework is paramount. In this article, we present our hands-on experiences working with Tensorflow Serving (TFS) - a flexible, high-performance serving system for machine learning models, designed for production environments. We specifically highlight several work arounds we successfully implemented to make it more enterprise ready.

One of the reasons we have been looking at TFS as our choice serving framework is because it makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. Lifecycle management, version control, flexibility to address all kinds of ML applications are just a few in a list of crucial advantages of the serving framework. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be “easily” (as claimed in the documentation) extended to serve other types of models and data (our experiments with custom deployment is work in progress).

Now what do we mean by “making tensorflow serving enterprise ready”? Tensorflow serving in its original avatar (r0.5) allowed us to serve a single model as part of a deployable service. To ensure fault-tolerance and effective management of deployments, each service could be containerized and deployed in a Kubernetes (K8S) cluster. How to do this is explained in the tensorflow serving documentation (https://www.tensorflow.org/serving/serving_inception) with the example of serving an inception model for image classification. But this approach fails to effectively utilize expensive resources like GPUs and adds to resource costs due to Kubernetes constraint of one container per node. Also using this approach makes it hard to manage multiple clients and their multiple applications. In this post we present our solution which has been committed to the TFS code base from version r0.5 onwards. 

Before we go into the details of what changes we made to TFS it is important to understand what is happening behind the hood in TFS. So first we present the sequence of steps that take place behind the scenes in bringing up a service with reference to the TFS code base. The TFS model server is started with the following call:

    bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --enable_batching --port=9000 --model_name=inception   --model_base_path=/serving/inception

The entry point for the tensorflow_model_server is the file main.cc (model_servers/main.cc) which accepts a set of flags (minimally the model_name, model_base_path and the port) which define the default behaviour of the server. The main.cc uses this information to create a server_model_config proto. Then it creates a ServerCore (ref Fig. 1) using this proto to start a service instance.

Figure 1 reproduces the TFS architecture/block level control flow diagram [TFS-arch](https://www.tensorflow.org/serving/architecture_overview) with overlapping references to the functions from the code base that handles each of these blocks. 

![Figure 1](https://github.com/tvkpz/tvkpz.github.io/tree/master/_posts/images/Figure1.jpg)



