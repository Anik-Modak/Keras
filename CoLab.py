{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Nahid-Hassan/Keras/blob/master/CoLab.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fiGGIg1nvcKF",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 415
        },
        "outputId": "b7e09e4c-1892-4b21-b13d-b6fde9b9954f"
      },
      "source": [
        "# The MNIST dataset comes preloaded in Keras, in the form of a set of four Numpy arrays.\n",
        "\n",
        "# from keras.datasets import mnist(National Institude of Standerds and Technology)\n",
        "from keras.datasets import mnist\n",
        "from keras import models\n",
        "from keras import layers \n",
        "from keras.utils import to_categorical\n",
        "\n",
        "# load mnist data using load_data() method\n",
        "# train_images and train_labels form the training set, the data that the model will\n",
        "# learn from. The model will then be tested on the test set, test_images and test_labels .\n",
        "(train_images, train_labels), (test_images, test_labels) = mnist.load_data()\n",
        "\n",
        "#--------------------------------------------------------------------------------------\n",
        "# To know about MNIST data set(How does look mnist train_images, train Lables)\n",
        "# https://github.com/Nahid-Hassan/Keras/blob/master/mnist%20dataset%20observation.py\n",
        "#---------------------------------------------------------------------------------------\n",
        "\n",
        "# To build network\n",
        "network = models.Sequential()\n",
        "network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))\n",
        "network.add(layers.Dense(10, activation='softmax'))\n",
        "\n",
        "# To compilation step\n",
        "network.compile(optimizer='rmsprop',\n",
        "                loss='categorical_crossentropy',\n",
        "                metrics=['accuracy'])\n",
        "#Prepairing the image data(training data)\n",
        "train_images = train_images.reshape((60000, 28 * 28))\n",
        "train_images = train_images.astype('float32') / 255\n",
        "\n",
        "test_images = test_images.reshape((10000, 28 * 28))\n",
        "test_images = test_images.astype('float32') / 255\n",
        "\n",
        "# We also need to categorically encode the labels\n",
        "# Prepairing the labels\n",
        "train_labels = to_categorical(train_labels)\n",
        "test_labels = to_categorical(test_labels)\n",
        "\n",
        "# Now we are ready to train the network, which in keras is done via call to the network 'fit' method\n",
        "network.fit(train_images, train_labels, epochs=5, batch_size=128)\n",
        "\n",
        "'''\n",
        "Using TensorFlow backend.\n",
        "WARNING:tensorflow:From /home/nahid/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
        "Instructions for updating:\n",
        "Colocations handled automatically by placer.\n",
        "WARNING:tensorflow:From /home/nahid/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
        "Instructions for updating:\n",
        "Use tf.cast instead.\n",
        "Epoch 1/5\n",
        "2019-04-23 02:47:28.100669: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA\n",
        "2019-04-23 02:47:28.126491: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2400000000 Hz\n",
        "2019-04-23 02:47:28.126780: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x55dd7cb7df20 executing computations on platform Host. Devices:\n",
        "2019-04-23 02:47:28.126805: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>\n",
        "OMP: Info #212: KMP_AFFINITY: decoding x2APIC ids.\n",
        "OMP: Info #210: KMP_AFFINITY: Affinity capable, using global cpuid leaf 11 info\n",
        "OMP: Info #154: KMP_AFFINITY: Initial OS proc set respected: 0-3\n",
        "OMP: Info #156: KMP_AFFINITY: 4 available OS procs\n",
        "OMP: Info #157: KMP_AFFINITY: Uniform topology\n",
        "OMP: Info #179: KMP_AFFINITY: 1 packages x 2 cores/pkg x 2 threads/core (2 total cores)\n",
        "OMP: Info #214: KMP_AFFINITY: OS proc to physical thread map:\n",
        "OMP: Info #171: KMP_AFFINITY: OS proc 0 maps to package 0 core 0 thread 0 \n",
        "OMP: Info #171: KMP_AFFINITY: OS proc 2 maps to package 0 core 0 thread 1 \n",
        "OMP: Info #171: KMP_AFFINITY: OS proc 1 maps to package 0 core 1 thread 0 \n",
        "OMP: Info #171: KMP_AFFINITY: OS proc 3 maps to package 0 core 1 thread 1 \n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18150 thread 0 bound to OS proc set 0\n",
        "2019-04-23 02:47:28.127104: I tensorflow/core/common_runtime/process_util.cc:71] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.\n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18175 thread 1 bound to OS proc set 1\n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18192 thread 2 bound to OS proc set 2\n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18193 thread 3 bound to OS proc set 3\n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18194 thread 4 bound to OS proc set 0\n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18174 thread 5 bound to OS proc set 1\n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18195 thread 6 bound to OS proc set 2\n",
        "60000/60000 [==============================] - 5s 79us/step - loss: 0.2607 - acc: 0.9237\n",
        "Epoch 2/5\n",
        "60000/60000 [==============================] - 5s 78us/step - loss: 0.1078 - acc: 0.9683\n",
        "Epoch 3/5\n",
        "60000/60000 [==============================] - 4s 72us/step - loss: 0.0716 - acc: 0.9782\n",
        "Epoch 4/5\n",
        "60000/60000 [==============================] - 4s 71us/step - loss: 0.0524 - acc: 0.9844\n",
        "Epoch 5/5\n",
        "60000/60000 [==============================] - 4s 71us/step - loss: 0.0398 - acc: 0.9879\n",
        "'''\n",
        "\n",
        "# Let's check that the model performs well on the test set, too\n",
        "test_loss, test_acc = network.evaluate(test_images,test_labels)\n",
        "print('test_acc: ', test_acc)  # test_acc:  0.9793\n",
        "'''\n",
        " 32/10000 [..............................] - ETA: 28sOMP: Info #250: KMP_AFFINITY: pid 18150 tid 18205 thread 7 bound to OS proc set 3\n",
        "OMP: Info #250: KMP_AFFINITY: pid 18150 tid 18206 thread 8 bound to OS proc set 0\n",
        "10000/10000 [==============================] - 1s 55us/step\n",
        "test_acc:  0.9793\n",
        "'''"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz\n",
            "11493376/11490434 [==============================] - 0s 0us/step\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Colocations handled automatically by placer.\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use tf.cast instead.\n",
            "Epoch 1/5\n",
            "60000/60000 [==============================] - 5s 90us/step - loss: 0.2525 - acc: 0.9263\n",
            "Epoch 2/5\n",
            "60000/60000 [==============================] - 5s 79us/step - loss: 0.1023 - acc: 0.9694\n",
            "Epoch 3/5\n",
            "60000/60000 [==============================] - 5s 81us/step - loss: 0.0677 - acc: 0.9797\n",
            "Epoch 4/5\n",
            "60000/60000 [==============================] - 5s 80us/step - loss: 0.0489 - acc: 0.9852\n",
            "Epoch 5/5\n",
            "60000/60000 [==============================] - 5s 78us/step - loss: 0.0369 - acc: 0.9891\n",
            "10000/10000 [==============================] - 0s 48us/step\n",
            "test_acc:  0.9816\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'\\n 32/10000 [..............................] - ETA: 28sOMP: Info #250: KMP_AFFINITY: pid 18150 tid 18205 thread 7 bound to OS proc set 3\\nOMP: Info #250: KMP_AFFINITY: pid 18150 tid 18206 thread 8 bound to OS proc set 0\\n10000/10000 [==============================] - 1s 55us/step\\ntest_acc:  0.9793\\n'"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    }
  ]
}