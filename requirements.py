import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from datasets import load_dataset
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
import threading
import subprocess
import time

