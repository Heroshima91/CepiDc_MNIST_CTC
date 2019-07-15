import tensorflow as tf
import pickle
import model
import os
import random
import json
import numpy as np

model_dir = 'model'
BATCH_SIZE = 100

lr = [0.1,0.01,0.001]
do = [0.1,0.2,0.3]
hs = [20,30,50]
pr = [1.9,2.0,2.1]
ks = [3,5,7]

def generate_dict(lr,do,hs,pr,ks):
    dict = [(l,d,h,p,k) for l in lr for d in do for h in hs for p in pr for k in ks]
    return dict


def get_next_model_dir():
    list_name = [int(name[name.find('_') + 1:]) for name in os.listdir('model_directory')]

    if len(list_name) is 0:
        last_model = 0
    else:
        last_model = max(list_name)

    return 'model_directory/model_' + str(last_model + 1)

def generate_output_size(h,p):
    output = [int(h*(p**n)) for n in range(0,5)]
    print(output)
    return output

def get_params(dict):
    idx = random.randint(0, len(dict))
    print(len(dict))
    params_rnd = dict.pop(idx)
    output_size = generate_output_size(params_rnd[2],params_rnd[3])
    hparams = {
        'learning_rate':params_rnd[0],
        'drop_out':params_rnd[1],
        'output_size': output_size ,
        'kernel_size': [params_rnd[4],params_rnd[4]],
    }
    return hparams



tf.logging.set_verbosity(tf.logging.INFO)


data = np.loadtxt('dataset/train_data.csv', delimiter=' ', dtype='float32')
lab = np.genfromtxt('dataset/train_label.csv', dtype='unicode', delimiter='\n')

data_test = np.loadtxt('dataset/test_data.csv', delimiter=' ', dtype='float32')
lab_test  = np.genfromtxt('dataset/test_label.csv', dtype='unicode', delimiter='\n')

input_fn = tf.estimator.inputs.numpy_input_fn(
        data,
        y=lab,
        num_epochs=None,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

test_input_fn = tf.estimator.inputs.numpy_input_fn(
        data_test,
        y=lab_test,
        num_epochs=None,
        shuffle=True,
        batch_size=BATCH_SIZE
)

dict = generate_dict(lr,do,hs,pr,ks)

while True:

    next_model_dir = get_next_model_dir()
    hparams = get_params(dict)
    print(hparams)

    with open('log.txt', 'a+') as file:
        file.write(next_model_dir+'\n')
        file.write(json.dumps(hparams))

    run_config = tf.estimator.RunConfig(
        model_dir=get_next_model_dir(),
        save_checkpoints_steps=2500,
        keep_checkpoint_max=1,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        config=run_config,
        params=hparams
    )


    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=30000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=test_input_fn,
        steps=100,
        start_delay_secs=60, #1
        throttle_secs=120 #10
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)