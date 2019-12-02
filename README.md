**Status:** Maintenance (expect bug fixes and minor updates)

<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines)

# Baselines-selfplay
This repository is primarily made to make it so that OpenAI's baselines(which can be found [here](https://github.com/openai/baselines)) can do selfplay!
# Executiion
run
```
python -m baselines.run --env=your_custom_env_id --env_type=your_env_type --custom_env_module=your_module_name
```
Below I'll talk about the specifics of your_custom_env_id, your_env_type, and your_module_name
# Installation requirements
This repository, so far, is only tested with python 3.7.1 but it might work with other versions! Anyway, once you get that execute
```
git clone https://github.com/isamu-isozaki/baseline-selfplay.git
cd baseline-selfplay
pip install -e .
```
And it should be up and running!
# Environment requirements
I made this so that, I need to make sure, but it won't affect any of the usual openAI gyms. So, you can still do things with them. 
These requirements will include the requirements to make a custom environment 
However, requirements for the selfplay environment is to
1. It must be a class
2. It has a sides attribute denoting the number of sides
3. There must be methods step, reset and render
4. The observation space and action space must be defined as attributes in the __init__ function of the environment like
```
from gym import spaces
self.observation_space = spaces.Box(low=0.0, high=1.0, shape=[10,10,3], dtype=np.float32)
self.action_space = spaces.Box(low=0.0, high=1.0, shape=[10], dtype=np.float32)
```
To see the list of spaces see [here](https://github.com/openai/gym/tree/master/gym/spaces)!
## Step function requirements
1. The step function must accept an action which must be 1 dimensional.
2. The step function returns None, None, None, None and save the action of the side if some sides still haven't updated. This is because I wanted to update the environment only when all sides decided to make their move. Once all sides had set their action the function must return observations, rewards, whether the environment is done, and optional infos
3. All of the values returned must be returned in lists or numpy arrays where the base index denotes which side the given observation, reward, done, and optional info corresponds to. For example, obs[0] denotes the observation side 0 made after making the action and the environment updated.
## Reset function requirements
1. Resets environment and returns the observation in the same form as above but put it in a list. So, if we call the returned value obs, obs[0] corresponds the obs returned from the step function
## Render function requirements
1. Returns a rendered image in a list. So, img[0] will be the image.
## Env_type requirements
The is the folder in which the environment that is going to be installed will be held. The folder structure of environment modules are like(thanks [Ashish Poddar](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)!)
```
your_module_name/
  README.md
  setup.py
  your_module_name/
    __init__.py
    your_env_type/
      __init__.py
      env.py
```
This will be your_env_type. your_env_id will come from
```
from gym.envs.registration import register

register(
    id=your_env_id,
    entry_point=your_module_name.your_env_name:the_name_of_the_class_that_is_your_environment,
)
```
the name of the class that is your environment must be env.py or whatever name you want!
Of course, all the above needs to be strings. This will go in the outer __init__ function. In the inner __init__ function you pretty much just import your environment but it must be referencing the module. By this, I mean that the inner __init__ function must import the environment via
```
import your_module_name.your_env_name.env import the_name_of_the_class_that_is_your_environment
```
Then, in your setup.py, just write something like
```
from setuptools import setup

setup(name=your_module_name,
      version='0.0.1',
      install_requires=[installation requirements]
)
```
The installation requirements should be in a list like
```
install_requires=["tensorflow-model-optimization==0.1.1",
"tqdm==4.39.0",
"wincertstore==0.2"]
```
And finally, just do
```
pip install -e . 
```
At the top level of your directory and you have both your selfplay environment and baselines. So, finally, run
```
python -m baselines.run --env=your_custom_env_id --env_type=your_env_type --custom_env_module=your_module_name
```
and it should start training!

To cite this repository in publications:


    @misc{baselines\_selfplay,
      author = {Isamu Isozaki},
      title = {OpenAI Baselines selfplay},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/isamu-isozaki/baseline-selfplay.git}},
    }


    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai and Zhokhov, Peter},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }

