""" Introduce an API which is similar to keras to train RL algorithms.

    Author:     Sven Gronauer
    Date:       19.05.2020
    Updated:    14.04.2022  - discarded multi-processing code snippets
"""
import torch
import os
from typing import Optional
from phoenix_drone_simulation.utils.loggers import setup_separate_logger_kwargs
from phoenix_drone_simulation.utils import utils


class ModelS(object):

    def __init__(self,
                 alg: str,
                 env,
                 distb_type: str,
                 distb_level: float,
                 log_dir: str,
                 init_seed: int,
                 algorithm_kwargs: dict = {},
                 use_mpi: bool = False
                 ) -> None:
        """ Class Constructor  """
        self.alg = alg
        self.distb_type = distb_type
        self.distb_level = distb_level
        self.log_dir = log_dir
        self.init_seed = init_seed
        self.num_cores = 128  # set by compile()-method
        self.training = False
        self.compiled = False
        self.trained = False
        self.use_mpi = use_mpi
        self.env = env
        self.default_kwargs = utils.get_separate_defaults_kwargs(alg=alg)
        self.kwargs = self.default_kwargs.copy()
        self.kwargs['seed'] = init_seed
        self.kwargs['distb_type'] = distb_type
        self.kwargs['distb_level'] = distb_level
        self.kwargs.update(**algorithm_kwargs)
        self.logger_kwargs = None  # defined by compile (a specific seed might be passed)


    def _evaluate_model(self) -> None:
        from phoenix_drone_simulation.utils.evaluation import EnvironmentEvaluator
        evaluator = EnvironmentEvaluator(log_dir=self.logger_kwargs['log_dir'])
        evaluator.eval(env=self.trained_env, ac=self.actor_critic, num_evaluations=128 ,adv_policy=self.adversary_policy)
        # Close opened files to avoid number of open files overflow
        evaluator.close()

    def compile(self,
                num_cores=os.cpu_count(),
                **kwargs_update
                ) -> object:
        """Compile the model.

        Either use mpi for parallel computation or run N individual processes.

        Parameters
        ----------
        num_cores
        exp_name
        kwargs_update

        Returns
        -------

        """
        self.kwargs.update(kwargs_update)
        self.logger_kwargs = dict(log_dir=self.log_dir, level=1, use_tensor_board=True, verbose=True)
        self.compiled = True
        self.num_cores = num_cores
        return self
    
    def update_env(self, env):
        self.env = env
        self.compiled = True
        return

    def _eval_once(self, actor_critic, env, render) -> tuple:
        done = False
        self.trained_env.render() if render else None
        x = self.trained_env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            self.trained_env.render() if render else None
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, info = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0)
            ret += r
            episode_length += 1
        return ret, episode_length, costs

    def eval(self, **kwargs) -> None:
        self.actor_critic.eval()  # Set in evaluation mode before evaluation
        self._evaluate_model()
        self.actor_critic.train()  # switch back to train mode

    def fit(self, epochs=None, env=None) -> None:
        """ Train the model for a given number of epochs.

        Parameters
        ----------
        epochs: int
            Number of epoch to train. If None, use the standard setting from the
            defaults.py of the corresponding algorithm.
        env: gym.Env
            provide a custom environment for fitting the model, e.g. pass a
            virtual environment (based on NN approximation)

        Returns
        -------
        None

        """
        assert self.compiled, 'Call model.compile() before model.fit()'

        if epochs is None:
            epochs = self.kwargs.pop('epochs')
        else:
            self.kwargs.pop('epochs', None)  # pop to avoid double kwargs

        learn_func = utils.get_learn_function(self.alg)  # Hanyang: the learn_func is outside the class of the algorithm
        ac, env = learn_func(
            env=self.env,
            logger_kwargs=self.logger_kwargs,  # Hanyang: self.logger_kwargs is initialized in model.compile()
            epochs=epochs,
            **self.kwargs
        )
        self.actor_critic = ac
        self.trained_env = env
        self.trained = True
        # self.actor_critic.eval()
        
        return self.actor_critic, self.trained_env

    def play(self) -> None:
        """ Visualize model after training."""
        assert self.trained, 'Call model.fit() before model.play()'
        self.eval(episodes=5, render=True)

    def summary(self):
        """ print nice outputs to console."""
        raise NotImplementedError
