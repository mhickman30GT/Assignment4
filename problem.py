import multiprocessing
import os
import contextlib
import io
import time
import copy
import gym

from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import _printVerbosity, _MSG_STOP_MAX_ITER, _MSG_STOP_UNCHANGING_POLICY
from hiive.mdptoolbox import util

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheff
import numpy as np
import frozenlakemod as frozen_lake

# GLOBAL VARIABLES
RANDOM_SEED = 14
CORE_COUNT_PERCENTAGE = .75  # NOTE: Any increase past this and the comp is unusable
ONE_HOT_ENCODING = True


class Pool:
    """Class to run a pool of algorithms"""

    def __init__(self, algorithms, num_cores=6):
        """Constructor for Pool"""
        self.algorithms = algorithms
        self.num_cores = min(num_cores, len(algorithms))

    @staticmethod
    def _run_algorithm(algorithm):
        """Run algorithm (capture and store stdout)"""
        print(f"    Start : {algorithm.name}")
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io):
            algorithm.run()
        algorithm.stdout = string_io.getvalue()
        print(f"    Stop  : {algorithm.name}")
        return algorithm

    def run(self):
        """Run algorithms in multiprocessing pool"""
        print(f"Running {len(self.algorithms)} algorithms with {self.num_cores} cores")
        pool = multiprocessing.Pool(processes=self.num_cores)
        self.algorithms = pool.map(self._run_algorithm, self.algorithms)
        pool.close()
        pool.join()


class Environment:
    """ Class holding problem environment """

    def __init__(self, name, problem, param):
        """ Constructor for Dataset """
        self.name = name
        self.problem = problem
        self.title = f'{problem} Problem Environment'
        self.val = param
        self.env = None
        self.reward = None
        self.transition = None
        self.pi = None
        self.vi = None
        self.ql = None

    def process_matrices(self):
        """ Processes matrices between two APIs """
        # Run algorithm stolen from online to translate
        # Grab GYM matrices
        nA, nS = self.env.nA, self.env.nS

        # Translate them to MDPToolBox
        transition = np.zeros([nA, nS, nS])
        reward = np.zeros([nS, nA])
        # Note: This can take a huge amount of time
        for x in range(nS):
            for y in range(nA):
                transitions = self.env.P[x][y]
                for p_trans, next_s, rew, done in transitions:
                    transition[y, x, next_s] += p_trans
                    reward[x, y] = rew
                transition[y, x, :] /= np.sum(transition[y, x, :])
        # Save off new matrices
        self.transition = transition
        self.reward = reward

    def run(self):
        """ Run the environment through the algorithms """
        # Run each algorithm if it exists
        if self.pi:
            self.pi.run()
        if self.vi:
            self.vi.run()
        if self.ql:
            self.ql.run()


class VI:
    """ Class to run VI """

    def __init__(self, name, transition, reward, config, outdir):
        """ Constructor for VI """
        self.name = name
        self.title = "VI"
        self.transition = transition
        self.reward = reward
        self.outdir = outdir
        self.config = config
        self.results = None
        self.dataframe = None
        self.policy = None
        self.instance = ValueIteration(transition, reward, gamma=config['gamma'],
                                       epsilon=config['epsilon'], max_iter=config['max_iter'])

    def run(self):
        """ Run VI """
        self.results = self.instance.run()
        self.dataframe = pd.DataFrame(self.results)
        self.policy = self.instance.policy


class PolicyIterationMod(PolicyIteration):
    """ Mashing PImodified with PI to get Epsilon with run stats """
    def __init__(self, transition, reward, gamma, epsilon,
                 policy0=None, max_iter=1000, eval_type=0, skip_check=False, run_stat_frequency=None):
        """ Init original PI with modified PI espsilon handling """
        # Call PI constructor
        PolicyIteration.__init__(
            self,
            transition,
            reward,
            gamma,
            policy0=policy0,
            max_iter=max_iter,
            eval_type=eval_type,
            skip_check=skip_check,
            run_stat_frequency=run_stat_frequency,
        )
        # Copied Epsilon code from mdp.py
        # Set threshold based on epsilon
        self.epsilon = epsilon
        self.gamma = gamma
        if self.gamma != 1:
            self.thresh = self.epsilon * (1 - self.gamma) / self.gamma
        else:
            self.thresh = self.epsilon

    def run(self):
        """ Overriden run code to include exploration phase """
        # Run the policy iteration algorithm.
        self._startRun()
        self.run_stats = []
        self.error_mean = []
        error_cumulative = []
        self.v_mean = []
        v_cumulative = []
        self.p_cumulative = []
        run_stats = []
        while True:
            self.iter += 1
            take_run_stat = (
                self.iter % self.run_stat_frequency == 0 or self.iter == self.max_iter
            )
            # these _evalPolicy* functions will update the classes value
            # attribute
            policy_V, policy_R, itr = (
                self._evalPolicyMatrix()
                if self.eval_type == "matrix"
                else self._evalPolicyIterative()
            )
            if take_run_stat:
                v_cumulative.append(policy_V)
                if len(v_cumulative) == 100:
                    self.v_mean.append(np.mean(v_cumulative, axis=1))
                    v_cumulative = []
                if len(self.p_cumulative) == 0 or not np.array_equal(
                    self.policy, self.p_cumulative[-1][1]
                ):
                    self.p_cumulative.append((self.iter, self.policy.copy()))
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, next_v = self._bellmanOperator()
            error = np.absolute(next_v - policy_V).max()
            run_stats.append(
                self._build_run_stat(
                    i=self.iter,
                    s=None,
                    a=None,
                    r=np.max(policy_V),
                    p=policy_next,
                    v=policy_V,
                    error=error,
                )
            )
            if take_run_stat:
                error_cumulative.append(error)
                if len(error_cumulative) == 100:
                    self.error_mean.append(np.mean(error_cumulative))
                    error_cumulative = []
                self.run_stats.append(run_stats[-1])
                run_stats = []
            # Add variation calculation
            # Code from another area of mdp
            variation = util.getSpan(next_v - self.V)
            del next_v
            # calculate in how many places does the old policy disagree with
            # the new policy
            nd = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            if self.verbose:
                _printVerbosity(self.iter, nd)
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop
            # Error, rewards, and time for every iteration and number of PI steps which might be specific to my setup
            if nd == 0:
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break
            # New code: Add in an exploration phase
            elif variation < self.thresh:
                break
            else:
                self.policy = policy_next
        self._endRun()
        # add stragglers
        if len(v_cumulative) > 0:
            self.v_mean.append(np.mean(v_cumulative, axis=1))
        if len(error_cumulative) > 0:
            self.error_mean.append(np.mean(error_cumulative))
        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats
        return self.run_stats


class PI:
    """ Class to run PI """

    def __init__(self, name, transition, reward, config, outdir):
        """ Constructor for PI """
        self.name = name
        self.title = "PI"
        self.transition = transition
        self.reward = reward
        self.outdir = outdir
        self.config = config
        self.results = None
        self.policy = None
        self.dataframe = None
        self.instance = PolicyIterationMod(transition=transition, reward=reward, gamma=config['gamma'],
                                           epsilon=config['epsilon'], eval_type=config['eval_type'])

    def run(self):
        """ Run PI """
        self.results = self.instance.run()
        self.dataframe = pd.DataFrame(self.results)
        self.policy = self.instance.policy


class QL:
    """ Class to run QL """

    def __init__(self, name, transition, reward, config, outdir):
        """ Constructor for QL """
        self.name = name
        self.title = "QL"
        self.transition = transition
        self.reward = reward
        self.outdir = outdir
        self.config = config
        self.policy = None
        self.results = None
        self.dataframe = None
        self.instance = QLearning(transition, reward, gamma=config['gamma'],
                                  n_iter=config['n_iter'], alpha=config['alpha'],
                                  alpha_decay=config['alpha_decay'], alpha_min=config['alpha_min'],
                                  epsilon=config['epsilon'], epsilon_decay=config['epsilon_decay'],
                                  epsilon_min=config['epsilon_min'])

    def run(self):
        """ Run QL """
        self.results = self.instance.run()
        self.dataframe = pd.DataFrame(self.results)
        self.policy = self.instance.policy


class ExperimentClass:
    """ Class for running experiments """

    def __init__(self, name, exp_type, param, environment,
                 algorithm, size, run_config, cores, outdir):
        # Base Problem variables
        self.name = name
        self.type = exp_type
        self.param = param
        self.config = run_config
        self.outdir = outdir
        self.random_seed = RANDOM_SEED
        self.core_count = cores
        self.environment = environment
        self.algorithm = algorithm
        self.size = size
        self.states = None
        self.environment_class = None
        self.algorithm_class = None
        self.vi = None
        self.pi = None
        self.ql = None
        self.run_list = dict()
        self.pool_list = list()

        # Frozen lake settings
        self.lake_color_map = {
            'S': 'khaki',
            'G': 'lime',
            'F': 'lightskyblue',
            'H': 'orangered',
        }
        self.lake_direction_map = {
            3: r'$\uparrow$',
            2: r'$\rightarrow$',
            1: r'$\downarrow$',
            0: r'$\leftarrow$'
        }

    def process_environment(self, config, tune_param=0):
        """ Process instances for problem environment """
        print(f"Processing inputs for {self.type} in {self.environment}")

        # Generate environment class
        env_class = Environment(self.name, self.environment, tune_param)

        # Populate with instance
        if self.environment == "forest":
            # Generate fire management problem
            env_class.env = forest(S=config['states'], r1=config['reward1'],
                                   r2=config['reward2'], p=config['prob'],
                                   is_sparse=config['is_spare'])
            env_class.transition = env_class.env[0]
            env_class.reward = env_class.env[1]
            self.states = config['states']

        else:
            # Create FrozenLake and process matrices
            lake = frozen_lake.generate_random_map(size=config['size'], p=config['prob'])
            env_class.env = frozen_lake.FrozenLakeEnv(lake)
            env_class.process_matrices()
            self.states = config['size']**2

        return env_class

    def generate_vi(self, env, config):
        """ Process class instance for vi algorithm """
        # Generate VI class
        vi_class = VI(self.name, env.transition, env.reward, config, self.outdir)
        return vi_class

    def generate_pi(self, env, config):
        """ Process class instance for pi algorithm """
        # Generate PI class
        pi_class = PI(self.name, env.transition, env.reward, config, self.outdir)
        return pi_class

    def generate_ql(self, env, config):
        """ Process class instance for ql algorithm """
        # Generate QL class
        ql_class = QL(self.name, env.transition, env.reward, config, self.outdir)
        return ql_class

    def performance_plots(self, env):
        """ Plot performance curves for run """
        # Create VI plot
        vi_df = env.vi.dataframe
        fig, ax = plt.subplots()
        ax.set_title(f"Value Iteration: Performance over Iteration")
        ax.set_xlabel(f'Iteration')
        ax.set_ylabel("Value")
        vi_df.reset_index().plot(kind='line', ax=ax, label="Max Value",
                                 x='Iteration', y='Max V')
        vi_df.reset_index().plot(kind='line', ax=ax, label="Mean Value",
                                 x='Iteration', y='Mean V', color='green')
        ax2 = ax.twinx()
        ax2.set_ylabel("Error")
        vi_df.reset_index().plot(kind='line', ax=ax2, label="Error",
                                 x='Iteration', y='Error', color='red')

        # Hide grids since they look bad
        ax.grid(False)
        ax2.grid(False)

        # Combine legends
        ax2.get_legend().remove()
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best')

        fig.savefig(os.path.join(self.outdir, f"{self.name}_vi_curve.png"))

        # Create PI plot
        pi_df = env.pi.dataframe
        fig, ax = plt.subplots()
        ax.set_title(f"Policy Iteration: Performance over Iteration")
        ax.set_xlabel(f'Iteration')
        ax.set_ylabel("Value")
        pi_df.reset_index().plot(kind='line', ax=ax, label="Max Value",
                                 x='Iteration', y='Max V')
        pi_df.reset_index().plot(kind='line', ax=ax, label="Mean Value",
                                 x='Iteration', y='Mean V', color='green')
        ax2 = ax.twinx()
        ax2.set_ylabel("Error")
        pi_df.reset_index().plot(kind='line', ax=ax2, label="Error",
                                 x='Iteration', y='Error', color='red')
        # Hide grids since they look bad
        ax.grid(False)
        ax2.grid(False)

        # Combine legends
        ax2.get_legend().remove()
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best')

        fig.savefig(os.path.join(self.outdir, f"{self.name}_pi_curve.png"))

        # Create QL plot
        ql_df = env.ql.dataframe
        fig, ax = plt.subplots()
        ax.set_title(f"Q-Learning: Performance over Iteration")
        ax.set_xlabel(f'Iteration')
        ax.set_ylabel("Value")
        ql_df.reset_index().plot(kind='line', ax=ax, label="Max Value",
                                 x='Iteration', y='Max V')
        ql_df.reset_index().plot(kind='line', ax=ax, label="Mean Value",
                                 x='Iteration', y='Mean V', color='green')
        ax2 = ax.twinx()
        ax2.set_ylabel("Error")
        ql_df.reset_index().plot(kind='line', ax=ax2, label="Error",
                                 x='Iteration', y='Error', color='red')
        # Hide grids since they look bad
        ax.grid(False)
        ax2.grid(False)

        # Combine legends
        ax2.get_legend().remove()
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best')

        fig.savefig(os.path.join(self.outdir, f"{self.name}_ql_curve.png"))

    def tuning_plots(self, title, tune_df):
        """ Plot tuning curves passed in """
        # Create Time plot
        fig, ax = plt.subplots()
        ax.set_title(f"{title}: Reward vs {self.param}")
        ax.set_xlabel(f'{self.param}')
        ax.set_ylabel("Value")
        tune_df.reset_index().plot(kind='line', ax=ax, label="Max Value",
                                 x=f'{self.param}', y='Max Value', marker="o")
        tune_df.reset_index().plot(kind='line', ax=ax, label="Mean Value",
                                 x=f'{self.param}', y='Mean Value', color='green', marker="o")
        ax2 = ax.twinx()
        ax2.set_ylabel("Time")
        tune_df.reset_index().plot(kind='line', ax=ax2, label="Time",
                                 x=f'{self.param}', y='Time', color='red', marker="o")
        # Hide grids since they look bad
        ax.grid(False)
        ax2.grid(False)

        # Combine legends
        ax2.get_legend().remove()
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best')

        fig.savefig(os.path.join(self.outdir, f"{self.name}_time_curve.png"))

        # Create Iteration plot
        fig, ax = plt.subplots()
        ax.set_title(f"{title}: Reward vs {self.param}")
        ax.set_xlabel(f'{self.param}')
        ax.set_ylabel("Value")
        tune_df.reset_index().plot(kind='line', ax=ax, label="Max Value",
                                 x=f'{self.param}', y='Max Value', marker="o")
        tune_df.reset_index().plot(kind='line', ax=ax, label="Mean Value",
                                 x=f'{self.param}', y='Mean Value', color='green', marker="o")
        ax2 = ax.twinx()
        ax2.set_ylabel("Iteration")
        tune_df.reset_index().plot(kind='line', ax=ax2, label="Iteration",
                                 x=f'{self.param}', y='Iteration', color='red', marker="o")

        # Hide grids since they look bad
        ax.grid(False)
        ax2.grid(False)

        # Combine legends
        ax2.get_legend().remove()
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best')

        fig.savefig(os.path.join(self.outdir, f"{self.name}_iteration_curve.png"))

    def forest_policy(self, algorithm):
        """ Create forest state map """
        fig, ax = plt.subplots()
        policy = algorithm.policy
        length = int(np.sqrt(self.states))
        ax.pcolormesh(~np.array(policy).reshape(length, length), cmap="RdYlGn", edgecolors="k", linewidth=0.2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.set_title(algorithm.title)

        # Save fig
        fig.savefig(os.path.join(self.outdir, f"{self.name}_{algorithm.title}_forest_policy.png"))

    def lake_policy_plot(self, env, algorithm):
        """ Plot out the policy map """
        # Grab policy
        policy = algorithm.policy
        length = np.sqrt(len(policy)).astype(int)
        lake_map = np.array(list(env.env.desc)).reshape(length, length)
        lake_map = [[c.decode('utf-8') for c in line] for line in lake_map]

        # Create figure
        fig, ax = plt.subplots()
        ax.set(title=f'{algorithm.title}', xlim=(0, length), ylim=(0, length))
        ax.axis('off')
        ax.grid(False)

        for i in range(length):
            for j in range(length):
                x = j
                y = length - i - 1
                p = plt.Rectangle([x, y], 1, 1, alpha=1)
                p.set_facecolor(self.lake_color_map[lake_map[i][j]])
                p.set_edgecolor('black')
                ax.add_patch(p)
                if lake_map[i][j] == 'H' or lake_map[i][j] == 'G':
                    continue
                text = ax.text(x + 0.5, y + 0.5, self.lake_direction_map[policy[length * i + j]], size=5,
                               horizontalalignment='center', verticalalignment='center', color='black')
                text.set_path_effects([patheff.Stroke(linewidth=1, foreground='black'),
                                       patheff.Normal()])
        # Save fig
        fig.savefig(os.path.join(self.outdir, f"{self.name}_{algorithm.title}_lake_policy.png"))

    def run(self):
        """ Run the experiment """
        print(f"Running {self.type} for {self.environment}")
        results = None

        # Check the part of the project to run
        if self.type == "plots":
            l_config = self.config
            l_env = self.process_environment(l_config)

            # Setup algorithms
            l_env.vi = self.generate_vi(l_env, l_config['vi'])
            l_env.pi = self.generate_pi(l_env, l_config['pi'])
            l_env.ql = self.generate_ql(l_env, l_config['ql'])

            # Run the problem
            l_env.run()

            # Create plots
            self.performance_plots(l_env)

            # Create policy plots
            if self.environment == "lake":
                # Plot all 3 algorithms
                self.lake_policy_plot(l_env, l_env.vi)
                self.lake_policy_plot(l_env, l_env.pi)
                self.lake_policy_plot(l_env, l_env.ql)
            else:
                # Plot all 3 algorithms
                self.forest_policy(l_env.vi)
                self.forest_policy(l_env.pi)
                self.forest_policy(l_env.ql)

        # Part one can be tuning or plotting
        elif self.type == "tuning":
            # Check if tuning problem size or algorithm
            if self.size:
                # Create all the problem environments
                tune_list = self.config[self.param]
                l_config = self.config

                # Loop through tuning parameter
                for val in tune_list:
                    # Setup environment
                    l_config[self.param] = val
                    l_env = self.process_environment(l_config, tune_param=val)

                    # Setup algorithms
                    l_env.vi = self.generate_vi(l_env, l_config['vi'])
                    l_env.pi = self.generate_pi(l_env, l_config['pi'])
                    l_env.ql = self.generate_ql(l_env, l_config['ql'])

                    # Save off environment class
                    self.run_list[val] = l_env
                    self.pool_list.append(l_env)

                # Init pool
                pool = Pool(self.pool_list, self.core_count)

                # Run the pool
                pool.run()

                # Gather results from pool
                results = pool.algorithms

                # Dataframes for the tuning curves
                vi_tune_curve = pd.DataFrame()
                pi_tune_curve = pd.DataFrame()
                ql_tune_curve = pd.DataFrame()

                # Generate tuning curves
                for result in results:
                    # For each algorithm, setup data
                    # Setup VI tuning curve
                    res_vi = result.vi
                    vi_last = res_vi.dataframe.iloc[-1]

                    # Create data row
                    vi_dic = dict()
                    vi_dic[f'{self.param}'] = result.val
                    vi_dic["Reward"] = vi_last["Reward"]
                    vi_dic["Error"] = vi_last["Error"]
                    vi_dic["Time"] = vi_last["Time"]
                    vi_dic["Max Value"] = vi_last["Max V"]
                    vi_dic["Mean Value"] = vi_last["Mean V"]
                    vi_dic["Iteration"] = vi_last["Iteration"]

                    # Add to dataframe
                    if not vi_tune_curve.empty:
                        vi_tune_curve = vi_tune_curve.append(vi_dic, ignore_index=True)
                    else:
                        vi_tune_curve = pd.DataFrame(vi_dic, index=[0])

                    # Setup PI tuning curve
                    res_pi = result.pi
                    pi_last = res_pi.dataframe.iloc[-1]

                    # Create data row
                    pi_dic = dict()
                    pi_dic[f'{self.param}'] = result.val
                    pi_dic["Reward"] = pi_last["Reward"]
                    pi_dic["Error"] = pi_last["Error"]
                    pi_dic["Time"] = pi_last["Time"]
                    pi_dic["Max Value"] = pi_last["Max V"]
                    pi_dic["Mean Value"] = pi_last["Mean V"]
                    pi_dic["Iteration"] = pi_last["Iteration"]

                    # Add to dataframe
                    if not pi_tune_curve.empty:
                        pi_tune_curve = pi_tune_curve.append(pi_dic, ignore_index=True)
                    else:
                        pi_tune_curve = pd.DataFrame(pi_dic, index=[0])

                    # Setup QL tuning curve
                    res_ql = result.ql
                    ql_last = res_ql.dataframe.iloc[-1]

                    # Create data row
                    ql_dic = dict()
                    ql_dic[f'{self.param}'] = result.val
                    ql_dic["Reward"] = ql_last["Reward"]
                    ql_dic["Error"] = ql_last["Error"]
                    ql_dic["Time"] = ql_last["Time"]
                    ql_dic["Max Value"] = ql_last["Max V"]
                    ql_dic["Mean Value"] = ql_last["Mean V"]
                    ql_dic["Iteration"] = ql_last["Iteration"]

                    # Add to dataframe
                    if not ql_tune_curve.empty:
                        ql_tune_curve = ql_tune_curve.append(ql_dic, ignore_index=True)
                    else:
                        ql_tune_curve = pd.DataFrame(ql_dic, index=[0])

            # Otherwise tune the algorithm
            else:
                # Find tuning param and create pool
                alg_config = self.config[self.algorithm]
                tune_list = alg_config[self.param]
                l_config = alg_config

                # Loop through and create
                for val in tune_list:
                    # Create problem environment
                    l_env = self.process_environment(self.config)

                    # Setup config
                    l_config[self.param] = val
                    l_env.val = val

                    # Setup algorithms
                    if self.algorithm == "vi":
                        l_env.vi = self.generate_vi(l_env, l_config)
                    elif self.algorithm == "pi":
                        l_env.pi = self.generate_pi(l_env, l_config)
                    elif self.algorithm == "ql":
                        l_env.ql = self.generate_ql(l_env, l_config)

                    # Save off environment class
                    self.run_list[val] = l_env
                    self.pool_list.append(l_env)

                # Init pool
                pool = Pool(self.pool_list, self.core_count)

                # Run the pool
                pool.run()

                # Gather results from pool
                results = pool.algorithms

                # Dataframes for the tuning curves
                vi_tune_curve = pd.DataFrame()
                pi_tune_curve = pd.DataFrame()
                ql_tune_curve = pd.DataFrame()

                # Generate tuning curves
                for result in results:
                    # Find the algorithm tuned
                    if self.algorithm == "vi":
                        # Setup VI tuning curve
                        res_vi = result.vi
                        vi_last = res_vi.dataframe.iloc[-1]

                        # Create data row
                        vi_dic = dict()
                        vi_dic[f'{self.param}'] = result.val
                        vi_dic["Reward"] = vi_last["Reward"]
                        vi_dic["Error"] = vi_last["Error"]
                        vi_dic["Time"] = vi_last["Time"]
                        vi_dic["Max Value"] = vi_last["Max V"]
                        vi_dic["Mean Value"] = vi_last["Mean V"]
                        vi_dic["Iteration"] = vi_last["Iteration"]

                        # Add to dataframe
                        if not vi_tune_curve.empty:
                            vi_tune_curve = vi_tune_curve.append(vi_dic, ignore_index=True)
                        else:
                            vi_tune_curve = pd.DataFrame(vi_dic, index=[0])

                    elif self.algorithm == "pi":
                        # Setup PI tuning curve
                        res_pi = result.pi
                        pi_last = res_pi.dataframe.iloc[-1]

                        # Create data row
                        pi_dic = dict()
                        pi_dic[f'{self.param}'] = result.val
                        pi_dic["Reward"] = pi_last["Reward"]
                        pi_dic["Error"] = pi_last["Error"]
                        pi_dic["Time"] = pi_last["Time"]
                        pi_dic["Max Value"] = pi_last["Max V"]
                        pi_dic["Mean Value"] = pi_last["Mean V"]
                        pi_dic["Iteration"] = pi_last["Iteration"]

                        # Add to dataframe
                        if not pi_tune_curve.empty:
                            pi_tune_curve = pi_tune_curve.append(pi_dic, ignore_index=True)
                        else:
                            pi_tune_curve = pd.DataFrame(pi_dic, index=[0])

                    elif self.algorithm == "ql":
                        # Setup QL tuning curve
                        res_ql = result.ql
                        ql_last = res_ql.dataframe.iloc[-1]

                        # Create data row
                        ql_dic = dict()
                        ql_dic[f'{self.param}'] = result.val
                        ql_dic["Reward"] = ql_last["Reward"]
                        ql_dic["Error"] = ql_last["Error"]
                        ql_dic["Time"] = ql_last["Time"]
                        ql_dic["Max Value"] = ql_last["Max V"]
                        ql_dic["Mean Value"] = ql_last["Mean V"]
                        ql_dic["Iteration"] = ql_last["Iteration"]

                        # Add to dataframe
                        if not ql_tune_curve.empty:
                            ql_tune_curve = ql_tune_curve.append(ql_dic, ignore_index=True)
                        else:
                            ql_tune_curve = pd.DataFrame(ql_dic, index=[0])

                # Create tuning plots
                if self.algorithm == "vi":
                    self.tuning_plots("Value Iteration", vi_tune_curve)
                elif self.algorithm == "pi":
                    self.tuning_plots("Policy Iteration", pi_tune_curve)
                elif self.algorithm == "ql":
                    self.tuning_plots("Q-Learning", ql_tune_curve)

        return results
