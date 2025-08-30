from bayes_opt.bayesian_optimization import BayesianOptimization


class BOServerTools:

    def tool_initialize(self):
        raise NotImplementedError

    def tool_observe(self):
        raise NotImplementedError

    def tool_suggest(self):
        raise NotImplementedError
