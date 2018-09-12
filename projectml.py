from easydict import EasyDict

class ProjectML:
    """A framework class for driving Machine Learning Project."""
    def __init__(self,
                 setup_fn=None,
                 cycle_update_parameter_fn=None,
                 cycle_setup_data_fn=None,
                 cycle_train_model_fn=None,
                 cycle_evaluate_fn=None,
                 cycle_update_policy_fn=None,
                 summarize_total_fn=None,
                 dataset_policy={},
                 training_policy={},
                 parameters={},
    ):
        """Instantiate project."""
        self.setup_fn = setup_fn
        self.cycle_update_parameter_fn = cycle_update_parameter_fn
        self.cycle_setup_data_fn = cycle_setup_data_fn
        self.cycle_train_model_fn = cycle_train_model_fn
        self.cycle_evaluate_fn = cycle_evaluate_fn
        self.cycle_update_policy_fn = cycle_update_policy_fn
        self.summarize_total_fn = summarize_total_fn
        self.dataset_policy = EasyDict(dataset_policy)
        self.training_policy = EasyDict(training_policy)
        self.prms = EasyDict(parameters)
        self.vars = EasyDict()
        self.results = EasyDict()
    def _call(self, fn):
        """Call function if it is valid."""
        if fn is not None:
            return fn(self)
    def setup(self, show_policy=False):
        """Setup project. Call once when you start."""
        self.vars._cycle = 0
        self._call(self.setup_fn)
        if show_policy:
            print('Dataset policy: {}'.format(self.dataset_policy))
            print('Training policy: {}'.format(self.training_policy))
            print('Parameters: {}'.format(self.prms))
            print('Variables: {}'.format(self.vars))
    def iterate_cycle(self):
        """Iterate one project cycle, returns False if finished."""
        print('\n#{}'.format(self.vars._cycle))
        self._call(self.cycle_update_parameter_fn)
        self._call(self.cycle_setup_data_fn)
        self._call(self.cycle_train_model_fn)
        self._call(self.cycle_evaluate_fn)
        cycle_in_progress = self._call(self.cycle_update_policy_fn)
        self.vars._cycle += 1
        return cycle_in_progress
    def summary(self):
        """Summarize overall performance."""
        print('\n# Summary')
        self._call(self.summarize_total_fn)
    def is_first_cycle(self):
        """Return True if it is the first cycle."""
        return self.vars._cycle == 0
    def run(self):
        """Run all through life of this project."""
        project.setup(show_policy=True)
        while project.iterate_cycle():
            print('finished #{}\n'.format(self.vars._cycle))
        project.summary()


