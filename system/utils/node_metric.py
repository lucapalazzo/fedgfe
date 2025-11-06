from collections.abc import MutableMapping, MutableSequence
from enum import Enum, auto

class NodeMetric(MutableSequence, MutableMapping):
    class Phase(Enum):
        UNKNOWN = auto()
        PREPROCESS = auto()
        POSTPROCESS = auto()
        TRAIN = auto()
        VALIDATE = auto()
        TEST = auto()
        INFERENCE = auto()

    class TaskType(Enum):
        UNKNOWN = auto()
        CLASSIFICATION = auto()
        REGRESSION = auto()
        SEGMENTATION = auto()
        OBJECT_DETECTION = auto()
        CLUSTERING = auto()
        DIMENSIONALITY_REDUCTION = auto()
        ANOMALY_DETECTION = auto()
        REINFORCEMENT_LEARNING = auto()
        NATURAL_LANGUAGE_PROCESSING = auto()
        COMPUTER_VISION = auto()
        TIME_SERIES_FORECASTING = auto()
        AUDIO_PROCESSING = auto()
        GRAPH_ANALYSIS = auto()
        RECOMMENDATION_SYSTEMS = auto()
        OTHER = auto()

    def __init__(self, seq=None, mapping=None, phase=Phase.UNKNOWN, task_count=1):
        self._list = list(seq or [])
        self._dict = dict(mapping or {})
        self._phase = phase
        self._defined_metrics = None
        self.task_type = None
        self.node_id = -1
        self._task_count = task_count
        self._task_name = "task"
        self._dict['samples'] = 0
        self._dict['steps'] = 0
        self._calculated_metrics = [ 'mean', 'std', 'min', 'max', 'min_index', 'max_index' ]
    
    # ----  parte lista (MutableSequence) ----
    def __len__(self):
        return len(self._dict)

    def __getitem__(self, i):
        self.calculate_metrics()
        if i not in self._dict:
            raise KeyError(f"Index '{i}' not found in the dictionary.")
        # steps = self._dict.get('steps', 1) if self._dict.get('steps', 1) > 0 else 1
        return self._dict[i]
    def __setitem__(self, i, v):
        self._dict[i] = v
    def __delitem__(self, i):
        del self._dict[i]

    def __iter__(self):
        return iter(self._dict)
    def __contains__(self, k):
        return k in self._dict
    def __repr__(self):
        return f"NodeMetric({self._list}, {self._dict})"

    def calculate_metrics(self):
        if self._defined_metrics is None:
            raise ValueError("Metrics are not defined.")
        
        if self._task_count == 0:
            return self
        
        task_list = [k for k in self._dict.keys() if isinstance(k, int)]

        for metric in self._defined_metrics:
            if metric not in self._dict:
                raise KeyError(f"Metric '{metric}' not found in the dictionary.")
            metric_values = [self._dict[k][metric] for k in task_list if isinstance(self._dict[k], NodeMetric) and metric in self._dict[k]]
            steps = [self._dict[k]['steps'] for k in task_list if isinstance(self._dict[k], NodeMetric) and 'steps' in self._dict[k]]
            # check that steps are the same for all tasks
            # if not all(s == steps[0] for s in steps):
            #     raise ValueError("Steps are not the same for all tasks.")
            steps = steps[0] if steps and steps[0] > 0 else 1  # Default to 1 if no steps are found
            # metric_values = [v[metric] for v in self._dict.values() if isinstance(v, dict) and metric in v]

            metric_values = [0 if v is None else v for v in metric_values]
            
            if metric_values:
                self._dict[metric]['mean'] = sum(metric_values) / (len(metric_values)*steps) if len(metric_values) > 0 else 0
                self._dict[metric]['std'] = (sum((x/steps - self._dict[metric]['mean']) ** 2 for x in metric_values) / len(metric_values)) ** 0.5
                self._dict[metric]['min'] = min(metric_values)/steps
                self._dict[metric]['min_index'] = metric_values.index(min(metric_values))
                self._dict[metric]['max'] = max(metric_values)/steps
                self._dict[metric]['max_index'] = metric_values.index(max(metric_values))
                self._dict['samples'] = sum(self._dict[k]['samples'] for k in task_list if isinstance(self._dict[k], NodeMetric) and 'samples' in self._dict[k])
                self._dict['samples'] = self._dict['samples']//len(task_list) if len(task_list) > 0 else 0
                self._dict['steps'] = sum(self._dict[k]['steps'] for k in task_list if isinstance(self._dict[k], NodeMetric) and 'steps' in self._dict[k])
                self._dict['steps'] = self._dict['steps']//len(task_list) if len(task_list) > 0 else 0
        return self

    def insert(self, i, v):
        if i not in self._dict:
            self._dict[i] = v

    def append(self, value):
        max_index = max(self._dict.keys(), default=-1)
        self._dict[max_index + 1] = value

    def extend(self, iterable):
        for value in iterable:
            self.append(value)

    def define_metrics(self, metrics, task_count=1):
        if not isinstance(metrics, (dict)):
            raise TypeError("Metrics must be a dict.")
        self._defined_metrics = list(metrics.keys()) if isinstance(metrics, dict) else metrics
        self._metrics = metrics
        self._task_count = task_count
        if task_count > 0:
            for i in range(task_count):
                # self._dict[i] = {metric: None for metric in metrics}
                self._dict[i] = NodeMetric(phase=self._phase, task_count=0)
                self._dict[i].define_metrics(self._metrics, task_count=0)
            for metric in self._defined_metrics:
                self._dict[metric] = { 'mean': None, 'std': None, 'min': None, 'max': None }
        else:
           for metric in self._defined_metrics:
                self._dict[metric] = None
        return self
    
    def min(self):
        min_values = {'task_index': None}
        if not self._defined_metrics:
            raise ValueError("Metrics are not defined.")
        for task in range(self._task_count):
            if task not in self._dict:
                raise KeyError(f"Task '{task}' not found in the dictionary.")
            if self._dict[task] is None:
                continue
            for metric in self._defined_metrics:
                if metric not in self._dict[task]:
                    raise KeyError(f"Metric '{metric}' not found in task '{task}'.")
                if metric not in min_values:
                    min_values[metric] = self._dict[task][metric]
                    min_values['task_index'] = task
                elif self._dict[task][metric] is not None and (min_values[metric] is None or self._dict[task][metric] < min_values[metric]):
                    min_values[metric] = self._dict[task][metric]
                    min_values['task_index'] = task
        return min_values
    
    def max(self):
        max_values = { 'task_index': None }
        if not self._defined_metrics:
            raise ValueError("Metrics are not defined.")
        for task in range(self._task_count):
            if task not in self._dict:
                raise KeyError(f"Task '{task}' not found in the dictionary.")
            if self._dict[task] is None:
                continue
            for metric in self._defined_metrics:
                if metric not in self._dict[task]:
                    raise KeyError(f"Metric '{metric}' not found in task '{task}'.")
                if metric not in max_values:
                    max_values[metric] = self._dict[task][metric]
                    max_values['task_index'] = task
                elif self._dict[task][metric] is not None and (max_values[metric] is None or self._dict[task][metric] > max_values[metric]):
                    max_values[metric] = self._dict[task][metric]
                    max_values['task_index'] = task
        return max_values
    
    def __iadd__(self, values):
        if not isinstance(values, (dict, NodeMetric)):
            raise TypeError("Values must be a dict or NodeMetric instance.")
        if values._defined_metrics != self._defined_metrics:
            raise ValueError("Defined metrics do not match.")
        if isinstance(values, NodeMetric):
            if values._defined_metrics is None:
                raise ValueError("Defined metrics are not set in the NodeMetric instance.")
            if self._defined_metrics is None:
                self.define_metrics(values._defined_metrics, values._task_count)
            elif set(self._defined_metrics) != set(values._defined_metrics):
                raise ValueError("Defined metrics do not match.")
            for task in range(values._task_count):
                for metric in values._defined_metrics:
                    if metric not in self._defined_metrics:
                        self._dict[task][metric] = values._dict[task][metric]
                    elif metric not in self._dict[task] or ( self._dict[task][metric] is None and values._dict[task][metric] is not None):
                        self._dict[task][metric] = values._dict[task][metric]
                    elif values._dict[task][metric] is not None:
                        self._dict[task][metric] += values._dict[task][metric]
                    # else:
                    #     raise ValueError(f"Metric '{metric}' not found in task '{task}'.")
                self._dict[task]['steps'] += values._dict[task]['steps']
                self._dict[task]['samples'] += values._dict[task]['samples']
            # if 'samples' not in self._dict:
            #     self._dict['samples'] = values._dict['samples']
            # else:
            #     self._dict['samples'] += values._dict['samples']
            # # self._dict['steps'] += values._dict['steps']
            # self._dict['steps'] += values._dict['steps']
        return self
    
    def __str__(self):
        self.calculate_metrics()
        string = ""
        if self._defined_metrics is None:
            return "NodeMetric: No defined metrics."
        for metric in self._defined_metrics:
            if metric in self._dict:
                string += f"{self._task_name} {metric}: {self._dict[metric]['mean']:.2f} (std: {self._dict[metric]['std']:.2f}, min: {self._dict[metric]['min']:.2f} at index {self._dict[metric]['min_index']}, max: {self._dict[metric]['max']:2f} at index {self._dict[metric]['max_index']})\n"
            else:
                string += f"{metric}: Not defined\n"
        return string
    
    @property
    def phase(self):
        return self._phase

    @property
    def defined_metrics(self):
        return self._defined_metrics
    
    @phase.setter 
    def phase(self, phase):
        if not isinstance(phase, NodeMetric.Phase):
            raise TypeError("Phase must be an instance of NodeMetric.Phase.")
        self._phase = phase
        return self
    
    @property
    def steps(self):
        return self._dict.get('steps', 0)

    @steps.setter

    def steps(self, value):
        if not isinstance(value, int):
            raise TypeError("Steps must be an integer.")
        self._dict['steps'] = value
        return self

    @property
    def task_count(self):
        return self._task_count
    @task_count.setter
    def task_count(self, value):
        if not isinstance(value, int):
            raise TypeError("Task count must be an integer.")
        self._task_count = value
        return self
    
    @property
    def task_name(self):
        return self._task_name
    @task_name.setter
    def task_name(self, value):
        if not isinstance(value, str):
            raise TypeError("Task name must be a string.")
        self._task_name = value
        return self
    
    @property
    def defined_metrics(self):
        return self._defined_metrics
    
    @property
    def metrics(self):
        return self._defined_metrics