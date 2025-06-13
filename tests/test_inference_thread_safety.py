# type: ignore
import pytest
import time
import random
from een_eval.workflow.inference import InferenceEngine
from een_eval.core.dataset import Dataset, DatasetItem

# Dummy classes for testing
token_call_sequence = []
class DummyResult:
    def __init__(self, response, inference_time=0.0):
        self.error = None
        self.response = response
        self.inference_time = inference_time
        self.timestamp = time.time()
        self.metadata = {}
        self.tokens_per_second = None
        self.total_tokens = None
        self.prompt_tokens = None
        self.completion_tokens = None

class DummyModel:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    def generate(self, prompt, **kwargs):
        # introduce random small delay to simulate variable latency
        time.sleep(random.uniform(0, 0.01))
        # return response that identifies the prompt
        return DummyResult(response=f"[{self.name}]:{prompt.upper()}", inference_time=0.01)

class DummyOutputManager:
    def __init__(self):
        self.saved_batches = []
        self.saved_status = []
    def save_inference_metadata(self, metadata):
        pass
    def save_responses_batch(self, responses):
        # store sample ids for verification
        self.saved_batches.append([r.sample_id for r in responses])
    def save_status(self, status):
        self.saved_status.append(status.processed_samples)

class Status:
    def __init__(self):
        self.processed_samples = 0
        self.errors = []
        self.current_model = None
    def to_dict(self):
        return {
            'processed_samples': self.processed_samples,
            'errors': self.errors,
            'current_model': self.current_model
        }

@pytest.fixture
def small_dataset():
    items = [DatasetItem(id=str(i), data={'prompt': f'item{i}'}) for i in range(5)]
    return Dataset(items)

@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_parallel_and_sequential_results_equal(small_dataset, max_workers):
    models = [DummyModel('modelA')]
    params = {'num_samples': 3}
    output = DummyOutputManager()
    # sequential
    engine_seq = InferenceEngine(models, small_dataset, params, output_manager=output, batch_size=2, max_workers=1)
    status_seq = Status()
    results_seq = engine_seq._run_sequential_inference(models[0], engine_seq._create_batches(), status_seq)
    # parallel
    engine_par = InferenceEngine(models, small_dataset, params, output_manager=output, batch_size=2, max_workers=max_workers)
    status_par = Status()
    results_par = engine_par._run_parallel_inference(models[0], engine_par._create_batches(), status_par)

    # Compare lengths and sample_ids
    ids_seq = sorted([r.sample_id for r in results_seq])
    ids_par = sorted([r.sample_id for r in results_par])
    assert ids_seq == ids_par, "Mismatch in sample IDs between sequential and parallel runs"
    assert len(results_seq) == len(results_par)
    # Ensure processed_samples increment matches
    assert status_par.processed_samples == len(results_par)
    assert status_seq.processed_samples == len(results_seq)

def test_no_race_in_counter(small_dataset):
    models = [DummyModel('modelB')]
    params = {'num_samples': 2}
    output = DummyOutputManager()
    engine = InferenceEngine(models, small_dataset, params, output_manager=output, batch_size=1, max_workers=4)
    status = Status()
    # Run parallel multiple times to check counter reset
    results = engine._run_parallel_inference(models[0], engine._create_batches(), status)
    # check completed_samples matches
    assert engine.completed_samples == len(results)
    # check saved_status tracked each batch
    assert output.saved_status[-1] == engine.completed_samples

if __name__ == '__main__':
    pytest.main()
