import os
import sys
sys.path.append('../')
from module.engine import EarlyStopping, save_checkpoint, _n_unit
import numpy as np
import torch

def test_EarlyStopping():
    early_stopping = EarlyStopping(patience=3, delta= 0.1)
    assert early_stopping.early_stop == False
    assert early_stopping.counter == 0

    # # score가 best_score + delta 보다 작은 경우, counter가 1 증가
    early_stopping(0.01)
    assert early_stopping.counter == 1
    assert early_stopping.early_stop == False

    # # score가 best_score + delta 보다 큰 경우, best_score가 score로 업데이트되고 counter가 초기화
    early_stopping(0.3)
    assert early_stopping.counter == 0
    assert early_stopping.early_stop == False
    assert early_stopping.best_score == 0.3

    # # counter가 patience 값에 도달하면 early_stop이 True로 설정
    early_stopping(0.2)
    assert early_stopping.counter == 1
    assert early_stopping.early_stop == False
    early_stopping(0.1)
    assert early_stopping.counter == 2
    assert early_stopping.early_stop == False
    early_stopping(0.05)
    assert early_stopping.counter == 3
    assert early_stopping.early_stop == True

def test_save_checkpoint():
    # Create a dummy model and a save path
    model = torch.nn.Linear(10, 1)
    save_path = "model.pt"

    # Save the model
    save_checkpoint(model, save_path)

    # Check that the saved file exists
    assert os.path.exists(save_path)

    # Load the saved model
    loaded_model = torch.nn.Linear(10, 1)
    loaded_model.load_state_dict(torch.load(save_path))

    # Check that the loaded model has the same state as the original model
    assert torch.allclose(model.state_dict()["weight"], loaded_model.state_dict()["weight"])
    assert torch.allclose(model.state_dict()["bias"], loaded_model.state_dict()["bias"])

    # Remove the saved file
    os.remove(save_path)


def test_n_unit():
    # Test case 1
    n_layers = 1
    n_input = 10
    n_output = 5
    n_tip_point = 5
    tip_layer = 0
    result = _n_unit(n_layers, n_input, n_output, n_tip_point, tip_layer)
    assert result == [(n_input, n_output)], f"Test case 1 failed: expected {(n_input, n_output)}, but got {result}"

    # Test case 2
    n_layers = 3
    n_input = 10
    n_output = 5
    n_tip_point = 5
    tip_layer = 1
    result = _n_unit(n_layers, n_input, n_output, n_tip_point, tip_layer)
    expected = [(10, 5), (5, 5), (5, 5)]
    assert result == expected, f"Test case 2 failed: expected {expected}, but got {result}"

    # Test case 3
    n_layers = 4
    n_input = 10
    n_output = 5
    n_tip_point = 3
    tip_layer = 2
    result = _n_unit(n_layers, n_input, n_output, n_tip_point, tip_layer)
    expected = [(10, 5), (5, 3), (3, 4), (4, 5)]
    assert result == expected, f"Test case 3 failed: expected {expected}, but got {result}"

    print("All test cases passed!")
