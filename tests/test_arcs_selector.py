import torch

from arcs.selector import ARCSSelector


def test_arcs_selector_returns_unique_budget():
    torch.manual_seed(7)

    pool_size = 100
    num_classes = 10
    budget = 20

    embeddings = torch.randn(pool_size, 16)
    fp_logits = torch.randn(pool_size, num_classes)
    q_logits = fp_logits + 0.2 * torch.randn(pool_size, num_classes)
    dataset_indices = list(range(1000, 1000 + pool_size))

    result = ARCSSelector().select(
        embeddings=embeddings,
        dataset_indices=dataset_indices,
        fp_logits=fp_logits,
        q_logits=q_logits,
        num_select=budget,
        seed=1005,
    )

    assert len(result.selected_dataset_indices) == budget
    assert len(set(result.selected_dataset_indices)) == budget
    assert set(result.selected_dataset_indices).issubset(set(dataset_indices))
    assert 0.0 <= result.metadata["severity"] <= 1.0
