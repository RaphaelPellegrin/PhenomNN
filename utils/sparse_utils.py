import torch


def create_from_coo(src, dst, values, shape):
    """
    Create a sparse tensor from COO format.

    Args:
        src: source indices
        dst: destination indices
        values: values for the sparse tensor
        shape: shape of the tensor

    Returns:
        Sparse tensor
    """
    indices = torch.stack([src, dst])
    return torch.sparse_coo_tensor(indices, values, shape)


def diag(values):
    """
    Create a diagonal matrix from values.

    Args:
        values: diagonal values

    Returns:
        Diagonal matrix as a sparse tensor
    """
    # If values is a sparse tensor, convert to dense
    if torch.is_tensor(values) and values.is_sparse:
        values = values.to_dense()
    if isinstance(values, torch.Tensor):
        n = values.size(0)
        indices = torch.arange(n, device=values.device)
        indices = torch.stack([indices, indices])
        return torch.sparse_coo_tensor(indices, values, (n, n))
    else:
        # Handle scalar or list input
        if not isinstance(values, (list, tuple)):
            values = [values]
        n = len(values)
        indices = torch.arange(n)
        indices = torch.stack([indices, indices])
        return torch.sparse_coo_tensor(
            indices, torch.tensor(values, dtype=torch.float32), (n, n)
        )


def identity(size, device=None):
    """
    Create an identity matrix.

    Args:
        size: size of the identity matrix
        device: device to place the tensor on

    Returns:
        Identity matrix as a sparse tensor
    """
    if isinstance(size, (tuple, torch.Size)):
        n = size[0]
    else:
        n = size
    indices = torch.arange(n, device=device)
    indices = torch.stack([indices, indices])
    values = torch.ones(n, device=device, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n))
