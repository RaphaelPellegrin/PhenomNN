import torch


"""
The current dgl library has:
# def from_coo(
#     row: torch.Tensor,
#     col: torch.Tensor,
#     val: Optional[torch.Tensor] = None,
#     shape: Optional[Tuple[int, int]] = None,
# ) -> SparseMatrix:
#     r""Creates a sparse matrix from a coordinate list (COO), which stores a list
#     of (row, column, value) tuples.

#     See `COO in Wikipedia
#     <https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)>`_.

#     Parameters
#     ----------
#     row : torch.Tensor
#         The row indices of shape ``(nnz)``
#     col : torch.Tensor
#         The column indices of shape ``(nnz)``
#     val : torch.Tensor, optional
#         The values of shape ``(nnz)`` or ``(nnz, D)``. If None, it will be a
#         tensor of shape ``(nnz)`` filled by 1.
#     shape : tuple[int, int], optional
#         If not specified, it will be inferred from :attr:`row` and :attr:`col`,
#         i.e., ``(row.max() + 1, col.max() + 1)``. Otherwise, :attr:`shape`
#         should be no smaller than this.

#     Returns
#     -------
#     SparseMatrix
#         Sparse matrix

#     Examples
#     --------

#     Case1: Sparse matrix with row and column indices without values.

#     >>> dst = torch.tensor([1, 1, 2])
#     >>> src = torch.tensor([2, 4, 3])
#     >>> A = dglsp.from_coo(dst, src)
#     SparseMatrix(indices=tensor([[1, 1, 2],
#                                  [2, 4, 3]]),
#                  values=tensor([1., 1., 1.]),
#                  shape=(3, 5), nnz=3)
#     >>> # Specify shape
#     >>> A = dglsp.from_coo(dst, src, shape=(5, 5))
#     SparseMatrix(indices=tensor([[1, 1, 2],
#                                  [2, 4, 3]]),
#                  values=tensor([1., 1., 1.]),
#                  shape=(5, 5), nnz=3)

#     Case2: Sparse matrix with scalar values.

#     >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
#     >>> val = torch.tensor([[1.], [2.], [3.]])
#     >>> A = dglsp.spmatrix(indices, val)
#     SparseMatrix(indices=tensor([[1, 1, 2],
#                                  [2, 4, 3]]),
#                  values=tensor([[1.],
#                                 [2.],
#                                 [3.]]),
#                  shape=(3, 5), nnz=3, val_size=(1,))

#     Case3: Sparse matrix with vector values.

#     >>> dst = torch.tensor([1, 1, 2])
#     >>> src = torch.tensor([2, 4, 3])
#     >>> val = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
#     >>> A = dglsp.from_coo(dst, src, val)
#     SparseMatrix(indices=tensor([[1, 1, 2],
#                                  [2, 4, 3]]),
#                  values=tensor([[1., 1.],
#                                 [2., 2.],
#                                 [3., 3.]]),
#                  shape=(3, 5), nnz=3, val_size=(2,))
#     ""
#     assert row.shape[0] == col.shape[0]
#     return spmatrix(torch.stack([row, col]), val, shape)



"""


def create_from_coo(src, dst, values, shape):
    """
    Create a sparse tensor from COO format.

    Args:
        src:
            source indices
        dst:
            destination indices
        values:
            values for the sparse tensor
        shape:
            shape of the tensor

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
