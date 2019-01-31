from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix, vstack

import sparse

from numpy.testing import assert_allclose

def test_lil():
    # 2D only
    A = lil_matrix((3, 2))

    # item assignment
    A[1, 0] = 7

    # subscriptable
    assert A[1, 0] == 7

    # no fill value (unset elements always 0)
    assert A[1, 1] == 0

    # rowview allows updates
    row = A.getrowview(1)
    row[0, 0] = 8
    assert A[1, 0] == 8

    # rows are not truly sparse since some storage is used even for empty rows
    assert len(A.rows) == 3

    # vstack
    B = lil_matrix((1, 2))
    B[0, 0] = 9

    C = lil_matrix((4, 2))
    C[1, 0] = 8
    C[3, 0] = 9
    assert_allclose(vstack((A, B), format="lil").toarray(), C.toarray())

def test_dok():
    # 2D only
    A = dok_matrix((3, 2))

    # item assignment
    A[1, 0] = 7

    # subscriptable
    assert A[1, 0] == 7

    # no fill value (unset elements always 0)
    assert A[1, 1] == 0

    # no rowview

    # vstack
    B = dok_matrix((1, 2))
    B[0, 0] = 9

    C = dok_matrix((4, 2))
    C[1, 0] = 7
    C[3, 0] = 9
    assert_allclose(vstack((A, B), format="dok").toarray(), C.toarray())

def test_coo():
    # 2D only
    A = coo_matrix((3, 2))

    # no item assignment

    # not subscriptable

    # no rowview

    # vstack
    B = coo_matrix((1, 2))
    C = coo_matrix((4, 2))
    assert_allclose(vstack((A, B), format="coo").toarray(), C.toarray())

def test_csr():
    # 2D only
    A = csr_matrix((3, 2))

    # item assignment (expensive)
    A[1, 0] = 7

    # subscriptable
    assert A[1, 0] == 7

    # no fill value (unset elements always 0)
    assert A[1, 1] == 0

    # no rowview

    # vstack
    B = csr_matrix((1, 2))
    B[0, 0] = 9

    C = csr_matrix((4, 2))
    C[1, 0] = 7
    C[3, 0] = 9
    assert_allclose(vstack((A, B), format="csr").toarray(), C.toarray())

def test_pydata_dok():
    # 3D
    A = sparse.DOK((3, 2, 2), fill_value=-1)

    # item assignment
    A[1, 0, 0] = 7

    # subscriptable
    assert A[1, 0, 0] == 7

    # fill value
    assert A[1, 1, 0] == -1

    # no rowview

    # vstack (concatenate)
    B = sparse.DOK((1, 2, 2), fill_value=-1)
    B[0, 0, 0] = 9

    C = sparse.DOK((4, 2, 2), fill_value=-1)
    C[1, 0, 0] = 7
    C[3, 0, 0] = 9
    # note that sparse.concatenate always creates a COO array
    assert_allclose(sparse.concatenate((A, B), axis=0).todense(), C.todense())
