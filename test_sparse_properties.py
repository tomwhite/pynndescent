from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix, vstack

from numpy.testing import assert_allclose

def test_lil():
    A = lil_matrix((3, 2))

    # item assignment
    A[1, 0] = 7

    # subscriptable
    assert A[1, 0] == 7

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
    A = dok_matrix((3, 2))

    # item assignment
    A[1, 0] = 7

    # subscriptable
    assert A[1, 0] == 7

    # no rowview

    # vstack
    B = dok_matrix((1, 2))
    B[0, 0] = 9

    C = dok_matrix((4, 2))
    C[1, 0] = 7
    C[3, 0] = 9
    assert_allclose(vstack((A, B), format="dok").toarray(), C.toarray())

def test_coo():
    A = coo_matrix((3, 2))

    # no item assignment

    # not subscriptable

    # no rowview

    # vstack
    B = coo_matrix((1, 2))
    C = coo_matrix((4, 2))
    assert_allclose(vstack((A, B), format="coo").toarray(), C.toarray())

def test_csr():
    A = csr_matrix((3, 2))

    # item assignment (expensive)
    A[1, 0] = 7

    # subscriptable
    assert A[1, 0] == 7

    # no rowview

    # vstack
    B = csr_matrix((1, 2))
    B[0, 0] = 9

    C = csr_matrix((4, 2))
    C[1, 0] = 7
    C[3, 0] = 9
    assert_allclose(vstack((A, B), format="csr").toarray(), C.toarray())