# Based on https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
import tables
import numpy as np

class BigH5Array():
    def __init__(self, filename, shape, atom=tables.Float32Atom()):
        self.filename = filename
        self.shape = shape
        self.atom = atom
    def open_for_write(self):
        self.f = tables.open_file(self.filename, mode='w')
        self.array_c = self.f.create_carray(self.f.root, 'carray', self.atom, self.shape)
    def open_for_write_expandable(self):
        self.f = tables.open_file(self.filename, mode='w')
        self.array_e = self.f.create_earray(self.f.root, 'data', self.atom, [0] + list(self.shape[1:]))
    def open_for_read(self):
        self.f = tables.open_file(self.filename, mode='r')
    def data(self): # for expandable
        # bigarray.data()[1:10,2:20]
        return self.f.root.data
    def append(self, row_data): # for expandable
        self.array_e.append(row_data)
    def __call__(self): # for random access
        return self.f.root.carray
    def close(self):
        self.f.close()

def big_h5_load(filename, shape):
    bigfile = BigH5Array(filename, shape)
    bigfile.open_for_read()
    bigarray = np.array(bigfile())
    bigfile.close()
    return bigarray

if __name__ == '__main__':
    import unittest

    class TestBigH5Array(unittest.TestCase):
        ROW_SIZE = 2*2*1056
        COL_SIZE = 23

        def recursive_test_almost_equal(self, a, b, msg):
            if isinstance(a, (list, np.ndarray)):
                for i, (_a, _b) in enumerate(zip(a, b)):
                    self.recursive_test_almost_equal(_a, _b, msg+('{},'.format(i)))
            else:
                self.assertAlmostEqual(a, b, msg=msg)

        def dotest_as_normal_array(self, shape, testdata):
            # write test - just confirm no error
            h5array = BigH5Array('test.h5', shape)
            h5array.open_for_write()
            h5array()[...] = testdata
            h5array.close()

            # read test - real test
            h5array = BigH5Array('test.h5', shape)
            h5array.open_for_read()
            for col in range(shape[0]):
                print('Testing ...[%d]' % col, testdata.shape, testdata[col])
                self.recursive_test_almost_equal(h5array()[col], testdata[col], 'failed@')
            h5array.close()

            # big_h5_load test
            x = big_h5_load('test.h5', shape)
            for col in range(shape[0]):
                print('Testing big_h5_load ...[%d]' % col, testdata.shape, testdata[col])
                self.recursive_test_almost_equal(x[col], testdata[col], 'failed@')

            print('')

        def test_1_as_normal_array(self):
            self.dotest_as_normal_array((TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE),
                                        np.random.rand(TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE))

        def test_2_as_3d_normal_array(self):
            self.dotest_as_normal_array((TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE, 3),
                                        np.random.rand(TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE, 3))

        def test_3_write_expandable(self):
            # write test - just confirm no error
            self.test_data = []
            writer = BigH5Array('test.h5', (TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE))
            writer.open_for_write_expandable()
            for col in range(TestBigH5Array.COL_SIZE):
                x = np.random.rand(1, TestBigH5Array.ROW_SIZE)
                self.test_data.append(x[0])
                writer.append(x)
            writer.close()

            # read test - real test
            reader = BigH5Array('test.h5', (TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE))
            reader.open_for_read()
            self.test_data = np.array(self.test_data)
            for col in range(TestBigH5Array.COL_SIZE):
                print('Testing ...[%d]' % col, self.test_data.shape, self.test_data[col])
                self.recursive_test_almost_equal(reader.data()[col], self.test_data[col], 'failed@')
            reader.close()

            print('')

    unittest.main()
