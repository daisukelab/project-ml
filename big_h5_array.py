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
        self.array_e = self.f.create_earray(self.f.root, 'data', self.atom, (0, self.shape[1]))
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

if __name__ == '__main__':
    import unittest

    class TestBigH5Array(unittest.TestCase):
        ROW_SIZE = 2*2*1056
        COL_SIZE = 200

        def test_1_as_normal_array(self):
            # write test - just confirm no error
            array = BigH5Array('test2.h5', (TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE))
            self.opened_file = array
            array.open_for_write()
            x = np.random.rand(1, TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE)
            self.test_data = x[0]
            array()[...] = x
            array.close()

            # read test - real test
            array = BigH5Array('test2.h5', (TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE))
            self.opened_file = array
            array.open_for_read()
            self.test_data = np.array(self.test_data)
            for col in range(TestBigH5Array.COL_SIZE):
                print('Testing ...[%d]' % col, self.test_data.shape, self.test_data[col])
                row = 0
                for a, b in zip(array()[col], self.test_data[col]):
                    self.assertAlmostEqual(a, b, msg='fails at [{},{}]'.format(col, row))
                    row += 1
            array.close()

            self.opened_file = None

        def test_2_write_expandable(self):
            # write test - just confirm no error
            self.test_data = []
            writer = BigH5Array('test2.h5', (TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE))
            self.opened_file = writer
            writer.open_for_write_expandable()
            for col in range(TestBigH5Array.COL_SIZE):
                x = np.random.rand(1, TestBigH5Array.ROW_SIZE)
                self.test_data.append(x[0])
                writer.append(x)
            writer.close()

            # read test - real test
            reader = BigH5Array('test2.h5', (TestBigH5Array.COL_SIZE, TestBigH5Array.ROW_SIZE))
            self.opened_file = reader
            reader.open_for_read()
            self.test_data = np.array(self.test_data)
            for col in range(TestBigH5Array.COL_SIZE):
                x = reader.data()[col]
                print('Testing ...[%d]' % col, self.test_data.shape, self.test_data[col])
                row = 0
                for a, b in zip(x, self.test_data[col]):
                    self.assertAlmostEqual(a, b, msg='fails at [{},{}]'.format(col, row))
                    row += 1
            reader.close()

            self.opened_file = None

        def tearDown(self):
            if self.opened_file is not None:
                self.opened_file.close()

    unittest.main()
