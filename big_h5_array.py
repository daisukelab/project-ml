# Based on https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
import tables
import numpy as np

class BigH5Array():
    def __init__(self, filename, row_size, atom=tables.Float32Atom()):
        self.filename = filename
        self.row_size = row_size
        self.atom = atom
    def open_for_write(self):
        self.f = tables.open_file(self.filename, mode='w')
        self.array_c = self.f.create_earray(self.f.root, 'data', self.atom, (0, self.row_size))
    def open_for_read(self):
        self.f = tables.open_file(self.filename, mode='r')
    def data(self):
        # bigarray.data()[1:10,2:20]
        return self.f.root.data
    def write(self, row_data):
        self.array_c.append(row_data)
    def close(self):
        self.f.close()

if __name__ == '__main__':
    import unittest

    class TestBigH5Array(unittest.TestCase):
        ROW_SIZE = 2*2*1056
        COL_SIZE = 200

        def test_read_write(self):
            # write test - just confirm no error
            self.test_data = []
            writer = BigH5Array('test2.h5', TestBigH5Array.ROW_SIZE)
            self.opened_file = writer
            writer.open_for_write()
            for col in range(TestBigH5Array.COL_SIZE):
                x = np.random.rand(1, TestBigH5Array.ROW_SIZE)
                self.test_data.append(x[0])
                writer.write(x)
            writer.close()

            # read test - real test
            reader = BigH5Array('test2.h5', TestBigH5Array.ROW_SIZE)
            self.opened_file = writer
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
