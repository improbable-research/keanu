package io.improbable.keanu.tensor.dbl;

import org.junit.Test;

import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertTimesInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertTimesOperationEquals;

public class JVMBroadcastTensorTest {

    @Test
    public void canBroadcastMultiplyRank1AndMatrix() {

        /*
          a = np.array([1, 2, 3]).reshape(3)
          b = np.array([1, 2, 3, 5, 6, 7]).reshape(2, 3)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = JVMDoubleTensor.create(new double[]{

            1, 2, 3
        }, new long[]{3});

        DoubleTensor matrix = JVMDoubleTensor.create(new double[]{
            1, 2, 3,
            5, 6, 7
        }, new long[]{2, 3});

        DoubleTensor expected = JVMDoubleTensor.create(new double[]{
            1, 4, 9,
            5, 12, 21
        }, new long[]{2, 3});


        assertTimesOperationEquals(rank4, matrix, expected);
    }
}
