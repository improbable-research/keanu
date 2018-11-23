package io.improbable.keanu.tensor;

import static org.junit.Assert.assertEquals;

public class TensorTestHelper {

    public static void doesDownRankOnSliceRank3To2(Tensor x) {
        Tensor slice = x.slice(2, 0);
        assertEquals(2, slice.getRank());
    }

    public static void doesDownRankOnSliceRank2To1(Tensor x) {
        Tensor slice = x.slice(1, 0);
        assertEquals(1, slice.getRank());
    }

    public static void doesDownRankOnSliceRank1ToScalar(Tensor x) {
        Tensor slice = x.slice(0, 0);
        assertEquals(0, slice.getRank());
    }
}
