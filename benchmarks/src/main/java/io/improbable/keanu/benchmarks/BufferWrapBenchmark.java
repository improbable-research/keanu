package io.improbable.keanu.benchmarks;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

import java.util.Random;
import java.util.function.BiFunction;


@State(Scope.Benchmark)
public class BufferWrapBenchmark {

    public enum DoubleOp implements BiFunction<Double, Double, Double> {
        PLUS {
            public final Double apply(Double lhs, Double rhs) {
                return lhs + rhs;
            }
        }
    }

    public enum IntOp implements BiFunction<Integer, Integer, Integer> {
        PLUS {
            public final Integer apply(Integer lhs, Integer rhs) {
                return lhs + rhs;
            }
        }
    }

    //    @Param({"1", "2", "100", "10000", "1000000"})
    @Param({"100000"})
    public int size;


    double[] leftDbl;
    double[] rightDbl;

    DoubleArrayWrapper leftDblWrapped;
    DoubleArrayWrapper rightDblWrapped;

    int[] leftInt;
    int[] rightInt;

    IntegerArrayWrapper leftIntWrapped;
    IntegerArrayWrapper rightIntWrapped;

    private static final DoubleArrayWrapperFactory doubleFactory = new DoubleArrayWrapperFactory();
    private static final IntegerArrayWrapperFactory intFactory = new IntegerArrayWrapperFactory();

    @Setup
    public void setup() {
        Random r = new Random(1);

        leftDbl = new double[size];
        rightDbl = new double[size];

        leftDblWrapped = doubleFactory.create(size);
        rightDblWrapped = doubleFactory.create(size);

        leftInt = new int[size];
        rightInt = new int[size];

        leftIntWrapped = intFactory.create(size);
        rightIntWrapped = intFactory.create(size);

        for (int i = 0; i < size; i++) {
            leftDbl[i] = r.nextDouble();
            rightDbl[i] = r.nextDouble();
            leftInt[i] = r.nextInt();
            rightInt[i] = r.nextInt();

            leftDblWrapped.set(leftDbl[i], i);
            rightDblWrapped.set(rightDbl[i], i);

            leftIntWrapped.set(leftInt[i], i);
            rightIntWrapped.set(rightInt[i], i);
        }
    }

    @Benchmark
    public Object[] benchmarkWrappedWithLambda() {
        Object[] result = new Object[6];

        result[0] = wrappedWithLambdaGeneric(leftDblWrapped, rightDblWrapped, doubleFactory.create(size), DoubleOp.PLUS);
        result[1] = wrappedWithLambdaGeneric(leftDblWrapped, rightDblWrapped, doubleFactory.create(size), DoubleOp.PLUS);
        result[2] = wrappedWithLambdaGeneric(leftDblWrapped, rightDblWrapped, doubleFactory.create(size), DoubleOp.PLUS);

//        DoubleArrayWrapper a = doubleFactory.create(size);
//        IntegerArrayWrapper b = intFactory.create(size);
//
//        for (int i = 0; i < size; i++) {
//            a.set(DoubleOp.PLUS.apply(leftDblWrapped.get(i), rightDblWrapped.get(i)), i);
//            b.set(IntOp.PLUS.apply(leftIntWrapped.get(i), rightIntWrapped.get(i)), i);
//        }
//
//        result[0] = a;

        result[3] = wrappedWithLambdaGeneric(leftIntWrapped, rightIntWrapped, intFactory.create(size), IntOp.PLUS);
        result[4] = wrappedWithLambdaGeneric(leftIntWrapped, rightIntWrapped, intFactory.create(size), IntOp.PLUS);
        result[5] = wrappedWithLambdaGeneric(leftIntWrapped, rightIntWrapped, intFactory.create(size), IntOp.PLUS);


//        for (int i = 0; i < size; i++) {
//        }

//        result[1] = b;

        return result;
    }

    @Benchmark
    public Object[] benchmarkPrimitiveInLine() {

        Object[] result = new Object[6];

        result[0] = addDoubles(leftDbl, rightDbl, new double[size]);
        result[1] = addDoubles(leftDbl, rightDbl, new double[size]);
        result[2] = addDoubles(leftDbl, rightDbl, new double[size]);
        result[3] = addInts(leftInt, rightInt, new int[size]);
        result[4] = addInts(leftInt, rightInt, new int[size]);
        result[5] = addInts(leftInt, rightInt, new int[size]);

        return result;
    }

    public static double[] addDoubles(double[] leftDbl, double[] rightDbl, double[] buffer) {

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = leftDbl[i] + rightDbl[i];
        }

        return buffer;
    }

    public static int[] addInts(int[] leftInt, int[] rightInt, int[] buffer) {

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = leftInt[i] + rightInt[i];
        }

        return buffer;
    }

    public static <T extends Number> PrimitiveArrayWrapper<T> wrappedWithLambdaGeneric(final PrimitiveArrayWrapper<T> left,
                                                                                       final PrimitiveArrayWrapper<T> right,
                                                                                       final PrimitiveArrayWrapper<T> result,
                                                                                       final BiFunction<T, T, T> op) {
        final int length = result.getLength();

        for (int i = 0; i < length; i++) {
            result.set(op.apply(left.get(i), right.get(i)), i);
        }

        return result;
    }

    public static DoubleArrayWrapper wrappedWithLambdaDouble(final DoubleArrayWrapper left,
                                                             final DoubleArrayWrapper right,
                                                             final DoubleArrayWrapper result,
                                                             final BiFunction<Double, Double, Double> op) {
        final int length = result.getLength();

        for (int i = 0; i < length; i++) {
            result.set(op.apply(left.get(i), right.get(i)), i);
        }

        return result;
    }

    public static IntegerArrayWrapper wrappedWithLambdaInt(final IntegerArrayWrapper left,
                                                           final IntegerArrayWrapper right,
                                                           final IntegerArrayWrapper result,
                                                           final BiFunction<Integer, Integer, Integer> op) {
        final int length = result.getLength();

        for (int i = 0; i < length; i++) {
            result.set(op.apply(left.get(i), right.get(i)), i);
        }

        return result;
    }

    public interface PrimitiveArrayWrapper<T> {
        T get(int index);

        void set(T value, int index);

        int getLength();
    }

    public static final class DoubleArrayWrapper implements PrimitiveArrayWrapper<Double> {

        private final double[] array;

        public DoubleArrayWrapper(final double[] array) {
            this.array = array;
        }

        @Override
        public Double get(final int index) {
            return array[index];
        }

        @Override
        public void set(final Double value, final int index) {
            array[index] = value;
        }

        @Override
        public int getLength() {
            return array.length;
        }
    }

    public static final class IntegerArrayWrapper implements PrimitiveArrayWrapper<Integer> {

        private final int[] array;

        public IntegerArrayWrapper(final int[] array) {
            this.array = array;
        }

        @Override
        public Integer get(final int index) {
            return array[index];
        }

        @Override
        public void set(final Integer value, final int index) {
            array[index] = value;
        }

        @Override
        public int getLength() {
            return array.length;
        }
    }

    public static final class DoubleWrapper implements PrimitiveArrayWrapper<Double> {

        double value;

        public DoubleWrapper(final double value) {
            this.value = value;
        }

        @Override
        public Double get(final int index) {
            return value;
        }

        @Override
        public void set(final Double value, final int index) {
            this.value = value;
        }

        @Override
        public int getLength() {
            return 1;
        }
    }

    public static final class IntegerWrapper implements PrimitiveArrayWrapper<Integer> {

        private int value;

        public IntegerWrapper(final int value) {
            this.value = value;
        }

        @Override
        public Integer get(final int index) {
            return value;
        }

        @Override
        public void set(final Integer value, final int index) {
            this.value = value;
        }

        @Override
        public int getLength() {
            return 1;
        }
    }

    public interface ArrayWrapperFactory<D, T extends PrimitiveArrayWrapper<D>> {
        T create(int size);
    }

    public static final class DoubleArrayWrapperFactory implements ArrayWrapperFactory<Double, DoubleArrayWrapper> {

        @Override
        public DoubleArrayWrapper create(final int size) {
//            if (size == 1) {
//                return new DoubleWrapper(0);
//            } else {
            return new DoubleArrayWrapper(new double[size]);
//            }
        }
    }

    public static final class IntegerArrayWrapperFactory implements ArrayWrapperFactory<Integer, IntegerArrayWrapper> {

        @Override
        public IntegerArrayWrapper create(final int size) {
//            if (size == 1) {
//                return new IntegerWrapper(0);
//            } else {
            return new IntegerArrayWrapper(new int[size]);
//            }
        }
    }
}