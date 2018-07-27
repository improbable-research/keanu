package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DoubleIfVertex extends NonProbabilisticDouble {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends DoubleTensor> thn;
    private final Vertex<? extends DoubleTensor> els;

    public DoubleIfVertex(int[] shape,
                          Vertex<? extends BooleanTensor> predicate,
                          Vertex<? extends DoubleTensor> thn,
                          Vertex<? extends DoubleTensor> els) {

        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    private DoubleTensor op(BooleanTensor predicate, DoubleTensor thn, DoubleTensor els) {
        return predicate.setDoubleIf(thn, els);
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        BooleanTensor predicateValue = predicate.getValue();
        DualNumber thnDual = dualNumbers.get(thn);
        DualNumber elsDual = dualNumbers.get(els);
        Map<Long, List<DoubleTensor>> toBeConcatted = new HashMap<>();

        int[] thnShape = thn.getShape();

        if (predicateValue.isScalar()) {
            //or all are true
            if (predicateValue.scalar()) {
                return new DualNumber(dualNumbers.get(thn).getValue(), dualNumbers.get(thn).getPartialDerivatives());
            //or all are false
            } else {
                return new DualNumber(dualNumbers.get(els).getValue(), dualNumbers.get(els).getPartialDerivatives());
            }
        } else {

            double[] flatBools = predicateValue.asFlatDoubleArray();

            for (int i = 0; i < predicateValue.getLength(); i++) {
                boolean bool = flatBools[i] == 1.0;
                if (bool) {
                    DualNumber thnDuals = thnDual.pluck(thnShape, TensorShape.getFlatIndex(thnShape, TensorShape.getRowFirstStride(thnShape), i));
                    DualNumber elsDuals = elsDual.pluck(thnShape, TensorShape.getFlatIndex(thnShape, TensorShape.getRowFirstStride(thnShape), i));
                    Map<Long, DoubleTensor> thnDualsMap = thnDuals.getPartialDerivatives().asMap();
                    Map<Long, DoubleTensor> elsWithThnRemoved = removePrimaryFromSecondary(thnDuals, elsDuals);
                    addToMap(toBeConcatted, thnDualsMap, elsWithThnRemoved);
                } else {
                    DualNumber thnDuals = thnDual.pluck(thnShape, TensorShape.getFlatIndex(thnShape, TensorShape.getRowFirstStride(thnShape), i));
                    DualNumber elsDuals = elsDual.pluck(thnShape, TensorShape.getFlatIndex(thnShape, TensorShape.getRowFirstStride(thnShape), i));
                    Map<Long, DoubleTensor> elsDualsMap = elsDuals.getPartialDerivatives().asMap();
                    Map<Long, DoubleTensor> thnWithElsRemoved = removePrimaryFromSecondary(elsDuals, thnDuals);
                    addToMap(toBeConcatted, elsDualsMap, thnWithElsRemoved);
                }
            }
        }
        return null;
    }

    private Map<Long, List<DoubleTensor>> addToMap(Map<Long, List<DoubleTensor>> toBeConcatted, Map<Long, DoubleTensor> a, Map<Long, DoubleTensor> b) {
        for (Map.Entry<Long, DoubleTensor> entry : a.entrySet()) {
            toBeConcatted.computeIfAbsent(entry.getKey(), k -> new ArrayList<>()).add(entry.getValue());
        }
        for (Map.Entry<Long, DoubleTensor> entry : b.entrySet()) {
            toBeConcatted.computeIfAbsent(entry.getKey(), k -> new ArrayList<>()).add(entry.getValue());
        }
        return toBeConcatted;
    }

    private Map<Long, DoubleTensor> removePrimaryFromSecondary(DualNumber primary, DualNumber secondary) {
        Map<Long, DoubleTensor> primaryMap = primary.getPartialDerivatives().asMap();
        Map<Long, DoubleTensor> secondaryWithPrimaryRemoved = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : secondary.getPartialDerivatives().asMap().entrySet()) {
            if (!primaryMap.containsKey(entry.getKey())) {
                DoubleTensor toZero = secondary.getPartialDerivatives().asMap().get(entry.getKey());
                DoubleTensor zeroes = DoubleTensor.zeros(toZero.getShape());
                secondaryWithPrimaryRemoved.put(entry.getKey(), zeroes);
            }
        }

        return secondaryWithPrimaryRemoved;
    }

}
