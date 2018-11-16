package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.*;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class DoubleIfVertex extends DoubleVertex implements SaveableVertex, Differentiable, NonProbabilistic<DoubleTensor> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final DoubleVertex thn;
    private final DoubleVertex els;
    protected static final String PREDICATE_NAME = "predicate";
    protected static final String THN_NAME = "thn";
    protected static final String ELS_NAME = "els";

    @ExportVertexToPythonBindings
    public DoubleIfVertex(long[] shape,
                          Vertex<? extends BooleanTensor> predicate,
                          DoubleVertex thn,
                          DoubleVertex els) {
        super(shape);
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    public DoubleIfVertex(@LoadParentVertex(PREDICATE_NAME) Vertex<? extends BooleanTensor> predicate,
                          @LoadParentVertex(THN_NAME) DoubleVertex thn,
                          @LoadParentVertex(ELS_NAME) DoubleVertex els) {

        this(checkHasSingleNonScalarShapeOrAllScalar(thn.getShape(), els.getShape()), predicate, thn, els);
    }

    @SaveParentVertex(PREDICATE_NAME)
    public Vertex<? extends BooleanTensor> getPredicate() {
        return predicate;
    }

    @SaveParentVertex(THN_NAME)
    public DoubleVertex getThn() {
        return els;
    }

    @SaveParentVertex(ELS_NAME)
    public DoubleVertex getEls() {
        return thn;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {

        long[] ofShape = getShape();
        PartialDerivatives thnPartial = derivativeOfParentsWithRespectToInputs.get(thn);
        PartialDerivatives elsPartial = derivativeOfParentsWithRespectToInputs.get(els);
        BooleanTensor predicateValue = predicate.getValue();

        if (predicateValue.allTrue()) {
            return thnPartial;
        } else if (predicateValue.allFalse()) {
            return elsPartial;
        } else {
            return thnPartial.multiplyAlongOfDimensions(predicateValue.toDoubleMask(), ofShape)
                .add(elsPartial.multiplyAlongOfDimensions(predicateValue.not().toDoubleMask(), ofShape));
        }
    }

    @Override
    public DoubleTensor calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    private DoubleTensor op(BooleanTensor predicate, DoubleTensor thn, DoubleTensor els) {
        return predicate.doubleWhere(thn, els);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        BooleanTensor predicateValue = predicate.getValue();
        partials.put(thn, derivativeOfOutputsWithRespectToSelf
            .multiplyAlongWrtDimensions(predicateValue.toDoubleMask(), this.getShape()));
        partials.put(els, derivativeOfOutputsWithRespectToSelf
            .multiplyAlongWrtDimensions(predicateValue.not().toDoubleMask(), this.getShape()));
        return partials;
    }

}
