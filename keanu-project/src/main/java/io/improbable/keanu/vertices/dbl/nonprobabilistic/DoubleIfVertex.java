package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;


public class DoubleIfVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final DoubleVertex thn;
    private final DoubleVertex els;
    protected static final String PREDICATE_NAME = "predicate";
    protected static final String THEN_NAME = "then";
    protected static final String ELSE_NAME = "else";

    @ExportVertexToPythonBindings
    public DoubleIfVertex(@LoadParentVertex(PREDICATE_NAME) Vertex<? extends BooleanTensor> predicate,
                          @LoadParentVertex(THEN_NAME) DoubleVertex thn,
                          @LoadParentVertex(ELSE_NAME) DoubleVertex els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    @SaveParentVertex(PREDICATE_NAME)
    public Vertex<? extends BooleanTensor> getPredicate() {
        return predicate;
    }

    @SaveParentVertex(THEN_NAME)
    public DoubleVertex getThn() {
        return els;
    }

    @SaveParentVertex(ELSE_NAME)
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
