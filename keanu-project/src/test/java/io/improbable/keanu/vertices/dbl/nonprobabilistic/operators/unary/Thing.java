//package io.improbable.remorsefulradish.model.vertices;
//
//import io.improbable.keanu.tensor.dbl.DoubleTensor;
//import io.improbable.keanu.vertices.Vertex;
//import io.improbable.keanu.vertices.dbl.DoubleVertex;
//import io.improbable.keanu.vertices.dbl.KeanuRandom;
//import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
//import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;
//import io.improbable.remorsefulradish.model.acoustics.AcousticModel;
//import io.improbable.remorsefulradish.model.coordinates.RelativeCoordinates;
//
//import java.util.Map;
//
//public class TransmissionLossVertex extends DoubleVertex {
//    private DoubleVertex redPosition;
//    private DoubleVertex bluePosition;
//    private DoubleVertex redFrequency;
//    private AcousticModel acousticModel;
//
//    public TransmissionLossVertex(DoubleVertex redPosition,
//                                  DoubleVertex bluePosition,
//                                  DoubleVertex redFrequency,
//                                  AcousticModel acousticModel) {
//        super(new NonProbabilisticValueUpdater<>(v -> ((TransmissionLossVertex) v).op(redPosition.getValue(), bluePosition.getValue(), redFrequency.getValue())));
//        this.redPosition = redPosition;
//        this.bluePosition = bluePosition;
//        this.redFrequency = redFrequency;
//        this.acousticModel = acousticModel;
//        setParents(redPosition, bluePosition, redFrequency);
//    }
//
//    @Override
//    public DoubleTensor sample(KeanuRandom random) {
//        return op(redPosition.sample(random), bluePosition.sample(random), redFrequency.sample(random));
//    }
//
//    private DoubleTensor op(DoubleTensor redPosition, DoubleTensor bluePosition, DoubleTensor redFrequency) {
//        RelativeCoordinates sourcePosition = new RelativeCoordinates(redPosition);
//        RelativeCoordinates receiverPosition = new RelativeCoordinates(bluePosition);
//        double frequency = redFrequency.getValue(0);
//
//        double transmissionLoss = acousticModel.getTransmissionLoss(sourcePosition, receiverPosition, frequency);
//
//        return DoubleTensor.scalar(transmissionLoss);
//    }
//
//    @Override
//    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
//        return null;
//    }
//}