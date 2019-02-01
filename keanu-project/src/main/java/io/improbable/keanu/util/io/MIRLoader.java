package io.improbable.keanu.util.io;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.mir.MIR;
import io.improbable.mir.SavedBayesNet;

import java.io.IOException;
import java.io.InputStream;

public class MIRLoader extends ProtobufLoader {

    @Override
    public BayesianNetwork loadNetwork(InputStream input) throws IOException {
        MIR.Model model = MIR.Model.parseFrom(input);
        checkModelIsCompatible(model);

        return extractGraphAndLoadNetwork(model);
    }

    private static void checkModelIsCompatible(MIR.Model model) {
        checkProperties(model);
        checkEntryPoint(model);
        checkFunction(model);
    }

    private static void checkFunction(MIR.Model model) {
        if (model.getFunctionsByNameMap().get(MIRSaver.ENTRY_POINT_NAME) == null) {
            throw new IllegalArgumentException("Expected Entry Point not found");
        }
    }

    private static void checkEntryPoint(MIR.Model model) {
        if (model.getEntryPointName() != MIRSaver.ENTRY_POINT_NAME) {
            throw new IllegalArgumentException("Keanu only supports loading Keanu generated Graphs");
        }
    }

    private static void checkProperties(MIR.Model model) {
        MIR.ModelProperties properties = model.getProperties();

        if (properties.getMirVersion() != MIR.VersionNumber.VERSION_1) {
            throw new IllegalArgumentException("Keanu only supports Version 1 of MIR");
        }

        if (properties.getRecursiveTracesCount() != 0) {
            throw new IllegalArgumentException("Keanu Does not support Recursive models");
        }

        if (properties.getLoopMetadata().getDimensionGenerating()
            || properties.getRecursionMetadata().getDimensionGenerating()) {
            throw new IllegalArgumentException("Keanu does not support Dimension Generation");
        }

        if (properties.getLoopMetadata().getIteration() != MIR.IterationType.NONE
            || properties.getRecursionMetadata().getIteration() != MIR.IterationType.NONE) {
            throw new IllegalArgumentException("Keanu does not support iteration");
        }
    }

    private BayesianNetwork extractGraphAndLoadNetwork(MIR.Model model) {
        SavedBayesNet.Graph graph = model
            .getFunctionsByNameMap()
            .get(MIRSaver.ENTRY_POINT_NAME)
            .getInstructionGroups(0)
            .getGraph();

        return loadNetwork(graph);
    }
}
