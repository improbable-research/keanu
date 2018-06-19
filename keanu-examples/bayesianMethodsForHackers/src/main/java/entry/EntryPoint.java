package entry;

import examples.ABTesting;
import examples.ChallengerDisaster;
import examples.CheatingStudents;
import examples.TextMessaging;
import py4j.GatewayServer;

/**
 * Provides Python API for Graphing the results of Keanu Examples
 *
 * Run "main" to start the JVM, then connect a Python client.
 *
 * example Python code:
 *
%matplotlib inline
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from py4j.java_gateway import JavaGateway
import numpy as np


gateway = JavaGateway()
ab = gateway.entry_point.getCheatingStudentsPosteriors()
p_trace = np.reshape(np.array(list(ab.getFreqCheating())), (len(ab.getFreqCheating()), 1))
print "Finished fitting model"

plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30,
         label="posterior distribution", color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.2)
plt.xlim(0, 1)
plt.legend()

 *
 */
public class EntryPoint
{
    public ChallengerDisaster.ChallengerPosteriors getChallengerPosteriors() {
        return ChallengerDisaster.run();
    }

    public ABTesting.ABTestingPosteriors getABTestingPosteriors() {
        return ABTesting.run();
    }

    public CheatingStudents.CheatingStudentsPosteriors getCheatingStudentsPosteriors() {
        return CheatingStudents.run();
    }

    public TextMessaging.TextMessagingResults getTextMessagingPosteriors() {
        return TextMessaging.run();
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new EntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }
}
