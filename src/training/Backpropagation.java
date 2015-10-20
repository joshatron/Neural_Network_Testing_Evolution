package training;

import feedforward.FeedForwardNeuralNetwork;

/**
 * Created by joshua on 10/14/15.
 */
public class Backpropagation implements Trainer
{
    private double learningRate;
    private double momentum;
    private double[] lastDeltas;

    @Override
    public FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples, double[] parameters)
    {
        lastDeltas = new double[net.getWeights().length];
        learningRate = parameters[0];
        momentum = parameters[1];

        int[] sizes = net.getSizes();

        for(int k = 0; k < examples.length; k++)
        {
            double[] input = new double[sizes[0]];
            double[] output = new double[sizes[sizes.length - 1]];

            int t = 0;
            for(int a = 0; a < examples[0].length; a++)
            {
                if(a < input.length)
                {
                    input[a] = examples[k][a];
                }
                else
                {
                    output[a - t] = examples[k][a];
                    t++;
                }
            }

            backprop(input, output, net);
        }

        return net;
    }

    /**
     * Given an example with the inputs and expected outputs, trains the net
     * @param inputs the inputs for the example
     * @param expectedOutputs what the outputs should be
     */
    public void backprop(double[] inputs, double[] expectedOutputs, FeedForwardNeuralNetwork net)
    {
        int[] sizes = net.getSizes();
        int biggestSize = 0;
        for(int k = 0; k < sizes.length; k++)
        {
            if(sizes[k] > biggestSize)
            {
                biggestSize = sizes[k];
            }
        }
        int hiddenLayers = sizes.length - 2;
        //if input or output wrong size, return
        if(inputs.length != sizes[0])
        {
            System.out.println("Invalid number of inputs");
            return;
        }

        if(expectedOutputs.length != sizes[sizes.length - 1])
        {
            System.out.println("Invalid number of outputs");
            return;
        }

        double[][] allOutputs = new double[sizes.length][biggestSize];
        double[][] allErrors = new double[sizes.length][biggestSize];

        //fill out first layer to temp output
        int lastLayer = sizes[0];
        for(int k = 0; k < lastLayer; k++)
        {
            allOutputs[0][k] = inputs[k];
        }

        //for each layer after the input
        for(int k = 1; k < hiddenLayers + 2; k++)
        {
            //for each node in that layer
            for(int a = 0; a < sizes[k]; a++)
            {
                //get sum and get activation function result and its derivative
                double sum = 0;
                for(int t = 0; t < lastLayer; t++)
                {
                    sum += allOutputs[k - 1][t] * net.getWeight(k - 1, t, k, a);
                }
                sum += net.getBiasNum() * net.getWeight(-1, 0, k, a);
                allOutputs[k][a] = net.applyActivationFunction(sum);
                allErrors[k][a] = net.applyActivationFunctionDerivative(sum);
            }
            lastLayer = sizes[k];
        }

        //go backward from output to first hidden layer
        for(int k = hiddenLayers + 1; k > 0; k--)
        {
            //for each node in that layer
            for(int a = 0; a < sizes[k]; a++)
            {
                //compute error for not output layer
                if(k != hiddenLayers + 1)
                {
                    double temp = allErrors[k][a];
                    allErrors[k][a] = 0;
                    for(int t = 0; t < sizes[k + 1]; t++)
                    {
                        allErrors[k][a] += net.getWeight(k, t, k + 1, a) * allErrors[k + 1][t];
                    }
                    allErrors[k][a] *= temp;
                }
                //compute error for output layer
                else
                {
                    allErrors[k][a] *= (expectedOutputs[a] - allOutputs[k][a]);
                }

                //for each weight node takes as input
                for(int t = 0; t < sizes[k - 1]; t++)
                {
                    //find the delta for the weight and apply
                    int index = net.getIndex(k - 1, t, k, a);
                    double delta = learningRate * allOutputs[k - 1][t] * allErrors[k][a]
                            + momentum * lastDeltas[index];

                    net.setWeight(k - 1, t, k, a, net.getWeight(k - 1, t, k, a) + delta);
                    lastDeltas[index] = delta;
                }
            }
        }

    }
}
