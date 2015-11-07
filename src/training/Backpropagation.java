package training;

import feedforward.FeedForwardNeuralNetwork;

/**
 * Created by joshua on 10/14/15.
 * This class teaches a given neural network with the backpropagation algorithm.
 */
public class Backpropagation implements Trainer
{
    private double learningRate;
    private double momentum;
    private double[] lastDeltas;
    private double[] parameters;

    public Backpropagation(double[] parameters)
    {
        this.parameters = parameters;
    }

    /**
     * This is the main runner for the algorithm, it loops through the examples,
     * training the network on each example
     * @param net the starting network
     * @param examples a 2-d array of examples. each example has a list of inputs and
     *                 expected outputs.
     * @return returns a network that is the result of running backpropagation on the examples
     */
    @Override
    public FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples)
    {
        System.out.println("Starting backprop");
        lastDeltas = new double[net.getWeights().length];
        learningRate = parameters[0];
        momentum = parameters[1];

        int[] sizes = net.getSizes();

        int value = 100000 / examples.length;
        for(int i = 0; i < value + 1; i++)
        {
            //for each example
            for(int k = 0; k < examples.length; k++)
            {
                double[] input = new double[sizes[0]];
                double[] output = new double[sizes[sizes.length - 1]];

                //separate input and output
                for(int a = 0; a < input.length; a++)
                {
                    input[a] = examples[k][a];
                }
                output[(int)Math.round(examples[k][examples[0].length - 1])] = 1;

                //run backprop on it
                backprop(input, output, net);
            }
        }

        double[] weights = net.getWeights();
        for(int i = 0; i < weights.length; i++)
        {
            System.out.print(weights[i] + ", ");
        }
        System.out.println();
        return net;
    }

    /**
     * Given an example with the inputs and expected outputs, trains the network
     * @param inputs the inputs for the example
     * @param expectedOutputs what the outputs should be
     * @param net the network to train
     */
    public void backprop(double[] inputs, double[] expectedOutputs, FeedForwardNeuralNetwork net)
    {
        //create variables that will be used later
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
                if(k != hiddenLayers + 1)
                {
                    allOutputs[k][a] = net.applyActivationFunction(sum, net.getHiddenActivationFunction());
                    allErrors[k][a] = net.applyActivationFunctionDerivative(sum, net.getHiddenActivationFunction());
                }
                else
                {
                    allOutputs[k][a] = net.applyActivationFunction(sum, net.getOutputActivationFunction());
                    allErrors[k][a] = net.applyActivationFunctionDerivative(sum, net.getOutputActivationFunction());
                }
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
