package feedforward;

import org.apache.commons.io.FileUtils;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.util.Random;

/**
 * Created by joshua on 9/22/15.
 * This class creates a fully connected feedforward neural network
 * with a variable number of hidden layers.
 * It can be imported and exported through the JSON format to make
 * saving the net much easier. There is also a regular constructor
 * which will create the net with random weights.
 * The net can currently use 2 different activation functions,
 * a linear function with a slope of 1 and the logistic function.
 * The net handles all learning. Pass it an array of JSON examples
 * and it will perform backprop once on each example. You can adjust
 * the momentum and learning rate when you construct the net.
 * Each node is connected to a bias node which has a value of 1.
 */
public class FeedForwardNeuralNetwork
{
    private double momentum;
    private double learningRate;
    private double[] weights;
    private int hiddenLayers;
    private int[] sizes;
    private int biggestSize;
    private ActivationFunction activationFunction;
    private double biasNum = 1.;
    private double linearSlope = 1.;

    private double[] lastDeltas;

    /**
     * Creates the net from a JSON file
     * @param file The file containing the JSON
     * @throws IOException
     */
    public FeedForwardNeuralNetwork(File file) throws IOException
    {
        this(getJSONFromFile(file));
    }

    /**
     * Creates net from string containing JSON
     * @param net the string containing the JSON
     */
    public FeedForwardNeuralNetwork(JSONObject net)
    {
        momentum = net.getDouble("momentum");
        learningRate = net.getDouble("learningRate");
        hiddenLayers = net.getInt("hiddenLayers");

        String activation = net.getString("activationFunction");
        if(activation.equals("LINEAR"))
        {
            activationFunction = ActivationFunction.LINEAR;
        }
        else if(activation.equals("LOGISTIC"))
        {
            activationFunction = ActivationFunction.LOGISTIC;
        }
        else
        {
            System.out.println("Illegal activation function.");
        }

        JSONArray sizeArray = net.getJSONArray("sizes");
        if(sizeArray.length() == hiddenLayers + 2)
        {
            sizes = new int[hiddenLayers + 2];
            biggestSize = 0;
            for(int k = 0; k < hiddenLayers + 2; k++)
            {
                int size = sizeArray.getInt(k);
                sizes[k] = size;
                if(size > biggestSize)
                {
                    biggestSize = size;
                }
            }
        }
        else
        {
            System.out.println("Illegal number of layer sizes.");
        }

        if(net.has("weights"))
        {
            int totalWeights = 0;
            for(int k = 0; k < sizes.length - 1; k++)
            {
                totalWeights += sizes[k] * sizes[k + 1];
                totalWeights += sizes[k];
            }
            totalWeights += sizes[sizes.length - 1];
            JSONArray weightArray = net.getJSONArray("weights");
            if(weightArray.length() == totalWeights)
            {
                weights = new double[totalWeights];
                for(int k = 0; k < totalWeights; k++)
                {
                    weights[k] = weightArray.getDouble(k);
                }
            }
            else
            {
                System.out.println("Illegal number of weights");
            }
        }
        else
        {
            generateRandomWeights();
        }

        lastDeltas = new double[weights.length];
    }

    /**
     * Initialize a new net with the following parameters and random weights
     * @param hiddenLayers Number of hidden layers
     * @param sizes Size of each layer, starting with the input and ending with the output
     * @param activationFunction Activation function to use
     * @param momentum Momentum for learning
     * @param learningRate Learning rate
     */
    public FeedForwardNeuralNetwork(int hiddenLayers, int[] sizes, ActivationFunction activationFunction, double momentum, double learningRate)
    {
        this.hiddenLayers = hiddenLayers;
        this.sizes = sizes;
        this.activationFunction = activationFunction;
        this.momentum = momentum;
        this.learningRate = learningRate;

        biggestSize = 0;
        for(int k = 0; k < sizes.length; k++)
        {
            if(sizes[k] > biggestSize)
            {
                biggestSize = sizes[k];
            }
        }

        generateRandomWeights();
        lastDeltas = new double[weights.length];
    }

    /**
     * Fills all the weights with random numbers between -1 and 1
     */
    private void generateRandomWeights()
    {
        int lowest = -1;
        int highest = 1;

        int totalWeights = 0;
        for(int k = 0; k < sizes.length - 1; k++)
        {
            totalWeights += sizes[k] * sizes[k + 1];
            totalWeights += sizes[k];
        }
        totalWeights += sizes[sizes.length - 1];

        Random rand = new Random();
        weights = new double[totalWeights];

        for(int k = 0; k < totalWeights; k++)
        {
            weights[k] = lowest + ((highest - lowest) * rand.nextDouble());
        }
    }

    /**
     * Puts JSON file contents into a string
     * @param file File containing JSON
     * @return String containing JSON
     * @throws IOException
     */
    private static JSONObject getJSONFromFile(File file) throws IOException
    {
        String input = FileUtils.readFileToString(file);
        return new JSONObject(input);
    }

    /**
     * Turns the net into a JSON string
     * @return The JSON string
     */
    public JSONObject export()
    {
        JSONObject net = new JSONObject();
        net.put("momentum", momentum);
        net.put("learningRate", learningRate);
        net.put("hiddenLayers", hiddenLayers);
        net.put("activationFunction", activationFunction.name());
        net.put("sizes", new JSONArray(sizes));
        net.put("weights", new JSONArray(weights));

        return net;
    }

    /**
     * Turns the net into a JSON string and saves it to a file
     * @param file File to save JSON to
     * @throws IOException
     */
    public void export(File file) throws IOException
    {
        JSONObject net = export();

        BufferedWriter out = new BufferedWriter(new PrintWriter(new FileWriter(file)));
        out.write(net.toString(2));
        out.close();
    }

    /**
     * Compute output nodes based on input
     * @param inputs Values for input layer
     * @return Values of the output layer
     */
    public double[] compute(double[] inputs)
    {
        if(inputs.length != sizes[0])
        {
            System.out.println("Invalid number of inputs");
            return null;
        }

        int lastLayer = sizes[0];
        double[] layerOut = new double[biggestSize];
        for(int k = 0; k < lastLayer; k++)
        {
            layerOut[k] = inputs[k];
        }

        for(int k = 1; k < hiddenLayers + 2; k++)
        {
            double[] tempOut = new double[sizes[k]];
            for(int a = 0; a < sizes[k]; a++)
            {
                double sum = 0;
                for(int t = 0; t < lastLayer; t++)
                {
                    sum += layerOut[t] * getWeight(k - 1, t, k, a);
                }
                sum += biasNum * getWeight(-1, 0, k, a);
                tempOut[a] = applyActivationFunction(sum);
            }
            lastLayer = sizes[k];
            for(int a = 0; a < lastLayer; a++)
            {
                layerOut[a] = tempOut[a];
            }
        }

        return layerOut;
    }

    /**
     * Given an example with the inputs and expected outputs, teaches the nets
     * @param inputs the inputs for the example
     * @param expectedOutputs what the outputs should be
     */
    public void backprop(double[] inputs, double[] expectedOutputs)
    {
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

        int lastLayer = sizes[0];
        for(int k = 0; k < lastLayer; k++)
        {
            allOutputs[0][k] = inputs[k];
        }

        for(int k = 1; k < hiddenLayers + 2; k++)
        {
            for(int a = 0; a < sizes[k]; a++)
            {
                double sum = 0;
                for(int t = 0; t < lastLayer; t++)
                {
                    sum += allOutputs[k - 1][t] * getWeight(k - 1, t, k, a);
                }
                sum += biasNum * getWeight(-1, 0, k, a);
                allOutputs[k][a] = applyActivationFunction(sum);
                allErrors[k][a] = applyActivationFunctionDerivative(sum);
            }
            lastLayer = sizes[k];
        }

        for(int k = hiddenLayers + 1; k > 0; k--)
        {
            for(int a = 0; a < sizes[k]; a++)
            {
                //not output layer
                if(k != hiddenLayers + 1)
                {
                    double temp = allErrors[k][a];
                    allErrors[k][a] = 0;
                    for(int t = 0; t < sizes[k + 1]; t++)
                    {
                        allErrors[k][a] += getWeight(k, t, k + 1, a) * allErrors[k + 1][t];
                    }
                    allErrors[k][a] *= temp;
                }
                //output layer
                else
                {
                    allErrors[k][a] *= (expectedOutputs[a] - allOutputs[k][a]);
                }

                for(int t = 0; t < sizes[k - 1]; t++)
                {
                    int index = getIndex(k - 1, t, k, a);
                    double delta = learningRate * allOutputs[k - 1][t] * allErrors[k][a]
                                   + momentum * lastDeltas[index];

                    setWeight(k - 1, t, k, a, getWeight(k - 1, t, k, a) + delta);
                    lastDeltas[index] = delta;
                }
            }
        }
    }

    /**
     * Gets weight between 2 nodes
     * @param layerStart Starting layer, -1 for bias node
     * @param start Node number in the starting layer
     * @param layerEnd Ending layer, should be one more than the start layer
     * @param end Node number in the ending layer
     * @return The value of the weight
     */
    private double getWeight(int layerStart, int start, int layerEnd, int end)
    {
        int index = getIndex(layerStart, start, layerEnd, end);
        return weights[index];
    }

    /**
     * Sets a new weight between 2 nodes
     * @param layerStart Starting layer, -1 for bias node
     * @param start Node number in the starting layer
     * @param layerEnd Ending layer, should be one more than the start layer
     * @param end Node number in the ending layer
     * @param newWeight New weight between the nodes
     */
    private void setWeight(int layerStart, int start, int layerEnd, int end, double newWeight)
    {
        int index = getIndex(layerStart, start, layerEnd, end);
        weights[index] = newWeight;
    }

    /**
     * Gets the index of the 1-d array that represents the weight between the 2 nodes
     * @param layerStart Starting layer, -1 for bias node
     * @param start Node number in the starting layer
     * @param layerEnd Ending layer, should be one more than the start layer
     * @param end Node number in the ending layer
     * @return The index where the weight is
     */
    private int getIndex(int layerStart, int start, int layerEnd, int end)
    {
        if(layerStart != -1)
        {
            int index = 0;
            for(int k = 0; k < layerStart; k++)
            {
                index += sizes[k] * sizes[k + 1];
            }

            index += start * sizes[layerEnd];
            index += end;

            return index;
        }
        else
        {
            int index = 0;
            for(int k = 0; k < hiddenLayers + 1; k++)
            {
                index += sizes[k] * sizes[k + 1];
            }

            for(int k = 0; k < layerEnd; k++)
            {
                index += sizes[k];
            }

            index += end;

            return index;
        }
    }

    /**
     * Applies the relevant activation function to the sum of a node
     * @param sum The value to plug into the function
     * @return The value returned by applying the function to the value
     */
    private double applyActivationFunction(double sum)
    {
        switch(activationFunction)
        {
            case LINEAR:
                return linearSlope * sum;
            case LOGISTIC:
                return 1.0 / (1.0 + Math.pow(Math.E, sum * -1.0));
        }

        System.out.println("Failed to apply activation function");
        return -9999;
    }

    /**
     * similar to applyActivationFunction, but applies the derivative for testing
     * @param sum the value to plug into the function
     * @return The value returned by applying the function to the value
     */
    private double applyActivationFunctionDerivative(double sum)
    {
        switch(activationFunction)
        {
            case LINEAR:
                return linearSlope;
            case LOGISTIC:
                return (Math.pow(Math.E, sum) / Math.pow(Math.pow(Math.E, sum) + 1, 2));
        }

        System.out.println("Failed to apply activation function");
        return -9999;
    }
}
